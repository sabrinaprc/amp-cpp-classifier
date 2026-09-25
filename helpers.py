import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix, accuracy_score
from datetime import datetime

def load_esm2_model(repo_id, device=None):
    """Canonical ESM2 tokenizer/model loader -- single source of truth so
    every script (hdc/esm2_hdc.py, esm2_svm.py, extract_embeddings.py,
    svm/infer_stapled_esm2svm.py, ...) embeds sequences identically."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM2: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id)
    model.to(device)
    model.eval()
    print(f"Embedding dimension: {model.config.hidden_size}")
    return tokenizer, model

def get_esm2_embedding(sequence, tokenizer, model):
    """Mean-pooled ESM2 embedding for a single sequence. Device is inferred
    from the model itself so callers don't need to track/pass a DEVICE."""
    device = next(model.parameters()).device
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

def load_sequences(filepath):
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                sequences.append(parts[1] if len(parts) > 1 else parts[0])
    return sequences

def extract_embeddings(get_embedding_fn, sequences, label):
    embeddings = []
    for i, seq in enumerate(sequences):
        embeddings.append(get_embedding_fn(seq))
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(sequences)}")
    return np.array(embeddings)

def run_kfold(X, y, model_name, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    all_y_test = []
    all_y_pred = []
    print(f"Running {n_folds}-fold cross validation...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"  Fold {fold+1} accuracy: {acc:.3f}")
    print(f"\nMean accuracy: {np.mean(fold_accuracies):.3f}")
    print(f"Std deviation: {np.std(fold_accuracies):.3f}")
    return fold_accuracies, np.array(all_y_test), np.array(all_y_pred)

def run_kfold_hdc(X, y, model_name, n_folds=5, dim=10000, lr=0.035, epochs=20,
                  normalizer='unit'):
    import torch
    import onlinehd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    all_y_test = []
    all_y_pred = []
    classes = len(np.unique(y))
    features = X.shape[1]
    print(f"Running {n_folds}-fold CV with OnlineHD "
          f"(dim={dim}, lr={lr}, epochs={epochs}, normalizer={normalizer})...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if normalizer in ('l2', 'unit'):
            scaler = Normalizer(norm='l2').fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        elif normalizer == 'standard':
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        elif normalizer == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalizer: {normalizer}")
        X_train_t = torch.from_numpy(X_train).float().to(device)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_train_t = torch.from_numpy(y_train).long().to(device)
        model = onlinehd.OnlineHD(classes, features, dim=dim).to(device)
        model.fit(X_train_t, y_train_t, bootstrap=1.0, lr=lr, epochs=epochs)
        y_pred = model(X_test_t).cpu().numpy()
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"  Fold {fold+1} accuracy: {acc:.3f}")
    print(f"\nMean accuracy: {np.mean(fold_accuracies):.3f}")
    print(f"Std deviation: {np.std(fold_accuracies):.3f}")
    return fold_accuracies, np.array(all_y_test), np.array(all_y_pred)

def hdc_grid_search(X, y, model_name,
                    dims=(5000, 10000, 15000),
                    lrs=(0.01, 0.035, 0.1),
                    epochs_list=(20, 50, 100),
                    normalizer='unit'):
    from itertools import product
    best_mcc = -1
    best_params = {}
    results = []
    total = len(dims) * len(lrs) * len(epochs_list)
    i = 0
    for dim, lr, ep in product(dims, lrs, epochs_list):
        i += 1
        print(f"\n=== [{i}/{total}] dim={dim}, lr={lr}, epochs={ep} ===")
        fold_accs, y_test, y_pred = run_kfold_hdc(
            X, y, model_name, n_folds=5,
            dim=dim, lr=lr, epochs=ep, normalizer=normalizer)
        mcc = matthews_corrcoef(y_test, y_pred)
        acc = float(np.mean(fold_accs))
        std = float(np.std(fold_accs))
        results.append({'dim': dim, 'lr': lr, 'epochs': ep,
                        'mcc': mcc, 'acc': acc, 'std': std})
        print(f"  MCC: {mcc:.3f}  Acc: {acc:.3f}")
        if mcc > best_mcc:
            best_mcc = mcc
            best_params = {'dim': dim, 'lr': lr, 'epochs': ep}
    print(f"\nBest: {best_params} -> MCC {best_mcc:.3f}")
    return results, best_params

def save_grid_csv(results, model_name, out_dir="reports"):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/{model_name}_grid_{ts}.csv"
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Grid CSV saved to {path}")

def save_report(fold_accuracies, all_y_test, all_y_pred, model_name, extra_info={}, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{out_dir}/{model_name}_{timestamp}.txt"
    report = classification_report(all_y_test, all_y_pred, target_names=['AMP_pore', 'CPP_pore'])
    mcc = matthews_corrcoef(all_y_test, all_y_pred)
    cm = confusion_matrix(all_y_test, all_y_pred)
    with open(filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, val in extra_info.items():
            f.write(f"{key}: {val}\n")
        f.write(f"Dataset: AMP_pore={sum(all_y_test==0)} CPP_pore={sum(all_y_test==1)}\n")
        f.write(f"Cross Validation: {len(fold_accuracies)}-fold\n")
        f.write("="*50 + "\n\n")
        for i, acc in enumerate(fold_accuracies):
            f.write(f"Fold {i+1} accuracy: {acc:.3f}\n")
        f.write(f"\nMean accuracy: {np.mean(fold_accuracies):.3f}\n")
        f.write(f"Std deviation: {np.std(fold_accuracies):.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nMCC: {mcc:.3f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                Predicted AMP  Predicted CPP\n")
        f.write(f"Actual AMP         {cm[0][0]}              {cm[0][1]}\n")
        f.write(f"Actual CPP         {cm[1][0]}              {cm[1][1]}\n")
    print(f"Report saved to {filename}")