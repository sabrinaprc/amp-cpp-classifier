"""
Train and save the ESM2 + SVM (RBF) classifier: poreforming AMP (label 0) vs
poreforming CPP (label 1).

Run once (or again only if the training data or hyperparameters below change):
    conda run -n peptide python svm/train_esm2_svm_amp_cpp.py

infer_stapled_esm2svm.py loads the checkpoint this writes instead of retraining
on every run. The base SVM fit was already deterministic (fixed data, frozen
ESM2, no randomness in the RBF solver), but `probability=True`'s Platt-scaling
calibration shuffles data internally for its own CV -- without a fixed
random_state that let p_amp_pore/p_cpp_pore drift slightly run-to-run even
though the hard AMP_pore/CPP_pore call never did. Pinning random_state here
fixes that, and checkpointing means it only needs to happen once.

Config: kernel=rbf, C=1.0, gamma=scale (matches helpers.run_kfold / esm2_svm.py)

Before the final full-data fit, re-runs a 5-fold CV at this exact config (same
StratifiedKFold seed as esm2_svm.py) so the printed accuracy/MCC/AUC confirm
this environment reproduces the saved report (mean ~0.953, MCC ~0.906).

Saves:
    embeddings/esm2_t12_35M_UR50D_poreforming_amp_cpp.npz  (ESM2 embeddings)
    svm/checkpoints/esm2_svm_amp_cpp_rbf_C1.0_seed42.pkl   (fitted scaler + model)
    svm/reports/esm2_svm_amp_cpp_<timestamp>.txt            (CV report)
"""
import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))  # sabrina_peptide/svm/
ROOT = os.path.dirname(HERE)                        # sabrina_peptide/
HDC_DIR = os.path.join(ROOT, 'hdc')                  # sabrina_peptide/hdc/
for p in (HERE, ROOT, HDC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import esm2_hdc                       # reuse the existing ESM2 embedding pipeline
from helpers import load_sequences, run_kfold, save_report

DEVICE = esm2_hdc.DEVICE
DATA_DIR = esm2_hdc.DATA_DIR

SEED = 42
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'

CACHE_DIR = os.path.join(ROOT, 'embeddings')
CKPT_DIR = os.path.join(HERE, 'checkpoints')
TRAIN_EMB_CACHE = os.path.join(CACHE_DIR, 'esm2_t12_35M_UR50D_poreforming_amp_cpp.npz')
SVM_CKPT = os.path.join(CKPT_DIR, f'esm2_svm_amp_cpp_{SVM_KERNEL}_C{SVM_C}_seed{SEED}.pkl')


def get_train_embeddings():
    """Poreforming AMP (0) vs CPP (1) ESM2 embeddings -- cached after first run."""
    if os.path.exists(TRAIN_EMB_CACHE):
        print(f"Loading cached training embeddings: {TRAIN_EMB_CACHE}")
        d = np.load(TRAIN_EMB_CACHE)
        return d['X'], d['y']

    print("Extracting training embeddings (runs once, cached after)...")
    esm2_hdc.tokenizer, esm2_hdc.model = esm2_hdc.load_model(esm2_hdc.REPO_ID)
    amp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_AMPS.txt'))
    cpp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_CPPs.txt'))
    print(f"  Poreforming_AMPS: {len(amp_seqs)}  Poreforming_CPPs: {len(cpp_seqs)}")
    amp_emb = esm2_hdc.extract_embeddings(esm2_hdc.get_embedding, amp_seqs, "AMP")
    cpp_emb = esm2_hdc.extract_embeddings(esm2_hdc.get_embedding, cpp_seqs, "CPP")
    X = np.vstack([amp_emb, cpp_emb])
    y = np.array([0] * len(amp_seqs) + [1] * len(cpp_seqs))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(TRAIN_EMB_CACHE, X=X, y=y)
    print(f"  Cached training embeddings -> {TRAIN_EMB_CACHE}")
    return X, y


def main():
    print(f"Device: {DEVICE}")
    X_train, y_train = get_train_embeddings()
    print(f"Training set: AMP_pore={int((y_train == 0).sum())}  "
          f"CPP_pore={int((y_train == 1).sum())}")

    # ---- confirm accuracy: 5-fold CV at the final config before the full-data fit ----
    fold_accs, y_test_cv, y_pred_cv = run_kfold(
        X_train, y_train, 'esm2_svm_amp_cpp', n_folds=5)

    mcc = matthews_corrcoef(y_test_cv, y_pred_cv)
    try:
        auc = roc_auc_score(y_test_cv, y_pred_cv)
    except ValueError:
        auc = float('nan')
    print(f"\n5-fold CV results (kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA}):")
    print(f"  Mean accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  MCC:           {mcc:.4f}")
    print(f"  AUC:           {auc:.4f}")

    save_report(fold_accs, y_test_cv, y_pred_cv, 'esm2_svm_amp_cpp',
                extra_info={"ESM2 model": esm2_hdc.REPO_ID, "Device": DEVICE,
                            "Kernel": SVM_KERNEL, "C": SVM_C, "Gamma": SVM_GAMMA,
                            "Seed": SEED, "AUC": f"{auc:.4f}"},
                out_dir=os.path.join(HERE, 'reports'))

    # ---- final fit on the FULL labeled set (this is what gets checkpointed and
    # used to score the stapled candidates -- the CV above is only a sanity check) ----
    print(f"\nTraining final SVM on full data "
          f"(kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA}, seed={SEED})...")
    scaler = StandardScaler().fit(X_train)
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True,
              random_state=SEED)
    svm.fit(scaler.transform(X_train), y_train)

    os.makedirs(CKPT_DIR, exist_ok=True)
    joblib.dump({'scaler': scaler, 'svm': svm, 'seed': SEED}, SVM_CKPT)
    print(f"\nSaved SVM checkpoint -> {SVM_CKPT}")


if __name__ == '__main__':
    main()
