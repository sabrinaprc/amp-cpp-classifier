"""
Train and save the ESM2 + OnlineHD classifier: poreforming AMP (label 0) vs
poreforming CPP (label 1).

Run once (or again only if the training data or hyperparameters below change):
    conda run -n peptide python hdc/train_esm2_hdc_amp_cpp.py

infer_stapled_esm2hdc.py loads the checkpoint this writes instead of retraining
on every run -- training here is fully deterministic (fixed data, frozen ESM2,
fixed seed), so refitting each time was pure wasted work.

Best config (reports_epoch_sweep/esm2_t12_35M_UR50D_HDC_grid_*.csv):
    dim=5000, lr=0.01, epochs=400, normalizer=l2   (5-fold acc 0.938, AUC 0.945)

Before the final full-data fit, re-runs a 5-fold CV at this exact config (same
StratifiedKFold seed as the original grid search) so the printed accuracy/MCC/
confusion matrix confirm this environment reproduces the numbers above.

Saves:
    embeddings/esm2_t12_35M_UR50D_poreforming_amp_cpp.npz  (ESM2 embeddings)
    hdc/checkpoints/esm2_hdc_amp_vs_cpp_d{dim}_lr{lr}_ep{epochs}_seed{seed}.pt  (fitted model)
    hdc/reports/esm2_hdc_amp_vs_cpp_<timestamp>.txt                  (CV report)
"""
import os
import sys
import numpy as np
import torch
import onlinehd
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import Normalizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import esm2_hdc                       # reuse the existing ESM2+HDC pipeline module
from helpers import load_sequences, run_kfold_hdc, save_report

DEVICE = esm2_hdc.DEVICE
DATA_DIR = esm2_hdc.DATA_DIR

# Best HDC config (from the grid search reports)
HDC_DIM = 5000
HDC_LR = 0.01
HDC_EPOCHS = 400
SEED = 42

# Scale of the encoder's random projection basis (onlinehd.Encoder.basis,
# normally ~N(0,1)): sigma<1 gives a wider/more inclusive kernel, sigma>1 a
# narrower/more exclusive one. Applied here (not by editing the installed
# onlinehd package) because torch.save(model, ...) below saves the whole
# model object, encoder basis included -- so this is fully captured in the
# checkpoint and every downstream script that loads it just works.
HDC_SIGMA = 1.0

CACHE_DIR = os.path.join(ROOT, 'embeddings')
CKPT_DIR = os.path.join(HERE, 'checkpoints')
TRAIN_EMB_CACHE = os.path.join(CACHE_DIR, 'esm2_t12_35M_UR50D_poreforming_amp_cpp.npz')
# Only tag the filename with sigma when it deviates from the default --
# keeps the existing sigma=1.0 checkpoint (trained before HDC_SIGMA existed)
# resolvable under its original name instead of orphaning it.
_sigma_tag = '' if HDC_SIGMA == 1.0 else f'_sigma{HDC_SIGMA}'
HDC_CKPT = os.path.join(
    CKPT_DIR, f'esm2_hdc_amp_vs_cpp_d{HDC_DIM}_lr{HDC_LR}_ep{HDC_EPOCHS}_seed{SEED}{_sigma_tag}.pt')


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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    X_train, y_train = get_train_embeddings()
    print(f"Training set: AMP_pore={int((y_train == 0).sum())}  "
          f"CPP_pore={int((y_train == 1).sum())}")

    # ---- confirm accuracy: 5-fold CV at the final config before the full-data fit ----
    fold_accs, y_test_cv, y_pred_cv = run_kfold_hdc(
        X_train, y_train, 'esm2_hdc_amp_vs_cpp',
        n_folds=5, dim=HDC_DIM, lr=HDC_LR, epochs=HDC_EPOCHS, normalizer='l2')

    mcc = matthews_corrcoef(y_test_cv, y_pred_cv)
    try:
        auc = roc_auc_score(y_test_cv, y_pred_cv)
    except ValueError:
        auc = float('nan')
    print(f"\n5-fold CV results (dim={HDC_DIM}, lr={HDC_LR}, epochs={HDC_EPOCHS}, l2):")
    print(f"  Mean accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  MCC:           {mcc:.4f}")
    print(f"  AUC:           {auc:.4f}")
    print("\nClassification report (pooled across folds):")
    print(classification_report(y_test_cv, y_pred_cv, target_names=['AMP_pore', 'CPP_pore']))
    print("Confusion matrix (rows=actual, cols=predicted; [AMP_pore, CPP_pore]):")
    print(confusion_matrix(y_test_cv, y_pred_cv))

    save_report(fold_accs, y_test_cv, y_pred_cv, 'esm2_hdc_amp_vs_cpp',
                extra_info={"ESM2 model": esm2_hdc.REPO_ID, "Device": DEVICE,
                            "HDC dim": HDC_DIM, "HDC lr": HDC_LR,
                            "HDC epochs": HDC_EPOCHS,
                            "HDC sigma (final checkpoint only -- CV above always uses sigma=1.0)": HDC_SIGMA,
                            "Normalizer": "l2",
                            "Seed": SEED, "AUC": f"{auc:.4f}"},
                out_dir=os.path.join(HERE, 'reports'))

    # ---- final fit on the FULL labeled set (this is what gets checkpointed and
    # used to score the stapled candidates -- the CV above is only a sanity check) ----
    print(f"\nTraining final OnlineHD on full data "
          f"(dim={HDC_DIM}, lr={HDC_LR}, epochs={HDC_EPOCHS}, l2)...")
    Xtr = Normalizer(norm='l2').fit(X_train).transform(X_train)
    model = onlinehd.OnlineHD(2, Xtr.shape[1], dim=HDC_DIM)
    if HDC_SIGMA != 1.0:
        print(f"Scaling encoder basis by sigma={HDC_SIGMA}")
        model.encoder.basis = torch.randn(HDC_DIM, Xtr.shape[1]) * HDC_SIGMA
    model = model.to(DEVICE)
    model.fit(torch.from_numpy(Xtr).float().to(DEVICE),
              torch.from_numpy(y_train).long().to(DEVICE),
              bootstrap=1.0, lr=HDC_LR, epochs=HDC_EPOCHS)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model, HDC_CKPT)
    print(f"\nSaved OnlineHD checkpoint -> {HDC_CKPT}")


if __name__ == '__main__':
    main()
