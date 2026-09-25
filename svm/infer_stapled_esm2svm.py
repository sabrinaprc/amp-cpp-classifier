"""
Score the stapled peptides with the ESM2 + SVM classifier.
Mirrors infer_stapled_esm2hdc.py: loads the checkpoint trained by
train_esm2_svm_amp_cpp.py instead of retraining on every run -- run that
script first if svm/checkpoints/esm2_svm_amp_cpp_*.pkl doesn't exist yet.

Two-stage pipeline:
    Stage 1 (symbolic OnlineHD, already run): AMP-plausible vs decoy.
        stapled_predictions.csv `pred_label` -- 1 = AMP-plausible.
    Stage 2 (this script): of the stage-1 survivors, poreforming AMP (0)
        vs poreforming CPP (1).

Run:
    conda run -n peptide python svm/train_esm2_svm_amp_cpp.py   # once
    conda run -n peptide python svm/infer_stapled_esm2svm.py
"""
import argparse
import os
import sys
import csv
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))  # sabrina_peptide/svm/
ROOT = os.path.dirname(HERE)                        # sabrina_peptide/
HDC_DIR = os.path.join(ROOT, 'hdc')                # sabrina_peptide/hdc/
for p in (HERE, ROOT, HDC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import esm2_hdc
from train_esm2_svm_amp_cpp import SVM_CKPT, TRAIN_EMB_CACHE

DEVICE       = esm2_hdc.DEVICE
DATA_DIR     = esm2_hdc.DATA_DIR
DEFAULT_INPUT_CSV  = os.path.join(ROOT, 'symbolic_onlinehd_best_bundle', 'predictions', 'stapled_predictions.csv')
DEFAULT_OUTPUT_CSV = os.path.join(HERE, 'predictions', 'stapled_predictions_esm2svm.csv')
SEQUENCE_COL     = "sequence"
STAGE1_LABEL_COL = "pred_label"
CLASS_NAMES  = {0: "AMP_pore", 1: "CPP_pore"}


# ------------------------------------------------------------------ helpers
def print_report(rows, seqs, probs, preds):
    n     = len(preds)
    n_amp = int((preds == 0).sum())
    n_cpp = int((preds == 1).sum())
    p_amp = probs[:, 0]
    p_cpp = probs[:, 1]

    title = "ESM2 + SVM (RBF) RESULTS  |  AMP_pore vs CPP_pore"
    print(f"\n{title}")
    print("=" * len(title))
    print(f" Predicted AMP_pore : {n_amp}  ({100*n_amp/n:.1f}%)")
    print(f" Predicted CPP_pore : {n_cpp}  ({100*n_cpp/n:.1f}%)")
    print(f"\n p_amp_pore  min/mean/max: "
          f"{p_amp.min():.4f} / {p_amp.mean():.4f} / {p_amp.max():.4f}")
    print(f" p_cpp_pore  min/mean/max: "
          f"{p_cpp.min():.4f} / {p_cpp.mean():.4f} / {p_cpp.max():.4f}")

    slen = sorted(len(s) for s in seqs)
    print(f" Seq length  min/med/max : "
          f"{slen[0]} / {slen[len(slen)//2]} / {slen[-1]}")

    order = np.argsort(p_amp)

    print(f"\n Top 5 most CPP_pore-like (lowest p_amp_pore):")
    for i in order[:5]:
        r = rows[i]
        print(f"   p_amp={p_amp[i]:.4f}  p_cpp={p_cpp[i]:.4f}  "
              f"{r.get('DRAMP_ID',''):<12}  {r.get('Name',''):<26}  {seqs[i]}")

    print(f"\n Top 5 most AMP_pore-like (highest p_amp_pore):")
    for i in order[::-1][:5]:
        r = rows[i]
        print(f"   p_amp={p_amp[i]:.4f}  p_cpp={p_cpp[i]:.4f}  "
              f"{r.get('DRAMP_ID',''):<12}  {r.get('Name',''):<26}  {seqs[i]}")


def parse_args():
    p = argparse.ArgumentParser(description="Score stapled peptides with ESM2 + SVM")
    p.add_argument("--input", default=DEFAULT_INPUT_CSV,
                    help="Stage-1 (symbolic OnlineHD) predictions CSV to score")
    p.add_argument("--output", default=DEFAULT_OUTPUT_CSV,
                    help="Where to write the ESM2+SVM predictions CSV")
    p.add_argument("--emb-output", default=None,
                    help="Where to save the ESM2 embeddings .npz (default: "
                         "embeddings/stapled_esm2svm_{input basename}_embeddings.npz, "
                         "tagged so a different --input never clobbers a prior run's cache)")
    return p.parse_args()


def _tag_from_input(input_csv):
    base = os.path.splitext(os.path.basename(input_csv))[0]
    if base.endswith('_predictions'):
        base = base[:-len('_predictions')]
    return base


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    INPUT_CSV, OUTPUT_CSV = args.input, args.output
    print(f"Device: {DEVICE}")

    print("\n" + "=" * 70)
    print("MODEL SETUP  (fixed 634-sequence training set -- NOT your input file)")
    print("=" * 70)

    # 1. load the pre-trained checkpoint (scaler + SVM fit on all 634 sequences)
    if not os.path.exists(SVM_CKPT) or not os.path.exists(TRAIN_EMB_CACHE):
        raise FileNotFoundError(
            f"Missing trained checkpoint/embeddings ({SVM_CKPT}).\n"
            f"Run `python svm/train_esm2_svm_amp_cpp.py` first.")

    print(f"Loading SVM checkpoint: {SVM_CKPT}")
    ckpt = joblib.load(SVM_CKPT)
    svm, scaler = ckpt['svm'], ckpt['scaler']

    y_train = np.load(TRAIN_EMB_CACHE)['y']
    print(f" Training sequences: {y_train.shape[0]}  "
          f"(AMP_pore={int((y_train==0).sum())}, CPP_pore={int((y_train==1).sum())})")

    # 2. load stage-1 predictions and filter to NGC-positive
    with open(INPUT_CSV, newline='') as f:
        reader    = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows      = list(reader)

    if STAGE1_LABEL_COL not in fieldnames:
        raise ValueError(
            f"Expected stage-1 column '{STAGE1_LABEL_COL}' in {INPUT_CSV}. "
            f"Run symbolic_onlinehd_best_bundle/run_stapled.sh first.")

    keep_idx   = [i for i, r in enumerate(rows) if int(r[STAGE1_LABEL_COL]) == 1]
    n_filtered = len(rows) - len(keep_idx)

    print("\n" + "=" * 70)
    print(f"SCORING YOUR INPUT  ({os.path.basename(INPUT_CSV)})")
    print("=" * 70)
    print(f" Rows in input                  : {len(rows)}")
    print(f" Passed stage-1 filter (kept)   : {len(keep_idx)}")
    print(f" Dropped by stage-1 (decoy-like): {n_filtered}")

    seqs = [rows[i][SEQUENCE_COL] for i in keep_idx]

    # 3. embed with ESM2
    if not hasattr(esm2_hdc, 'tokenizer') or esm2_hdc.tokenizer is None:
        esm2_hdc.tokenizer, esm2_hdc.model = esm2_hdc.load_model(esm2_hdc.REPO_ID)
    X_staple = esm2_hdc.extract_embeddings(esm2_hdc.get_embedding, seqs, "stapled")

    # 4. predict with SVM
    Xst   = scaler.transform(X_staple)
    probs = svm.predict_proba(Xst)   # shape (n, 2): col0=AMP_pore, col1=CPP_pore
    distances = svm.decision_function(Xst)
    preds = probs.argmax(axis=1)

    scored_rows = [rows[i] for i in keep_idx]

    # 5. write output CSV
    new_cols = ["esm2svm_p_amp_pore", "esm2svm_p_cpp_pore",
            "esm2svm_pred_label", "esm2svm_pred_class", "esm2svm_svm_distance"]
    out_fields = list(fieldnames) + [c for c in new_cols if c not in fieldnames]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for idx, (r, p, lab) in enumerate(zip(scored_rows, probs, preds)):
            r = dict(r)
            r["esm2svm_p_amp_pore"] = f"{p[0]:.6f}"
            r["esm2svm_p_cpp_pore"] = f"{p[1]:.6f}"
            r["esm2svm_pred_label"] = int(lab)
            r["esm2svm_pred_class"] = CLASS_NAMES[int(lab)]
            r["esm2svm_svm_distance"] = f"{distances[idx]:.6f}"
            w.writerow(r)

    print(f"\n Wrote predictions -> {OUTPUT_CSV}  ({len(scored_rows)} rows)")

    # 6. save embeddings for t-SNE visualization
    emb_out = args.emb_output or os.path.join(
        ROOT, 'embeddings', f'stapled_esm2svm_{_tag_from_input(INPUT_CSV)}_embeddings.npz')
    os.makedirs(os.path.dirname(emb_out), exist_ok=True)
    np.savez(emb_out,
             X=X_staple,
             sequences=np.array(seqs),
             pred_class=np.array([CLASS_NAMES[int(p)] for p in preds]),
             p_amp_pore=probs[:, 0],
             p_cpp_pore=probs[:, 1],
             dramp_id=np.array([r.get('DRAMP_ID', '') for r in scored_rows]))
    print(f" Saved ESM2 embeddings -> {emb_out}  "
          f"({len(seqs)} sequences, dim={X_staple.shape[1]})")

    # 7. print summary report
    print_report(scored_rows, seqs, probs, preds)


if __name__ == '__main__':
    main()
