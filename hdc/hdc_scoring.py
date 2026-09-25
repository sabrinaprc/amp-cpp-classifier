"""
Compute a single combined HDC "axis score" per stapled peptide -- the HDC
equivalent of the SVM decision-function distance.

Instead of two separate class-hypervector similarities (cos_amp, cos_cpp),
this builds one combined axis vector pointing from the AMP class hypervector
toward the CPP class hypervector, then projects each peptide's hypervector
onto that axis via cosine similarity.

    axis_hv = cpp_class_hv - amp_class_hv
    score   = cosine_similarity(peptide_hv, axis_hv)

Interpretation:
    score > 0  -> peptide sits on the CPP side of the axis
    score < 0  -> peptide sits on the AMP side of the axis
    score = 0  -> peptide sits exactly on the AMP/CPP boundary

This mirrors esm2svm_svm_distance -- same sign convention (negative=AMP,
positive=CPP), same "rank, don't threshold" interpretation, just computed
from the HDC's own hypervector space instead of the SVM's hyperplane.

Can be run standalone (reads a CSV with a `sequence` column, appends the
score) or imported and called directly from infer_stapled_esm2hdc.py right
after prediction, passing the already-loaded model/norm/device so nothing
gets loaded twice.

Standalone run:
    conda run -n peptide python hdc/hdc_scoring.py
"""
import argparse
import os
import sys
import csv
import numpy as np
import onlinehd                       # noqa: F401  (needed to unpickle the checkpoint)
import torch
from sklearn.preprocessing import Normalizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import esm2_hdc
from train_esm2_hdc_amp_cpp import HDC_CKPT, TRAIN_EMB_CACHE

DEVICE = esm2_hdc.DEVICE
DEFAULT_INPUT_CSV = os.path.join(HERE, 'predictions', 'stapled_predictions_esm2hdc.csv')
SEQUENCE_COL      = "sequence"
COS_AMP_COL       = "esm2hdc_cos_amp"
COS_CPP_COL       = "esm2hdc_cos_cpp"
AXIS_SCORE_COL    = "esm2hdc_axis_score"


def compute_axis_score(seqs, model, norm, device):
    """Encode sequences with ESM2, score against the AMP<->CPP HDC axis.

    Uses model.scores()'s per-class-normalized cosine similarity (the same
    computation the model's actual prediction is based on: cos(x, amp_hv)
    and cos(x, cpp_hv) each normalized by their own class hypervector's
    norm), not cosine similarity onto the raw (cpp_hv - amp_hv) difference
    vector. The latter shares a single denominator across both classes, so
    if amp_hv and cpp_hv don't have equal norms (they generally don't) its
    sign can disagree with the model's real decision for peptides near the
    boundary -- exactly the failure mode this replaces.

    Returns (cos_amp, cos_cpp, axis_score) -- the two raw per-class cosine
    similarities as well as their difference (axis_score = cos_cpp - cos_amp,
    the same signed value plotting code already reads: negative = AMP side,
    positive = CPP side). Nothing here is shifted or clipped to force a
    particular sign; cos_amp/cos_cpp are each naturally in [-1, 1] since
    they're genuine cosine similarities.
    """
    if not hasattr(esm2_hdc, 'tokenizer') or esm2_hdc.tokenizer is None:
        esm2_hdc.tokenizer, esm2_hdc.model = esm2_hdc.load_model(esm2_hdc.REPO_ID)

    X = esm2_hdc.extract_embeddings(esm2_hdc.get_embedding, seqs, "stapled")
    Xn = norm.transform(X)
    Xt = torch.from_numpy(Xn).float().to(device)

    with torch.no_grad():
        scores = model.scores(Xt)  # (n, 2): col0=AMP_pore, col1=CPP_pore
        cos_amp = scores[:, 0].cpu().numpy()
        cos_cpp = scores[:, 1].cpu().numpy()
        axis_score = cos_cpp - cos_amp

    return cos_amp, cos_cpp, axis_score


def add_axis_score(input_csv, output_csv, model=None, norm=None, device=None):
    """Read a predictions CSV, append esm2hdc_cos_amp/esm2hdc_cos_cpp/
    esm2hdc_axis_score, write it back out.

    If model/norm/device are already loaded (e.g. by infer_stapled_esm2hdc.py),
    pass them in directly to avoid reloading the checkpoint and refitting the
    normalizer a second time.
    """
    device = device or DEVICE

    if model is None:
        if not os.path.exists(HDC_CKPT) or not os.path.exists(TRAIN_EMB_CACHE):
            raise FileNotFoundError(
                f"Missing trained checkpoint/embeddings ({HDC_CKPT}).\n"
                f"Run `python hdc/train_esm2_hdc_amp_cpp.py` first.")
        model = torch.load(HDC_CKPT, map_location=device, weights_only=False)

    if norm is None:
        X_train = np.load(TRAIN_EMB_CACHE)['X']
        norm = Normalizer(norm='l2').fit(X_train)

    with open(input_csv, newline='') as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows       = list(reader)

    if SEQUENCE_COL not in fieldnames:
        raise ValueError(f"Expected '{SEQUENCE_COL}' column in {input_csv}")

    seqs = [r[SEQUENCE_COL] for r in rows]
    print(f"  Scoring {len(seqs)} sequences for HDC axis score...")

    cos_amp, cos_cpp, axis_score = compute_axis_score(seqs, model, norm, device)

    print(f"  cos_amp     min/mean/max: "
          f"{cos_amp.min():.4f} / {cos_amp.mean():.4f} / {cos_amp.max():.4f}")
    print(f"  cos_cpp     min/mean/max: "
          f"{cos_cpp.min():.4f} / {cos_cpp.mean():.4f} / {cos_cpp.max():.4f}")
    print(f"  axis_score  min/mean/max: "
          f"{axis_score.min():.4f} / {axis_score.mean():.4f} / {axis_score.max():.4f}")
    print("    (negative = AMP side, positive = CPP side, 0 = boundary)")

    new_cols = [COS_AMP_COL, COS_CPP_COL, AXIS_SCORE_COL]
    out_fields = list(fieldnames) + [c for c in new_cols if c not in fieldnames]

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for idx, r in enumerate(rows):
            r = dict(r)
            r[COS_AMP_COL] = f"{cos_amp[idx]:.6f}"
            r[COS_CPP_COL] = f"{cos_cpp[idx]:.6f}"
            r[AXIS_SCORE_COL] = f"{axis_score[idx]:.6f}"
            w.writerow(r)

    print(f"  Done -> {output_csv}  ({len(rows)} rows, {', '.join(new_cols)} added)")

    order = np.argsort(axis_score)
    print("\n  Top 5 most AMP-side (most negative axis_score):")
    for i in order[:5]:
        r = rows[i]
        print(f"     {axis_score[i]:.4f}  {r.get('DRAMP_ID',''):<12}"
              f"{r.get('Name',''):<26}  {seqs[i]}")
    print("\n  Top 5 most CPP-side (most positive axis_score):")
    for i in order[::-1][:5]:
        r = rows[i]
        print(f"     {axis_score[i]:.4f}  {r.get('DRAMP_ID',''):<12}"
              f"{r.get('Name',''):<26}  {seqs[i]}")

    return axis_score


def parse_args():
    p = argparse.ArgumentParser(description="Compute combined HDC axis score")
    p.add_argument("--input", default=DEFAULT_INPUT_CSV,
                   help="Predictions CSV with a 'sequence' column")
    p.add_argument("--output", default=None,
                   help="Where to write the CSV with the axis score added "
                        "(default: overwrite --input)")
    return p.parse_args()


def main():
    args = parse_args()
    add_axis_score(input_csv=args.input, output_csv=args.output or args.input)


if __name__ == '__main__':
    main()
    

