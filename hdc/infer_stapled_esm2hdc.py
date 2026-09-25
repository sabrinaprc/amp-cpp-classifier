"""
Score the stapled peptides (from the symbolic-OnlineHD stapled_predictions.csv)
with the repo's ESM2 + OnlineHD classifier.
This is a thin inference wrapper: it reuses the existing pipeline code rather than
re-implementing it --
    - load_model / get_embedding   <- hdc/esm2_hdc.py   (ESM2 mean-pool embedding)

Two-stage pipeline:
    Stage 1 (symbolic OnlineHD, upstream, already run): AMP-plausible vs decoy.
        stapled_predictions.csv's `pred_label` column -- 1 = AMP-plausible.
    Stage 2 (this script): of the stage-1 survivors, poreforming AMP (label 0)
        vs poreforming CPP (label 1)

Loads the checkpoint trained by train_esm2_hdc_amp_cpp.py -- run that first if
hdc/checkpoints/esm2_hdc_amp_vs_cpp_*.pt doesn't exist yet.

Output only contains the stage-1 survivors that actually got scored (186, not
the full 365) -- rows stage 1 filtered out as decoy-like aren't written at all.

Appends to each row:
    esm2hdc_p_amp_pore  : OnlineHD probability of class 0 (poreforming AMP)
    esm2hdc_p_cpp_pore  : OnlineHD probability of class 1 (poreforming CPP)
    esm2hdc_pred_label  : argmax label (0=AMP_pore, 1=CPP_pore)
    esm2hdc_pred_class  : "AMP_pore" or "CPP_pore"

For the combined AMP<->CPP axis score (esm2hdc_axis_score), run
hdc_scoring.py afterward -- it reads this script's output CSV and appends
that one column, kept separate so this script stays focused on prediction.

Run:
    conda run -n peptide python hdc/train_esm2_hdc_amp_cpp.py   # once
    conda run -n peptide python hdc/infer_stapled_esm2hdc.py
    conda run -n peptide python hdc/hdc_scoring.py               # adds axis score
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
import hdc_scoring

DEVICE = esm2_hdc.DEVICE
DEFAULT_INPUT_CSV  = os.path.join(ROOT, 'symbolic_onlinehd_best_bundle', 'predictions', 'stapled_predictions.csv')
DEFAULT_OUTPUT_CSV = os.path.join(HERE, 'predictions', 'stapled_predictions_esm2hdc.csv')
SEQUENCE_COL       = "sequence"
STAGE1_LABEL_COL   = "pred_label"
CLASS_NAMES        = {0: "AMP_pore", 1: "CPP_pore"}


def parse_args():
    p = argparse.ArgumentParser(description="Score stapled peptides with ESM2 + OnlineHD")
    p.add_argument("--input",  default=DEFAULT_INPUT_CSV)
    p.add_argument("--output", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--emb-output", default=None,
                   help="Where to save the ESM2 embeddings .npz (default: "
                        "embeddings/stapled_esm2_{input basename}_embeddings.npz, "
                        "tagged so a different --input never clobbers a prior run's cache)")
    return p.parse_args()


def _tag_from_input(input_csv):
    base = os.path.splitext(os.path.basename(input_csv))[0]
    if base.endswith('_predictions'):
        base = base[:-len('_predictions')]
    return base


def main():
    args = parse_args()
    INPUT_CSV, OUTPUT_CSV = args.input, args.output

    print(f"Device: {DEVICE}")
    if not os.path.exists(HDC_CKPT) or not os.path.exists(TRAIN_EMB_CACHE):
        raise FileNotFoundError(
            f"Missing trained checkpoint/embeddings ({HDC_CKPT}).\n"
            f"Run `python hdc/train_esm2_hdc_amp_cpp.py` first.")

    print(f"Loading OnlineHD checkpoint: {HDC_CKPT}")
    model = torch.load(HDC_CKPT, map_location=DEVICE, weights_only=False)

    X_train = np.load(TRAIN_EMB_CACHE)['X']
    norm = Normalizer(norm='l2').fit(X_train)

    # ---- load input + stage-1 filter ----
    with open(INPUT_CSV, newline='') as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows       = list(reader)

    if STAGE1_LABEL_COL not in fieldnames:
        raise ValueError(
            f"Expected stage-1 column '{STAGE1_LABEL_COL}' in {INPUT_CSV} "
            f"(run symbolic_onlinehd_best_bundle/run_stapled.sh first).")

    keep_idx   = [i for i, r in enumerate(rows) if int(r[STAGE1_LABEL_COL]) == 1]
    n_filtered = len(rows) - len(keep_idx)
    print(f"Stapled candidates: {len(rows)}  "
          f"(stage-1 AMP-plausible: {len(keep_idx)}, "
          f"filtered out as decoy-like: {n_filtered})")

    seqs = [rows[i][SEQUENCE_COL] for i in keep_idx]

    # ---- embed with ESM2 ----
    esm2_hdc.tokenizer, esm2_hdc.model = esm2_hdc.load_model(esm2_hdc.REPO_ID)
    X_staple = esm2_hdc.extract_embeddings(esm2_hdc.get_embedding, seqs, "stapled")
    Xst = norm.transform(X_staple)
    Xst_tensor = torch.from_numpy(Xst).float().to(DEVICE)

    # ---- HDC predictions ----
    with torch.no_grad():
        probs = model.probabilities(Xst_tensor).cpu().numpy()

    preds = probs.argmax(axis=1)
    scored_rows = [rows[i] for i in keep_idx]

    # ---- write output CSV ----
    new_cols = ["esm2hdc_p_amp_pore", "esm2hdc_p_cpp_pore",
                "esm2hdc_pred_label", "esm2hdc_pred_class"]
    out_fields = list(fieldnames) + [c for c in new_cols if c not in fieldnames]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for idx, (r, p, lab) in enumerate(zip(scored_rows, probs, preds)):
            r = dict(r)
            r["esm2hdc_p_amp_pore"] = f"{p[0]:.6f}"
            r["esm2hdc_p_cpp_pore"] = f"{p[1]:.6f}"
            r["esm2hdc_pred_label"] = int(lab)
            r["esm2hdc_pred_class"] = CLASS_NAMES[int(lab)]
            w.writerow(r)

    print(f"\nDone -> {OUTPUT_CSV}  ({len(scored_rows)} rows)")

    # ---- append the combined AMP<->CPP axis score ----
    print("\nComputing combined HDC axis score...")
    hdc_scoring.add_axis_score(
        input_csv=OUTPUT_CSV, output_csv=OUTPUT_CSV,
        model=model, norm=norm, device=DEVICE)

    # ---- save embeddings for t-SNE ----
    emb_out = args.emb_output or os.path.join(
        ROOT, 'embeddings', f'stapled_esm2_{_tag_from_input(INPUT_CSV)}_embeddings.npz')
    os.makedirs(os.path.dirname(emb_out), exist_ok=True)
    np.savez(emb_out,
             X=X_staple,
             sequences=np.array(seqs),
             pred_class=np.array([CLASS_NAMES[int(p)] for p in preds]),
             dramp_id=np.array([r.get('DRAMP_ID', '') for r in scored_rows]))
    print(f"Saved ESM2 embeddings -> {emb_out}  "
          f"({len(seqs)} sequences, dim={X_staple.shape[1]})")

    print_report(scored_rows, seqs, probs, preds,
                 n_total=len(rows), n_filtered=n_filtered)


def print_report(rows, seqs, probs, preds, n_total, n_filtered):
    n     = len(preds)
    n_amp = int((preds == 0).sum())
    n_cpp = int((preds == 1).sum())
    p_amp = probs[:, 0]
    slen  = sorted(len(s) for s in seqs)

    title = "ESM2 + OnlineHD  |  stapled peptides  |  AMP_pore vs CPP_pore"
    print(f"\n{title}")
    print("=" * len(title))
    print(f" Stapled candidates (total)       : {n_total}")
    print(f" Filtered by stage 1 (decoy-like) : {n_filtered}")
    print(f" Scored by stage 2                : {n}")
    print(f" Predicted AMP_pore : {n_amp}  ({100*n_amp/n:.1f}%)")
    print(f" Predicted CPP_pore : {n_cpp}  ({100*n_cpp/n:.1f}%)")
    print(f"\n p_amp_pore  min/mean/max: "
          f"{p_amp.min():.4f} / {p_amp.mean():.4f} / {p_amp.max():.4f}")
    print(f"   (band width {p_amp.max()-p_amp.min():.4f} -> rank, don't threshold-read)")
    print(f" Seq length  min/med/max : {slen[0]} / {slen[len(slen)//2]} / {slen[-1]}")

    order = np.argsort(p_amp)

    print(f"\n Top 5 most CPP_pore-like (lowest p_amp_pore):")
    for i in order[:5]:
        r = rows[i]
        print(f"   p_amp={p_amp[i]:.4f}  "
              f"{r.get('DRAMP_ID',''):<12}  {r.get('Name',''):<26}  {seqs[i]}")

    print(f"\n Top 5 most AMP_pore-like (highest p_amp_pore):")
    for i in order[::-1][:5]:
        r = rows[i]
        print(f"   p_amp={p_amp[i]:.4f}  "
              f"{r.get('DRAMP_ID',''):<12}  {r.get('Name',''):<26}  {seqs[i]}")


if __name__ == '__main__':
    main()
    