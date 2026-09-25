# amp-cpp-classifier

ESM2-embedding-based classifiers for **poreforming AMP vs. poreforming CPP** character
in short peptides — two independent implementations trained on the same 634-sequence,
class-balanced dataset (AMP_pore=317, CPP_pore=317):

- **`svm/`** — ESM2 (mean-pooled) + RBF-kernel SVM
- **`hdc/`** — ESM2 (mean-pooled) + OnlineHD (hyperdimensional computing) classifier

Both are meant to run as **Stage 2** of a two-stage pipeline: Stage 1 (upstream, not
included here) is any AMP-plausible-vs-decoy screen that produces a `pred_label` column
(1 = plausible AMP, kept; 0 = decoy-like, dropped) alongside a `sequence` column. These
scripts score the Stage-1 survivors for AMP_pore (label 0) vs CPP_pore (label 1)
character.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. `onlinehd` is only needed for the `hdc/` scripts; if
`pip install onlinehd` doesn't resolve, install from source:
`pip install git+https://gitlab.com/biaslab/onlinehd`.

## What's included

Both classifiers ship **already trained** — the checkpoints are in this repo, so
inference works immediately, no training required first:

- `svm/checkpoints/esm2_svm_amp_cpp_rbf_C1.0_seed42.pkl` — fitted scaler + SVM
- `hdc/checkpoints/esm2_hdc_amp_vs_cpp_d5000_lr0.01_ep400_seed42*.pt` — 3 OnlineHD
  checkpoints (default `dim=5000,lr=0.01,epochs=400`, plus two `sigma` variants from
  hyperparameter sweeps)
- `embeddings/esm2_t12_35M_UR50D_poreforming_amp_cpp.npz` — cached ESM2 embeddings for
  the 634-sequence training set, so retraining (`train_esm2_*_amp_cpp.py`) doesn't need
  the original raw sequence files, which are **not** part of this repo. If you delete
  this cache, you'll need to re-source that training set yourself.

## Usage

### Inference

**Input file**: a CSV with at minimum a `sequence` column and a `pred_label` column
(the upstream Stage-1 AMP-plausible-vs-decoy screen's output; rows where
`pred_label != 1` are skipped — i.e. only Stage-1 survivors get scored). Any other
columns (e.g. `DRAMP_ID`, `Name`) are optional, passed through unchanged, and used
only to make the printed top-5 report more readable.

```csv
DRAMP_ID,Name,sequence,pred_label
PEP001,Example AMP,GIGKFLHSAKKFGKAFVGEIMNS,1
PEP002,Filtered decoy,ACDEFGHIKLMNPQRSTVWY,0
```

**Non-standard residues** (e.g. hydrocarbon-staple anchor codes like `X`/`Z`/`S5`/`R8`)
aren't in ESM2's vocabulary — substitute them with a natural stand-in (commonly
Leucine, to preserve bulky hydrophobic character) in the `sequence` column before
scoring.

```bash
python svm/infer_stapled_esm2svm.py --input your_stage1_output.csv --output predictions_svm.csv
python hdc/infer_stapled_esm2hdc.py --input your_stage1_output.csv --output predictions_hdc.csv
```

Both default to `--input symbolic_onlinehd_best_bundle/predictions/stapled_predictions.csv`
if `--input` is omitted (a path from the original monorepo this was extracted from —
you'll almost always want to pass `--input` explicitly).

### Artifacts produced

| Script | File | Contents |
|---|---|---|
| `svm/infer_stapled_esm2svm.py` | `--output` CSV (default `svm/predictions/stapled_predictions_esm2svm.csv`) | input CSV's rows (Stage-1 survivors only) + `esm2svm_p_amp_pore`, `esm2svm_p_cpp_pore`, `esm2svm_pred_label`, `esm2svm_pred_class`, `esm2svm_svm_distance` (signed distance from the SVM decision boundary; negative = AMP_pore side, positive = CPP_pore side) |
| | `embeddings/stapled_esm2svm_<input basename>_embeddings.npz` (or `--emb-output`) | raw ESM2 embeddings (`X`), `sequences`, `pred_class`, `p_amp_pore`, `p_cpp_pore`, `dramp_id` — for downstream use (e.g. t-SNE plots) |
| `hdc/infer_stapled_esm2hdc.py` | `--output` CSV (default `hdc/predictions/stapled_predictions_esm2hdc.csv`) | input CSV's rows (Stage-1 survivors only) + `esm2hdc_p_amp_pore`, `esm2hdc_p_cpp_pore`, `esm2hdc_pred_label`, `esm2hdc_pred_class`, and (via `hdc_scoring.py`, called automatically) `esm2hdc_cos_amp`, `esm2hdc_cos_cpp`, `esm2hdc_axis_score` (combined AMP&harr;CPP axis score in the OnlineHD hypervector space; same sign convention as the SVM distance) |
| | `embeddings/stapled_esm2_<input basename>_embeddings.npz` (or `--emb-output`) | raw ESM2 embeddings (`X`), `sequences`, `pred_class`, `dramp_id` |
| `hdc/hdc_scoring.py` (standalone) | overwrites its `--input` CSV (or writes `--output` if given) | appends just `esm2hdc_cos_amp`, `esm2hdc_cos_cpp`, `esm2hdc_axis_score` to a CSV that already has `esm2hdc_*` predictions — used if you ran `infer_stapled_esm2hdc.py` from an older version or want to recompute the axis score alone |

### (Re)training

Only needed if you want to reproduce the checkpoints from scratch or retrain on new
data — inference above does **not** require this:

```bash
python svm/train_esm2_svm_amp_cpp.py
python hdc/train_esm2_hdc_amp_cpp.py
```

Both re-run 5-fold CV at the pinned config first (expect mean accuracy ~0.95, MCC
~0.90) before the final full-data fit that overwrites the checkpoint. Neither takes an
input file — training data comes from `embeddings/esm2_t12_35M_UR50D_poreforming_amp_cpp.npz`
(see "What's included" above).

| Script | File | Contents |
|---|---|---|
| `svm/train_esm2_svm_amp_cpp.py` | `embeddings/esm2_t12_35M_UR50D_poreforming_amp_cpp.npz` | cached ESM2 embeddings (`X`) + labels (`y`) for the 634-sequence training set (written once; reused by every script above) |
| | `svm/checkpoints/esm2_svm_amp_cpp_rbf_C1.0_seed42.pkl` | fitted `{'svm': ..., 'scaler': ...}` — overwrites the shipped checkpoint |
| | `svm/reports/esm2_svm_amp_cpp_<timestamp>.txt` | 5-fold CV classification report, MCC, confusion matrix |
| `hdc/train_esm2_hdc_amp_cpp.py` | same `embeddings/*.npz` as above | shared with the SVM trainer |
| | `hdc/checkpoints/esm2_hdc_amp_vs_cpp_d{dim}_lr{lr}_ep{epochs}_seed{seed}.pt` | fitted OnlineHD model — overwrites the shipped checkpoint if hyperparameters match, otherwise adds a new file |
| | `hdc/reports/esm2_hdc_amp_vs_cpp_<timestamp>.txt` | 5-fold CV classification report, MCC, confusion matrix |

### Other scripts in `hdc/`

- `hdc_tuning.py` — grid search over `(dim, lr, epochs)` via `helpers.hdc_grid_search`
- `check_training_scores.py` — sanity-check a checkpoint's scores against the training set
- `plot_axis_score_and_plane.py` — visualize the OnlineHD axis score / decision plane
- `build_stapled_fastas.py` — CSV -> FASTA helper for downstream tools
- `pepbert_hdc.py` — alternate embedding backbone (PepBERT via `tokenizers` +
  `huggingface_hub`) experiment, independent of the ESM2 pipeline above

## Repo layout

```text
helpers.py                    # shared ESM2 loading/embedding + CV/report utilities
svm/
  train_esm2_svm_amp_cpp.py
  infer_stapled_esm2svm.py
  checkpoints/
hdc/
  esm2_hdc.py                 # ESM2 loading/embedding (imported by svm/ scripts too)
  train_esm2_hdc_amp_cpp.py
  infer_stapled_esm2hdc.py
  hdc_scoring.py
  hdc_tuning.py
  check_training_scores.py
  plot_axis_score_and_plane.py
  build_stapled_fastas.py
  pepbert_hdc.py
  checkpoints/
embeddings/                   # cached ESM2 training embeddings
```

Note: `svm/` scripts add `hdc/` to `sys.path` at runtime to reuse `esm2_hdc.py` — keep
both folders alongside each other.
