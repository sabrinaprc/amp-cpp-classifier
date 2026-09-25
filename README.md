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

Both inference scripts expect an input CSV with a `sequence` column and a `pred_label`
column (Stage-1 output; rows with `pred_label != 1` are skipped):

```bash
python svm/infer_stapled_esm2svm.py --input your_stage1_output.csv --output predictions_svm.csv
python hdc/infer_stapled_esm2hdc.py --input your_stage1_output.csv --output predictions_hdc.csv
```

`svm/infer_stapled_esm2svm.py` appends: `esm2svm_p_amp_pore`, `esm2svm_p_cpp_pore`,
`esm2svm_pred_label`, `esm2svm_pred_class`, `esm2svm_svm_distance` (signed distance from
the SVM decision boundary; negative = AMP_pore side, positive = CPP_pore side).

`hdc/infer_stapled_esm2hdc.py` appends: `esm2hdc_p_amp_pore`, `esm2hdc_p_cpp_pore`,
`esm2hdc_pred_label`, `esm2hdc_pred_class`, then calls `hdc/hdc_scoring.py` internally to
also append `esm2hdc_cos_amp`, `esm2hdc_cos_cpp`, `esm2hdc_axis_score` (combined
AMP&harr;CPP axis score in the OnlineHD hypervector space; same sign convention as the
SVM distance).

**Non-standard residues** (e.g. hydrocarbon-staple anchor codes like `X`/`Z`/`S5`/`R8`)
aren't in ESM2's vocabulary — substitute them with a natural stand-in (commonly
Leucine, to preserve bulky hydrophobic character) in the `sequence` column before
scoring.

### (Re)training

Only needed if you want to reproduce the checkpoints from scratch or retrain on new
data — inference above does **not** require this:

```bash
python svm/train_esm2_svm_amp_cpp.py
python hdc/train_esm2_hdc_amp_cpp.py
```

Both re-run 5-fold CV at the pinned config first (expect mean accuracy ~0.95, MCC
~0.90) before the final full-data fit that overwrites the checkpoint.

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
