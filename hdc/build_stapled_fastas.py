"""
Build the AMP_pore / CPP_pore FASTA files the VAE notebook projects, from the
current (stage-1-filtered) ESM2+HDC output.

Input:
    hdc/predictions/stapled_predictions_esm2hdc.csv   (written by infer_stapled_esm2hdc.py)

Output:
    VAE_trained_windows_original/stapled_candidates/stapled_AMP_pore_fasta.txt
    VAE_trained_windows_original/stapled_candidates/stapled_CPP_pore_fasta.txt

Run:
    python hdc/build_stapled_fastas.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INPUT_CSV = os.path.join(HERE, 'predictions', 'stapled_predictions_esm2hdc.csv')
OUT_DIR = os.path.join(ROOT, 'VAE_trained_windows_original', 'stapled_candidates')

OUT_FILES = {
    'AMP_pore': os.path.join(OUT_DIR, 'stapled_AMP_pore_fasta.txt07222026'),
    'CPP_pore': os.path.join(OUT_DIR, 'stapled_CPP_pore_fasta.txt07222026'),
}


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run `python hdc/infer_stapled_esm2hdc.py` first.")

    with open(INPUT_CSV, newline='') as f:
        rows = list(csv.DictReader(f))

    os.makedirs(OUT_DIR, exist_ok=True)
    counts = {'AMP_pore': 0, 'CPP_pore': 0}
    handles = {cls: open(path, 'w') for cls, path in OUT_FILES.items()}
    try:
        for r in rows:
            cls = r['esm2hdc_pred_class']
            handles[cls].write(f">{r['DRAMP_ID']}\n{r['sequence']}\n")
            counts[cls] += 1
    finally:
        for h in handles.values():
            h.close()

    for cls, path in OUT_FILES.items():
        print(f"{cls}: {counts[cls]} sequences -> {path}")


if __name__ == '__main__':
    main()
