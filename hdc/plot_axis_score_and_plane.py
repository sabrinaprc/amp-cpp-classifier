"""
Plots the raw signed HDC AMP<->CPP axis score, and the (cos_amp, cos_cpp)
plane it's built from, against MIC and hemolysis.

Background: esm2hdc_axis_score = cos_cpp - cos_amp (see hdc_scoring.py) was
already the raw signed value -- negative=AMP side, positive=CPP side, no
normalization/shift was ever applied to it. What the existing mic_analysis /
hemolysis plots do instead is split AMP_pore-predicted and CPP_pore-predicted
peptides into two separate panels, each plotted on its own sign-flipped
column (cpp_distance for the CPP panel, amp_distance = -cpp_distance for the
AMP panel) so "higher on this panel" always reads as "more confidently this
panel's own class". That per-panel mirroring is deliberate for those plots,
but it means neither panel alone shows the single shared signed score.

This script instead plots the one shared signed axis score directly, with
every peptide (AMP_pore and CPP_pore predicted) on the same axes:
    Plot 1: esm2hdc_axis_score vs MIC
    Plot 2: esm2hdc_axis_score vs hemolysis
    Plot 5: esm2hdc_axis_score vs lyticity index (linear x-axis -- lyticity
            index is a sequence-derived score, not a concentration, so unlike
            MIC/hemolysis it isn't log-distributed)
Class is shown via color/marker instead of via separate panels or sign flips.

It also breaks the axis score into its two raw components -- cos_amp
(similarity to the AMP class hypervector) and cos_cpp (similarity to the
CPP class hypervector) -- and plots them directly against each other:
    Plot 3: (cos_amp, cos_cpp) plane, colored by MIC
    Plot 4: (cos_amp, cos_cpp) plane, colored by hemolysis
Points below the y=x line are more AMP-like (cos_amp > cos_cpp); above it,
more CPP-like -- the same distinction axis_score makes, but visible as two
raw similarities instead of one difference.

Uses the esm2hdc_cos_amp / esm2hdc_cos_cpp / esm2hdc_axis_score columns
already computed by hdc_scoring.py -- no HDC scoring is redone here.

Run:
    conda run -n stap python hdc/plot_axis_score_and_plane.py
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_OUT_DIR = os.path.join(HERE, 'output')

DEFAULT_MIC_CSV = os.path.join(ROOT, 'mic_analysis', 'output', 'stapled_final_hdc.csv')
HDC_PRED_CSV  = os.path.join(HERE, 'predictions', 'stapled_predictions_esm2hdc.csv')
HEMOLYSIS_CSV = os.path.join(ROOT, 'hemolysis', 'output', 'stapled_amps_hemolysis_25uM_linear.csv')

CLASS_STYLE = {
    'AMP_pore': dict(color='#DD8452', marker='o', label='AMP_pore'),
    'CPP_pore': dict(color='#4C72B0', marker='^', label='CPP_pore'),
}


def load_mic_df(mic_csv):
    """Uncensored, MIC-known rows from the HDC-scored stapled candidates."""
    df = pd.read_csv(mic_csv)
    return df[~df['censored'] & df['mic_ug_ml'].notna()].copy()


def load_lyticity_df(mic_csv):
    """All HDC-scored stapled candidates with a computed lyticity index
    (independent of MIC/censoring -- lyticity is sequence-derived only)."""
    df = pd.read_csv(mic_csv)
    return df[df['lyticity_index'].notna()].copy()


def load_hemolysis_df(hdc_pred_csv):
    """HDC scores joined to %hemolysis on DRAMP_ID (mirrors plot_distance_vs_hemolysis.py)."""
    pred = pd.read_csv(hdc_pred_csv)
    hem = pd.read_csv(HEMOLYSIS_CSV)
    df = pred.merge(hem[['DRAMP_ID', 'hemolysis_pct_25uM']], on='DRAMP_ID', how='inner')
    return df[df['hemolysis_pct_25uM'].notna()].copy()


def plot_signed_score(df, x_col, x_label, title, outfile, x_log=True,
                       score_col='esm2hdc_axis_score', pred_class_col='esm2hdc_pred_class',
                       score_label='HDC axis score  (cos_cpp - cos_amp)'):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cls, sub in df.groupby(pred_class_col):
        style = CLASS_STYLE.get(cls, dict(color='gray', marker='o', label=cls))
        ax.scatter(sub[x_col], sub[score_col],
                   color=style['color'], marker=style['marker'],
                   edgecolor='k', linewidth=0.4, alpha=0.75, s=45,
                   label=f"{style['label']} (n={len(sub)})")

    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, label='boundary (score=0)')
    if x_log:
        ax.set_xscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(score_label)
    ax.set_title(f'{title}  (n={len(df)})')

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.text(0.01, 0.98, 'positive = CPP-like', transform=ax.transAxes,
            fontsize=8, va='top', ha='left', style='italic')
    ax.text(0.01, 0.02, 'negative = AMP-like', transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left', style='italic')

    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfile}")


def plot_plane(df, color_col, color_label, color_log, title, outfile):
    fig, ax = plt.subplots(figsize=(6.5, 6))

    vals = df[color_col].to_numpy()
    if color_log:
        norm = mcolors.LogNorm(vmin=vals.min(), vmax=vals.max())
    else:
        norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())

    sc = None
    for cls, sub in df.groupby('esm2hdc_pred_class'):
        style = CLASS_STYLE.get(cls, dict(color='gray', marker='o', label=cls))
        sc = ax.scatter(sub['esm2hdc_cos_amp'], sub['esm2hdc_cos_cpp'],
                        c=sub[color_col], cmap='viridis', norm=norm,
                        marker=style['marker'], edgecolor='k', linewidth=0.4,
                        s=55, label=f"{style['label']} (n={len(sub)})")

    lo = min(df['esm2hdc_cos_amp'].min(), df['esm2hdc_cos_cpp'].min())
    hi = max(df['esm2hdc_cos_amp'].max(), df['esm2hdc_cos_cpp'].max())
    pad = (hi - lo) * 0.08 or 0.01
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color='gray', linestyle='--', linewidth=1, zorder=1,
            label='cos_amp = cos_cpp (boundary)')
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('cos_amp  (cosine similarity to AMP hypervector)')
    ax.set_ylabel('cos_cpp  (cosine similarity to CPP hypervector)')
    ax.set_title(f'{title}  (n={len(df)})')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(color_label)

    ax.legend(fontsize=7.5, loc='upper left')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfile}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=DEFAULT_MIC_CSV,
                    help="HDC-scored MIC csv (needs esm2hdc_axis_score/cos_amp/cos_cpp, "
                         "mic_ug_ml, censored, esm2hdc_pred_class). Default: the E. coli "
                         "sigma=1.0 baseline.")
    p.add_argument('--hdc-pred-csv', default=HDC_PRED_CSV,
                    help="HDC-scored predictions csv used for the hemolysis plots (needs "
                         "DRAMP_ID, esm2hdc_cos_amp/cos_cpp/pred_class). Default: the "
                         "sigma=1.0 baseline predictions.")
    p.add_argument('--tag', default=None,
                    help="Suffix for output filenames, e.g. 'sigma0.5' -> "
                         "axis_score_vs_mic_sigma0.5.png (default: none, uses the plain names)")
    p.add_argument('--mic-only', action='store_true',
                    help="Only produce the axis-score-vs-MIC plot")
    p.add_argument('--linear-x', action='store_true',
                    help="Plot MIC on a linear x-axis instead of the default log scale")
    p.add_argument('--hemolysis-only', action='store_true',
                    help="Only produce the (cos_amp, cos_cpp) plane colored by hemolysis")
    p.add_argument('--outdir', default=DEFAULT_OUT_DIR,
                    help=f"Directory to write plots into (default: {DEFAULT_OUT_DIR})")
    p.add_argument('--classifier', choices=['hdc', 'svm'], default='hdc',
                    help="Which classifier's score column to plot against MIC. "
                         "hdc: esm2hdc_axis_score/esm2hdc_pred_class (default). "
                         "svm: esm2svm_svm_distance/esm2svm_pred_class -- same signed "
                         "convention (negative=AMP side, positive=CPP side), just a "
                         "hyperplane distance instead of a cosine-similarity margin. "
                         "Only affects the MIC/hemolysis signed-score plots -- the "
                         "(cos_amp, cos_cpp) plane is HDC-specific and always uses HDC columns.")
    return p.parse_args()


SCORE_CONFIG = {
    'hdc': dict(score_col='esm2hdc_axis_score', pred_class_col='esm2hdc_pred_class',
                score_label='HDC axis score  (cos_cpp - cos_amp)', title_prefix='HDC axis score'),
    'svm': dict(score_col='esm2svm_svm_distance', pred_class_col='esm2svm_pred_class',
                score_label='SVM distance to hyperplane  (cpp side > 0)', title_prefix='SVM distance'),
}


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f'_{args.tag}' if args.tag else ''
    score_cfg = SCORE_CONFIG[args.classifier]

    if args.hemolysis_only:
        hem_df = load_hemolysis_df(args.hdc_pred_csv)
        plot_plane(
            hem_df, 'hemolysis_pct_25uM', '%hemolysis at 25 uM', False,
            'cos_amp vs cos_cpp, colored by hemolysis', os.path.join(OUT_DIR, f'cos_plane_vs_hemolysis{suffix}.png'))
        return

    mic_df = load_mic_df(args.input)
    organism = mic_df['mic_organism'].dropna().iloc[0] if mic_df['mic_organism'].notna().any() else 'unknown organism'
    x_log = not args.linear_x
    scale_label = 'linear scale' if args.linear_x else 'log scale'
    plot_signed_score(
        mic_df, 'mic_ug_ml', f'MIC (µg/mL, {organism}, {scale_label})',
        f'{score_cfg["title_prefix"]} vs MIC ({organism})', os.path.join(OUT_DIR, f'axis_score_vs_mic{suffix}.png'),
        x_log=x_log, score_col=score_cfg['score_col'], pred_class_col=score_cfg['pred_class_col'],
        score_label=score_cfg['score_label'])

    if args.mic_only:
        return

    hem_df = load_hemolysis_df(args.hdc_pred_csv)
    lyt_df = load_lyticity_df(args.input)

    plot_signed_score(
        hem_df, 'hemolysis_pct_25uM', '%hemolysis at 25 uM (log scale)',
        'HDC axis score vs hemolysis', os.path.join(OUT_DIR, f'axis_score_vs_hemolysis{suffix}.png'))

    plot_signed_score(
        lyt_df, 'lyticity_index', 'Lyticity index',
        'HDC axis score vs lyticity index', os.path.join(OUT_DIR, f'axis_score_vs_lyticity{suffix}.png'),
        x_log=False)

    plot_plane(
        mic_df, 'mic_ug_ml', 'MIC (µg/mL)', True,
        'cos_amp vs cos_cpp, colored by MIC', os.path.join(OUT_DIR, f'cos_plane_vs_mic{suffix}.png'))

    plot_plane(
        hem_df, 'hemolysis_pct_25uM', '%hemolysis at 25 uM', False,
        'cos_amp vs cos_cpp, colored by hemolysis', os.path.join(OUT_DIR, f'cos_plane_vs_hemolysis{suffix}.png'))


if __name__ == '__main__':
    main()
