"""Figures built from finished runs: FLOPs-accuracy trade-off (paper Fig. 6 / Table I)."""
import glob
import json
import os

import plotstyle as ps
from common import RUNS
from motivation import FIGS
from plotstyle import plt


def load_runs():
    out = {}
    for f in glob.glob(os.path.join(RUNS, '*', 'results.json')):
        out[os.path.basename(os.path.dirname(f))] = json.load(open(f))  # key by run directory
    return out


def tradeoff(runs, base):
    dense = next(b for b in base if b['method'] == 'dense')
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(dense['top1'], color=ps.MUTED, lw=1, ls='--')
    ax.text(4.55, dense['top1'] + 0.25, f"dense DeiT-S {dense['top1']:.2f}", ha='right', color=ps.MUTED, fontsize=8)
    series = [
        ('ToMe (training-free)', ps.SERIES[2], sorted((b['gmacs'], b['top1']) for b in base if b['method'] == 'ToMe')),
        ('EViT (training-free)', ps.SERIES[1], sorted((b['gmacs'], b['top1']) for b in base
                                                      if b['method'] == 'EViT' and b['setting'] != [96, 49, 27])),
        ('ToFe, ours (5 ep, frozen backbone)', ps.SERIES[0],
         sorted((r['gmacs'], r['top1']) for n, r in runs.items() if n in ('tofe_b40', 'tofe_b50', 'tofe_b60', 'tofe_b70'))),
    ]
    for label, color, pts in series:
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker='o', color=color, label=label)
    extra = [('fixed_33_33_33', 'ToFe, fixed ratios (1/3,1/3,1/3)', 'o', (-9, 0)),
             ('noreuse_b50', 'ToFe w/o reuse', 's', (-9, 0))]
    for name, label, marker, off in extra:
        if name in runs:
            r = runs[name]
            ax.plot(r['gmacs'], r['top1'], marker=marker, ls='none', color=ps.SERIES[0], markerfacecolor='white',
                    markeredgewidth=1.6, markersize=7)
            ax.annotate(label, (r['gmacs'], r['top1']), xytext=off, textcoords='offset points', fontsize=7.5,
                        color=ps.INK2, ha='left' if off[0] > 0 else 'right', va='center')
    paper = [(2.0, 77.7, 'paper: ToFe (Table I)'), (2.3, 78.48, 'paper: ToFe (Table VII)')]
    for x, y, label in paper:
        ax.plot(x, y, marker='*', ls='none', color=ps.SERIES[6], markersize=11)
        ax.annotate(label, (x, y), xytext=(-2, 9), textcoords='offset points', fontsize=7.5, color=ps.INK2,
                    ha='center')
    ax.set(xlabel='GMACs (Eq. 4)', ylabel='ImageNet top-1 (%)')
    ax.legend(loc='lower right', fontsize=8)
    fig.savefig(os.path.join(FIGS, 'tradeoff.png'))
    plt.close(fig)


VARIANTS = [  # (group, label, runs averaged over seeds)
    ('main', 'ToFe (learned budget, reuse, MLP approx.)', ['tofe_b50', 'tofe_b50_s1']),
    ('training', '15 epochs instead of 5', ['tofe_b50_e15']),
    ('training', '+ 3 epochs backbone fine-tuning', ['tofe_b50_ft3']),
    ('Table V', 'w/o token reusing', ['noreuse_b50', 'noreuse_b50_s1']),
    ('Table VI', 'identity approximator', ['apx_identity']),
    ('Table VI', 'depth-wise conv approximator', ['apx_dwconv']),
    ('Table IV', '5-layer selector', ['sel_depth5']),
    ('Table IV', '7-layer selector', ['sel_depth7']),
    ('Table IV', 'DynamicViT selector', ['sel_dynamicvit']),
    ('Table VII', 'fixed ratios (0.5, 0.25, 0.125)', ['fixed_50_25_12']),
    ('Table VII', 'fixed ratios (0.4, 0.4, 0.1)', ['fixed_40_40_10']),
    ('Table VII', 'fixed ratios (1/3, 1/3, 1/3)', ['fixed_33_33_33']),
    ('Table VII', 'fixed ratios (0.4, 0.3, 0.3)', ['fixed_40_30_30']),
]


def variants(runs, base):
    """Dot plot of all ToFe variants at ~50% FLOPs against the training-free baselines."""
    rows = []
    for group, label, names in VARIANTS:
        vals = [runs[n]['top1'] for n in names if n in runs]
        if vals:
            rows.append((group, label, sum(vals) / len(vals), min(vals), max(vals),
                         sum(runs[n]['gmacs'] for n in names if n in runs) / len(vals)))
    fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(rows) + 1.0))
    ys = list(range(len(rows)))[::-1]
    refs = [('ToMe r=16 (2.29 G)', next(b['top1'] for b in base if b['method'] == 'ToMe' and b['setting'] == 16),
             ps.SERIES[2]),
            ('EViT (2.30 G)', next(b['top1'] for b in base if b['method'] == 'EViT' and b['setting'] == [96, 49, 27]),
             ps.SERIES[1])]
    for label, x, color in refs:
        ax.axvline(x, color=color, lw=1.5, ls='--')
        ax.text(x, len(rows) - 0.4, label, color=ps.INK2, fontsize=7.5, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    for y, (group, label, mean, lo, hi, g) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=ps.SERIES[0], lw=2)
        ax.plot(mean, y, 'o', color=ps.SERIES[0], markersize=6)
        ax.text(mean + 0.25, y, f'{mean:.2f}  ({g:.2f} G)', va='center', fontsize=7.5, color=ps.INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{label}  [{group}]' for group, label, *_ in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xlim(68, 80)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('ImageNet top-1 (%), batch-adaptive inference (bars: range over seeds)')
    fig.savefig(os.path.join(FIGS, 'variants.png'))
    plt.close(fig)


if __name__ == '__main__':
    base = json.load(open(os.path.join(RUNS, 'baselines.json')))
    runs = load_runs()
    tradeoff(runs, base)
    variants(runs, base)
