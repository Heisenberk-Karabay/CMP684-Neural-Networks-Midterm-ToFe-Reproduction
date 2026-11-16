"""Figures for the IEEE two-column paper (report/ieee/figs): one column = 3.4 in, 8 pt text.
Charts are SVG (vector), image grids are 300-dpi PNG. Built from the saved results; only the
example images need a few forward passes."""
import json
import os

import timm
import torch

import data
import plotstyle as ps
from analyze_tofe import load_run
from common import DEV, ROOT, RUNS, load_backbone
from make_figures import load_runs
from motivation import denorm, first_of, topk_mask, trace
from plotstyle import plt

OUT = os.path.join(ROOT, 'report', 'ieee', 'figs')
COL = 3.4  # IEEE column width (in)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'lines.linewidth': 1.5, 'lines.markersize': 4, 'svg.fonttype': 'none'})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def motivation(res):
    L = len(res['top50_attention_mass'])
    blocks = list(range(1, L + 1))
    fig, ax = plt.subplots(2, 1, figsize=(COL, 3.3))
    ax[0].plot(blocks, [100 * v for v in res['top50_attention_mass']], marker='o', color=ps.SERIES[0])
    ax[0].axhline(50, color=ps.MUTED, lw=0.8, ls='--')
    ax[0].text(12, 51.5, 'uniform', ha='right', color=ps.MUTED, fontsize=7)
    ax[0].set(ylabel='mass on top-50% (%)', xticks=blocks)
    ax[0].set_title('(a) [CLS] attention on the top half of the tokens', loc='left')
    for key, c, half in (('attentive', ps.SERIES[0], 'top 50%'), ('inattentive', ps.SERIES[1], 'bottom 50%')):
        ax[1].plot(blocks, res['eps_per_block'][key], marker='o', color=c, label=f'{key} ({half})')
    ax[1].set(xlabel='block', ylabel=r'$\|\Delta X\| / \|X\|$', xticks=blocks)
    ax[1].set_title('(b) relative change of each token in one block', loc='left')
    ax[1].legend(loc='upper right')
    fig.tight_layout(h_pad=0.8)
    save(fig, 'motivation.svg')


def cosine(res):
    cos = torch.tensor(res['cosine_similarity'])
    L = cos.size(0)
    fig, ax = plt.subplots(figsize=(COL * 0.78, COL * 0.66))
    im = ax.imshow(cos, cmap=ps.SEQ, vmin=0, vmax=1)
    ax.grid(False)
    ax.set(xticks=range(L), yticks=range(L), xticklabels=range(1, L + 1), yticklabels=range(1, L + 1),
           xlabel='block', ylabel='block')
    for a in range(L - 1):
        ax.text(a + 1, a, f'{cos[a, a + 1]:.2f}'[1:], ha='center', va='center', fontsize=5, color='white')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='cosine similarity')
    save(fig, 'cosine.svg')


def image_grids(vit, val, cfg):
    examples = {'stopwatch': 826, 'husky': 250, 'tench': 0}
    fig, axs = plt.subplots(3, 4, figsize=(COL, COL * 0.78))
    for r, (name, lab) in enumerate(examples.items()):
        img = val[first_of(val, lab)][0]
        _, cls = trace(vit, img.unsqueeze(0).to(DEV))
        pic = denorm(img, cfg)
        axs[r, 0].imshow(pic)
        axs[r, 0].set_ylabel(name, fontsize=7)
        for c, l in enumerate([3, 6, 9]):
            keep = topk_mask(cls[l], 0.5)[0].view(14, 14).float()
            m = keep.repeat_interleave(16, 0).repeat_interleave(16, 1).cpu().numpy()[..., None]
            axs[r, c + 1].imshow(pic * m)
            if r == 0:
                axs[r, c + 1].set_title(f'block {l + 1}')
        if r == 0:
            axs[r, 0].set_title('image')
        for a in axs[r]:
            ps.image_axes(a)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, 'masks.png')

    img = val[first_of(val, 330)][0]
    _, cls = trace(vit, img.unsqueeze(0).to(DEV))
    fig, axs = plt.subplots(2, 3, figsize=(COL, COL * 0.8))
    axs = axs.flatten()
    axs[0].imshow(denorm(img, cfg))
    axs[0].set_title('image')
    for a, l in zip(axs[1:], [1, 4, 6, 8, 9]):
        m = cls[l][0].float().cpu()
        a.imshow(m.view(14, 14), cmap=ps.SEQ, vmin=0, vmax=m.quantile(0.975).item())
        a.set_title(f'block {l + 1}')
        a.set_xlabel(f'top-2 tokens: {100 * (m.topk(2).values.sum() / m.sum()):.0f}%', fontsize=6.5, color=ps.INK2)
    for a in axs:
        ps.image_axes(a)
    fig.subplots_adjust(wspace=0.08, hspace=0.55)
    save(fig, 'attention.png')


def tokens(val, cfg):
    model, _ = load_run('tofe_b50')
    picks = {'husky': first_of(val, 250), 'stopwatch': first_of(val, 826), 'bus': first_of(val, 779)}
    g = model.vit.patch_embed.grid_size[0]
    S = len(model.locs)
    fig, axs = plt.subplots(len(picks), S + 2, figsize=(COL, COL * 0.62))
    for r, (label, i) in enumerate(picks.items()):
        img = val[i][0]
        with torch.no_grad():
            _, masks, _ = model.forward_train(img[None].to(DEV), hard_eval=True)
        pic = denorm(img, cfg)
        up = lambda t: t.view(g, g).repeat_interleave(16, 0).repeat_interleave(16, 1).cpu().numpy()[..., None]
        axs[r, 0].imshow(pic)
        usage = torch.ones(model.num_patches, device=DEV)
        for s, m in enumerate(masks):
            m = m[0]
            usage += m
            keep = up(m)
            axs[r, s + 1].imshow(pic * keep + (1 - keep) * (0.25 * pic + 0.75))
            if s > 0:
                for t in (m * (1 - masks[s - 1][0])).nonzero().flatten().tolist():
                    axs[r, s + 1].add_patch(plt.Rectangle(((t % g) * 16 - 0.5, (t // g) * 16 - 0.5), 16, 16,
                                                          fill=False, edgecolor=ps.SERIES[3], linewidth=0.5))
            axs[r, s + 1].set_xlabel(f'{int(m.sum())} kept', fontsize=6, color=ps.INK2, labelpad=1)
            if r == 0:
                axs[r, s + 1].set_title(f'stage {s + 1}')
        im = axs[r, -1].imshow(usage.view(g, g).cpu(), cmap=ps.SEQ, vmin=1, vmax=S + 1)
        if r == 0:
            axs[r, 0].set_title('image')
            axs[r, -1].set_title('usage')
        for a in axs[r]:
            ps.image_axes(a)
    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    save(fig, 'tokens.png')


def tradeoff(runs, base):
    dense = next(b for b in base if b['method'] == 'dense')
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    ax.axhline(dense['top1'], color=ps.MUTED, lw=0.8, ls='--')
    ax.text(4.0, dense['top1'] + 0.3, f"dense {dense['top1']:.1f}", ha='right', color=ps.MUTED, fontsize=7)
    curves = [('ToMe (no training)', ps.SERIES[2], [(b['gmacs'], b['top1']) for b in base if b['method'] == 'ToMe']),
              ('EViT (no training)', ps.SERIES[1], [(b['gmacs'], b['top1']) for b in base
                                                     if b['method'] == 'EViT' and b['setting'] != [96, 49, 27]]),
              ('ToFe (ours)', ps.SERIES[0], [(runs[n]['gmacs'], runs[n]['top1'])
                                             for n in ('tofe_b40', 'tofe_b50', 'tofe_b60', 'tofe_b70')])]
    for label, color, pts in curves:
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, marker='o', color=color, label=label)
    r = runs['fixed_33_33_33']
    ax.plot(r['gmacs'], r['top1'], 'o', color=ps.SERIES[0], markerfacecolor='white', markeredgewidth=1.2)
    ax.annotate('ToFe, fixed ratios', (r['gmacs'], r['top1']), xytext=(-5, 0), textcoords='offset points',
                ha='right', va='center', fontsize=6.5, color=ps.INK2)
    ax.plot([2.0, 2.3], [77.7, 78.48], '*', color=ps.SERIES[6], markersize=8, label='ToFe, reported in [1]')
    ax.set(xlabel='GMACs', ylabel='top-1 accuracy (%)', xlim=(1.8, 4.1))
    ax.legend(loc='lower right')
    save(fig, 'tradeoff.svg')


VARIANTS = [('ToFe (default)', ['tofe_b50', 'tofe_b50_s1']), ('15 epochs', ['tofe_b50_e15']),
            ('+ backbone fine-tuning', ['tofe_b50_ft3']), ('no reuse', ['noreuse_b50', 'noreuse_b50_s1']),
            ('identity approximator', ['apx_identity']), ('DW-conv approximator', ['apx_dwconv']),
            ('5-layer selector', ['sel_depth5']), ('7-layer selector', ['sel_depth7']),
            ('DynamicViT selector', ['sel_dynamicvit']), ('fixed (.5, .25, .125)', ['fixed_50_25_12']),
            ('fixed (.4, .4, .1)', ['fixed_40_40_10']), ('fixed (1/3, 1/3, 1/3)', ['fixed_33_33_33']),
            ('fixed (.4, .3, .3)', ['fixed_40_30_30'])]


def variants(runs, base):
    rows = [(label, [runs[n]['top1'] for n in names]) for label, names in VARIANTS]
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    ys = list(range(len(rows)))[::-1]
    for label, x, color in (('EViT', next(b['top1'] for b in base if b['method'] == 'EViT' and b['setting'] == [96, 49, 27]), ps.SERIES[1]),
                            ('ToMe', next(b['top1'] for b in base if b['method'] == 'ToMe' and b['setting'] == 16), ps.SERIES[2])):
        ax.axvline(x, color=color, lw=1.2, ls='--')
        ax.text(x, len(rows) - 0.3, label, color=ps.INK2, fontsize=7, ha='center', va='bottom')
    for y, (label, vals) in zip(ys, rows):
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y, y], color=ps.SERIES[0], lw=1.5)
        ax.plot(mean, y, 'o', color=ps.SERIES[0])
        ax.text(mean + 0.3, y, f'{mean:.2f}', va='center', fontsize=6.5, color=ps.INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) + 0.4)
    ax.set_xlim(68.5, 79)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('top-1 accuracy (%) at about 2.3 GMACs')
    save(fig, 'variants.svg')


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(os.path.join(RUNS, 'motivation.json')))
    runs, base = load_runs(), json.load(open(os.path.join(RUNS, 'baselines.json')))
    motivation(res)
    cosine(res)
    tradeoff(runs, base)
    variants(runs, base)
    vit = load_backbone()
    val = data.ImageNetFiles('val', data.eval_transform(vit))
    cfg = timm.data.resolve_data_config({}, model=vit)
    image_grids(vit, val, cfg)
    tokens(val, cfg)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
