"""Phase 1: check the paper's motivation (Sec. I, III-B, Figs. 3-4) on the dense DeiT-S.

M1  [CLS] attention is scattered in shallow blocks and concentrated in deep ones (Fig. 3)
M2  tokens that matter in deep blocks are dropped by top-50% pruning in shallow blocks
M3  token features are similar across consecutive blocks (Fig. 4b)
M4  inattentive tokens change little across blocks (Sec. III-B), so they can be approximated
"""
import json
import os

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import data
import plotstyle as ps
from common import DEV, ROOT, load_backbone
from plotstyle import plt

FIGS = os.path.join(ROOT, 'report', 'figs')
PROBE = [3, 6, 9]  # 4th, 7th, 10th block (Fig. 3)


@torch.no_grad()
def trace(vit, img):
    """Dense forward returning all block inputs/outputs and head-averaged [CLS]->patch attention."""
    x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(img))))
    xs, cls = [x], []
    for blk in vit.blocks:
        a = blk.attn
        B, T, C = x.shape
        qkv = a.qkv(blk.norm1(x)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q * a.scale) @ k.transpose(-2, -1)).softmax(-1)
        cls.append(attn[:, :, 0, 1:].mean(1))
        x = x + blk.ls1(a.proj(a.norm((attn @ v).transpose(1, 2).reshape(B, T, C))))
        x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        xs.append(x)
    return xs, cls


def topk_mask(score, frac):
    k = int(round(frac * score.size(1)))
    return torch.zeros_like(score, dtype=torch.bool).scatter_(1, score.topk(k, 1).indices, True)


def main(n_images=5000, bs=100):
    vit = load_backbone()
    tf = data.eval_transform(vit)
    val = data.ImageNetFiles('val', tf)
    idx = torch.randperm(len(val), generator=torch.Generator().manual_seed(0))[:n_images].tolist()
    loader = DataLoader(Subset(val, idx), batch_size=bs, num_workers=8)
    L = len(vit.blocks)

    top50_mass = torch.zeros(L)
    entropy = torch.zeros(L)
    cos = torch.zeros(L, L)
    eps_blk = torch.zeros(2, L)            # [inattentive, attentive] per block
    eps_stage = torch.zeros(2, len(PROBE))  # over the 3 blocks of each ToFe stage
    cos_stage = torch.zeros(2, len(PROBE))  # cos(X_l, X_l+3): direction kept?
    norm_stage = torch.zeros(2, len(PROBE))  # ||X_l+3|| / ||X_l||: magnitude growth
    overlap = {'J(4,10)': 0., 'J(7,10)': 0., 'J(4,7)': 0., 'recall@4 of top25%@10': 0.,
               'block-10 attn mass outside top50%@4': 0.}
    n = 0
    for img, _ in loader:
        xs, cls = trace(vit, img.to(DEV))
        B = img.size(0)
        n += B
        N = cls[0].size(1)
        for l in range(L):
            p = cls[l] / cls[l].sum(1, keepdim=True)
            entropy[l] += (-(p * p.clamp_min(1e-12).log()).sum(1) / torch.log(torch.tensor(float(N)))).sum().cpu()
            top50_mass[l] += p.topk(N // 2, 1).values.sum(1).sum().cpu()
            att = topk_mask(cls[l], 0.5)
            x0, x1 = xs[l][:, 1:], xs[l + 1][:, 1:]
            e = (x1 - x0).norm(dim=-1) / x0.norm(dim=-1)
            eps_blk[0, l] += (e * ~att).sum().cpu() / (N - att[0].sum().item())
            eps_blk[1, l] += (e * att).sum().cpu() / att[0].sum().item()
        u = [F.normalize(xs[l + 1][:, 1:], dim=-1) for l in range(L)]
        for a in range(L):
            for b in range(a, L):
                cos[a, b] += (u[a] * u[b]).sum(-1).mean(1).sum().cpu()
        for s, l in enumerate(PROBE):
            att = topk_mask(cls[l], 0.5)
            x0, x3 = xs[l][:, 1:], xs[l + 3][:, 1:]
            metrics = ((x3 - x0).norm(dim=-1) / x0.norm(dim=-1), F.cosine_similarity(x3, x0, dim=-1),
                       x3.norm(dim=-1) / x0.norm(dim=-1))
            for acc, e in zip((eps_stage, cos_stage, norm_stage), metrics):
                acc[0, s] += (e * ~att).sum().cpu() / (N - att[0].sum().item())
                acc[1, s] += (e * att).sum().cpu() / att[0].sum().item()
        m4, m7, m10 = (topk_mask(cls[l], 0.5) for l in PROBE)
        jac = lambda a, b: ((a & b).sum(1) / (a | b).sum(1)).sum().item()
        overlap['J(4,10)'] += jac(m4, m10)
        overlap['J(7,10)'] += jac(m7, m10)
        overlap['J(4,7)'] += jac(m4, m7)
        t25 = topk_mask(cls[9], 0.25)
        overlap['recall@4 of top25%@10'] += ((t25 & m4).sum(1) / t25.sum(1)).sum().item()
        p10 = cls[9] / cls[9].sum(1, keepdim=True)
        overlap['block-10 attn mass outside top50%@4'] += (p10 * ~m4).sum(1).sum().item()

    cos = cos / n
    cos = cos + cos.T - torch.diag(cos.diag())
    res = {
        'n_images': n,
        'top50_attention_mass': (top50_mass / n).tolist(),
        'normalized_entropy': (entropy / n).tolist(),
        'cosine_similarity': cos.tolist(),
        'eps_per_block': {'inattentive': (eps_blk[0] / n).tolist(), 'attentive': (eps_blk[1] / n).tolist()},
        'eps_per_stage': {'blocks': ['4-6', '7-9', '10-12'], 'inattentive': (eps_stage[0] / n).tolist(),
                          'attentive': (eps_stage[1] / n).tolist()},
        'cos_per_stage': {'inattentive': (cos_stage[0] / n).tolist(), 'attentive': (cos_stage[1] / n).tolist()},
        'norm_ratio_per_stage': {'inattentive': (norm_stage[0] / n).tolist(), 'attentive': (norm_stage[1] / n).tolist()},
        'overlap': {k: v / n for k, v in overlap.items()},
    }
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    with open(os.path.join(ROOT, 'runs', 'motivation.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'cosine_similarity'}, indent=1))
    print('cos(l, l+1):', [round(cos[l, l + 1].item(), 3) for l in range(L - 1)])
    plots(vit, val, tf, res)


def denorm(t, cfg):
    m = torch.tensor(cfg['mean']).view(3, 1, 1)
    s = torch.tensor(cfg['std']).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def first_of(val, label):
    return next(i for i, (_, t) in enumerate(val.samples) if t == label)


def plots(vit, val, tf, res):
    os.makedirs(FIGS, exist_ok=True)
    cfg = timm.data.resolve_data_config({}, model=vit)
    L = len(vit.blocks)
    blocks = list(range(1, L + 1))

    # M1 + M4 summary: two separate panels, no shared/dual axis
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.8))
    ax[0].plot(blocks, [100 * v for v in res['top50_attention_mass']], marker='o', color=ps.SERIES[0])
    ax[0].axhline(50, color=ps.MUTED, lw=1, ls='--')
    ax[0].text(L, 51, 'uniform attention', ha='right', va='bottom', color=ps.MUTED, fontsize=8)
    ax[0].set(xlabel='block', ylabel='[CLS] attention mass on top-50% tokens (%)', xticks=blocks)
    ax[0].set_title('(a) [CLS] attention concentration', loc='left')
    for key, c, half in (('attentive', ps.SERIES[0], 'top'), ('inattentive', ps.SERIES[1], 'bottom')):
        ys = res['eps_per_block'][key]
        ax[1].plot(blocks, ys, marker='o', color=c)
        ax[1].annotate(f'{key} ({half} 50%)', (L, ys[-1]), xytext=(4, 0), textcoords='offset points',
                       va='center', fontsize=8, color=ps.INK2)
    ax[1].set(xlabel='block', ylabel=r'relative change  $\|\Delta X\|/\|X\|$', xticks=blocks)
    ax[1].set_title('(b) per-token feature change', loc='left')
    ax[1].set_xlim(0.5, L + 4.5)
    fig.savefig(os.path.join(FIGS, 'motivation_curves.png'))
    plt.close(fig)

    # M3: cosine-similarity matrix (Fig. 4b)
    cos = torch.tensor(res['cosine_similarity'])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cos, cmap=ps.SEQ, vmin=0, vmax=1)
    ax.grid(False)
    ax.set(xticks=range(L), yticks=range(L), xticklabels=blocks, yticklabels=blocks, xlabel='block', ylabel='block')
    for a in range(L):
        for b in range(L):
            ax.text(b, a, f'{cos[a, b]:.2f}', ha='center', va='center', fontsize=5.5,
                    color='white' if cos[a, b] > 0.6 else ps.INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='mean token cosine similarity')
    fig.savefig(os.path.join(FIGS, 'cosine_similarity.png'))
    plt.close(fig)

    # Fig. 3 replica: keep top-50% tokens by [CLS] attention at blocks 4/7/10
    examples = {'stopwatch': 826, 'Siberian husky': 250, 'tench': 0}
    fig, axs = plt.subplots(len(examples), 4, figsize=(7.2, 5.6))
    for r, (name, lab) in enumerate(examples.items()):
        img = val[first_of(val, lab)][0]
        _, cls = trace(vit, img.unsqueeze(0).to(DEV))
        pic = denorm(img, cfg)
        axs[r, 0].imshow(pic)
        axs[r, 0].set_ylabel(name, color=ps.INK2)
        for c, l in enumerate(PROBE):
            keep = topk_mask(cls[l], 0.5)[0].view(14, 14).float()
            m = keep.repeat_interleave(16, 0).repeat_interleave(16, 1).cpu().numpy()[..., None]
            axs[r, c + 1].imshow(pic * m)
            if r == 0:
                axs[r, c + 1].set_title(f'block {l + 1}')
        if r == 0:
            axs[r, 0].set_title('image')
        for a in axs[r]:
            ps.image_axes(a)
    fig.savefig(os.path.join(FIGS, 'cls_top50_masks.png'))
    plt.close(fig)

    # Fig. 4a replica: [CLS] attention maps at blocks 2/5/7/9/10
    img = val[first_of(val, 330)][0]  # wood rabbit
    _, cls = trace(vit, img.unsqueeze(0).to(DEV))
    fig, axs = plt.subplots(1, 6, figsize=(9.6, 1.9))
    axs[0].imshow(denorm(img, cfg))
    axs[0].set_title('image')
    for a, l in zip(axs[1:], [1, 4, 6, 8, 9]):
        m = cls[l][0].float().cpu()
        # a few "attention sink" tokens dominate deep blocks; clip so the rest stays visible
        a.imshow(m.view(14, 14), cmap=ps.SEQ, vmin=0, vmax=m.quantile(0.975).item())
        a.set_title(f'block {l + 1}')
        top = m.topk(2).values / m.sum()
        a.set_xlabel(f'top-2 tokens: {100 * top.sum():.0f}%', color=ps.INK2, fontsize=8)
    for a in axs:
        ps.image_axes(a)
    fig.savefig(os.path.join(FIGS, 'cls_attention_maps.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
