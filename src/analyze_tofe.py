"""Analysis of a trained ToFe run (usage: python src/analyze_tofe.py <run-name>).

  Table VIII  batch-adaptive accuracy for several batch sizes + instance-adaptive (B=1)
  Table X     reused tokens per stage and attention energy proportion (AEP)
  Fig. 9      distribution of kept tokens per stage (instance-adaptive)
  Fig. 8/10   per-stage kept / frozen / reused tokens and token-usage heat map
"""
import json
import os
import sys
import time

import timm
import torch

import data
import plotstyle as ps
from common import AMP, DEV, RUNS, checkpoint_path, evaluate, load_backbone
from motivation import FIGS, denorm, first_of, trace
from plotstyle import plt
from tofe import ToFe


def load_run(name):
    ck = torch.load(checkpoint_path(name), map_location=DEV)
    a = ck['args']
    vit = load_backbone()
    N = vit.patch_embed.num_patches
    fixed = [round(float(r) * N) for r in a['fixed_keep'].split(',')] if a['fixed_keep'] else None
    model = ToFe(vit, reuse=not a['no_reuse'], approx=a['approx'], selector=a['selector'],
                 selector_depth=a['selector_depth'], fixed_keep=fixed).to(DEV)
    missing, unexpected = model.load_state_dict(ck['state'], strict=False)
    assert not unexpected and all(k.startswith('vit.') for k in missing), (missing, unexpected)
    model.eval()
    return model, fixed or [round(k) for k in ck['keep_avg']]


@torch.no_grad()
def instance_stats(model, loader):
    """Instance-adaptive decisions for the whole set, computed with the (equivalent, batched)
    masked forward. Returns accuracy, per-image kept counts, reused counts and AEP."""
    S = len(model.locs)
    correct, kept, reused, aep = 0, [], [], []
    for img, y in loader:
        img, y = img.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        with torch.autocast(**AMP):
            out, masks, _ = model.forward_train(img, hard_eval=True)
            _, cls = trace(model.vit, img)
        correct += (out.argmax(-1) == y).sum().item()
        kept.append(torch.stack([m.sum(1) for m in masks], 1).cpu())
        reused.append(torch.stack([torch.zeros_like(masks[0][:, 0])] +
                                  [(masks[s] * (1 - masks[s - 1])).sum(1) for s in range(1, S)], 1).cpu())
        p = [cls[l].float() / cls[l].float().sum(1, keepdim=True) for l in model.locs]
        aep.append(torch.stack([(p[s] * masks[s]).sum(1) for s in range(S)], 1).cpu())
    n = sum(k.size(0) for k in kept)
    return 100 * correct / n, torch.cat(kept), torch.cat(reused), torch.cat(aep)


def main(name):
    model, keep_n = load_run(name)
    vit = model.vit
    N = model.num_patches
    tf = data.eval_transform(vit)
    val = data.ImageNetFiles('val', tf)
    res = {'run': name, 'batch_keep_counts': keep_n}

    # Table VIII: batch size (batch-adaptive) vs instance-adaptive
    res['batch_size'] = {}
    for bs in [16, 32, 64, 128]:
        top1, _ = evaluate(lambda z: model.forward_infer(z, keep_counts=keep_n), data.CachedVal(bs))
        res['batch_size'][bs] = {'top1': top1, 'gmacs': model.flops(keep_n) / 1e9}
        print(f'batch-adaptive bs={bs}: {top1:.2f}', flush=True)
    top1, kept, reused, aep = instance_stats(model, data.CachedVal(256))
    macs = torch.tensor([model.flops(k.tolist()) / 1e9 for k in kept])
    res['batch_size'][1] = {'top1': top1, 'gmacs': macs.mean().item()}
    print(f'instance-adaptive: {top1:.2f} @ {macs.mean():.3f} GMACs', flush=True)

    # real single-image latency of the instance-adaptive path
    imgs = [img for _, (img, _) in zip(range(300), data.CachedVal(1))]
    with torch.no_grad(), torch.autocast(**AMP):
        for f in (lambda z: vit(z), lambda z: model.forward_infer(z)):
            for z in imgs[:20]:
                f(z)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for z in imgs:
                f(z)
            torch.cuda.synchronize()
            res.setdefault('latency_ms_bs1', []).append(1000 * (time.perf_counter() - t) / len(imgs))
    print('bs=1 latency dense / ToFe (ms):', res['latency_ms_bs1'])

    # Table X + Fig. 9 statistics
    res['stages'] = [{'remaining_mean': kept[:, s].mean().item(), 'remaining_std': kept[:, s].std().item(),
                      'reused_mean': reused[:, s].mean().item(),
                      'reused_ratio': (reused[:, s].sum() / kept[:, s].sum().clamp_min(1)).item(),
                      'AEP': aep[:, s].mean().item()} for s in range(len(model.locs))]
    res['gmacs_instance'] = {'mean': macs.mean().item(), 'min': macs.min().item(), 'max': macs.max().item()}
    print(json.dumps(res['stages'], indent=1))
    with open(os.path.join(RUNS, name, 'analysis.json'), 'w') as f:
        json.dump(res, f, indent=1)

    os.makedirs(FIGS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bins = torch.arange(0, N + 2, 4) - 0.5
    for s in range(len(model.locs)):
        ax.hist(kept[:, s].numpy(), bins=bins.numpy(), color=ps.SERIES[s], alpha=0.75, label=f'stage {s + 1}',
                edgecolor='white', linewidth=0.4)
    ax.set(xlabel='kept patch tokens per image', ylabel='images')
    ax.legend()
    fig.savefig(os.path.join(FIGS, f'{name}_token_hist.png'))
    plt.close(fig)

    # Fig. 8 / Fig. 10: kept / frozen / reused tokens per stage
    order = kept[:, 0].argsort()
    picks = {'easy': order[len(order) // 50].item(), 'normal': order[len(order) // 2].item(),
             'complex': order[-len(order) // 50].item(), 'Siberian husky': first_of(val, 250),
             'stopwatch': first_of(val, 826)}
    cfg = timm.data.resolve_data_config({}, model=vit)
    S = len(model.locs)
    fig, axs = plt.subplots(len(picks), S + 2, figsize=(1.75 * (S + 2), 1.85 * len(picks)))
    for r, (label, i) in enumerate(picks.items()):
        img = val[i][0]
        with torch.no_grad():
            _, masks, _ = model.forward_train(img[None].to(DEV), hard_eval=True)
        pic = denorm(img, cfg)
        g = vit.patch_embed.grid_size[0]
        up = lambda t: t.view(g, g).repeat_interleave(16, 0).repeat_interleave(16, 1).cpu().numpy()[..., None]
        axs[r, 0].imshow(pic)
        axs[r, 0].set_ylabel(label, color=ps.INK2)
        usage = torch.ones(N, device=DEV)
        for s, m in enumerate(masks):
            m = m[0]
            usage += m
            keep = up(m)
            axs[r, s + 1].imshow(pic * keep + (1 - keep) * (0.25 * pic + 0.75))  # frozen tokens washed out
            if s > 0:
                for t in (m * (1 - masks[s - 1][0])).nonzero().flatten().tolist():
                    axs[r, s + 1].add_patch(plt.Rectangle(((t % g) * 16 - 0.5, (t // g) * 16 - 0.5), 16, 16,
                                                          fill=False, edgecolor=ps.SERIES[3], linewidth=1.2))
            axs[r, s + 1].set_xlabel(f'{int(m.sum())} ({100 * m.mean():.0f}%)', color=ps.INK2, fontsize=8)
            if r == 0:
                axs[r, s + 1].set_title(f'stage {s + 1}')
        im = axs[r, -1].imshow(usage.view(g, g).cpu(), cmap=ps.SEQ, vmin=1, vmax=S + 1)
        if r == 0:
            axs[r, 0].set_title('image')
            axs[r, -1].set_title('token usage')
        for a in axs[r]:
            ps.image_axes(a)
    fig.colorbar(im, ax=axs[:, -1], fraction=0.05, pad=0.02, ticks=range(1, S + 2), label='times used')
    fig.savefig(os.path.join(FIGS, f'{name}_token_stages.png'))
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
