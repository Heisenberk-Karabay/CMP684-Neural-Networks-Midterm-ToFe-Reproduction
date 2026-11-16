"""Why does token reusing not pay off at our training scale? (writes runs/diagnostics.json)

D1  same trained model, reuse on/off at inference, at several kept-token allocations
D2  relative error of the approximated frozen tokens w.r.t. the dense teacher, vs identity reuse
D3  selector confidence, and irreversible pruning by selector score vs by [CLS] attention
"""
import json
import os

import torch

import data
from analyze_tofe import load_run
from baselines import _attention
from common import AMP, DEV, RUNS, checkpoint_path, evaluate
from tofe import embed, gather_tokens, masked_block

RUN_NAMES = ['tofe_b50', 'tofe_b50_e15', 'tofe_b50_ft3']
ALLOCS = {'paper (0.49,0.25,0.14)': [96, 49, 27], 'no-reuse run (learned)': [140, 31, 17]}


def subset(n=5000):
    cache = data.CachedVal(1)
    idx = torch.randperm(len(cache.y), generator=torch.Generator().manual_seed(0))[:n].sort().values
    x = torch.from_numpy(cache.x[idx.numpy()])
    for i in range(0, n, 250):
        yield (x[i:i + 250].to(DEV).float() - cache.mean) / cache.std, cache.y[idx[i:i + 250]]


@torch.no_grad()
def stage_trace(model, img):
    """Masked forward (deterministic decisions) returning per-stage inputs, masks, approximations."""
    x = embed(model.vit, img)
    for blk in model.vit.blocks[:model.locs[0]]:
        x = blk(x)
    ins, masks, apx = [], [], []
    for s, (lo, hi) in enumerate(model.stages):
        m = model._mask(model._logits(s, x[:, 1:]), hard_eval=True)
        policy = torch.cat([torch.ones_like(m[:, :1]), m], 1)
        x_in = x
        for blk in model.vit.blocks[lo:hi]:
            x = masked_block(blk, x, policy)
        if s < len(model.approximators):
            xa = model.approximators[s](x_in[:, 1:], frozen=1 - m)
            p = m.unsqueeze(-1).to(x.dtype)
            x = torch.cat([x[:, :1], p * x[:, 1:] + (1 - p) * xa], 1)
            ins.append(x_in[:, 1:])
            apx.append(xa)
        masks.append(m)
    return ins, masks, apx


@torch.no_grad()
def prune_forward(model, img, keep, by):
    """Irreversible pruning at the ToFe locations, ranking tokens by selector score or [CLS] attention."""
    vit = model.vit
    x = embed(vit, img)
    for i, blk in enumerate(vit.blocks):
        if i in model.locs:
            s = model.locs.index(i)
            if by == 'selector':
                lg = model._logits(s, x[:, 1:])
                score = lg[..., 0] - lg[..., 1]
            else:
                score = _attention(blk, x, need_cls=True)[1]
            x = torch.cat([x[:, :1], gather_tokens(x[:, 1:], score.topk(keep[s], 1).indices)], 1)
        x = blk(x)
    return vit.forward_head(vit.norm(x))


def main():
    out = {}
    for name in RUN_NAMES:
        model, keep = load_run(name)
        model.load_state_dict(torch.load(checkpoint_path(name))['state'], strict=False)
        res = {'learned_keep': keep, 'D1': {}}
        for label, k in [('learned', keep)] + list(ALLOCS.items()):
            row = {'keep': k, 'gmacs': model.flops(k) / 1e9}
            for reuse in (True, False):
                if not reuse and any(b > a for a, b in zip(k, k[1:])):
                    continue  # w/o reuse the kept sets must be nested
                model.reuse = reuse
                row['reuse' if reuse else 'no_reuse'] = evaluate(
                    lambda z: model.forward_infer(z, keep_counts=k), data.CachedVal(256))[0]
            model.reuse = True
            res['D1'][label] = row

        err = torch.zeros(2, 2)
        probs, n_img, prune = [], 0, {'selector': 0, 'cls_attention': 0}
        nested = [keep[0], min(keep[0], keep[1]), min(keep[0], keep[1], keep[2])]
        for img, y in subset():
            with torch.autocast(**AMP):
                t_states, _ = model.teacher_states(img)
                ins, masks, apx = stage_trace(model, img)
                for by in prune:
                    prune[by] += (prune_forward(model, img, nested, by).argmax(-1).cpu() == y).sum().item()
            for s in range(2):
                frz = (1 - masks[s]).bool()
                t = t_states[s].float()
                for j, pred in enumerate((ins[s], apx[s])):
                    err[s, j] += ((pred.float() - t).norm(dim=-1) / t.norm(dim=-1))[frz].mean().cpu() * len(y)
            x = embed(model.vit, img)
            for blk in model.vit.blocks[:model.locs[0]]:
                x = blk(x)
            probs.append(model._logits(0, x[:, 1:]).softmax(-1)[..., 0].flatten().cpu())
            n_img += len(y)
        p = torch.cat(probs)
        res['D2'] = {f'stage{s + 1}': {'identity': (err[s, 0] / n_img).item(), 'approximator': (err[s, 1] / n_img).item()}
                     for s in range(2)}
        res['D3'] = {'keep_prob_quantiles_10_25_50_75_90': p.quantile(torch.tensor([.1, .25, .5, .75, .9])).tolist(),
                     'fraction_between_0.3_0.7': ((p > 0.3) & (p < 0.7)).float().mean().item(),
                     'nested_keep': nested,
                     'prune_top1': {k: 100 * v / n_img for k, v in prune.items()}}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    with open(os.path.join(RUNS, 'diagnostics.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
