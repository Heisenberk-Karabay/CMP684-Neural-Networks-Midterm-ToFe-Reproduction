"""Dense DeiT-S and training-free baselines on the full ImageNet-1k validation set.
Writes runs/baselines.json (one record per method/setting) for Table I and Fig. 6."""
import json
import os

import torch

import data
from baselines import evit_flops, evit_forward, tome_flops, tome_forward
from common import RUNS, evaluate, load_backbone, throughput
from tofe import embed, gather_tokens

LOCS = [3, 6, 9]


@torch.no_grad()
def random_forward(vit, img, keep_counts):
    """Sanity baseline: irreversible pruning of uniformly random tokens at the ToFe locations."""
    x = embed(vit, img)
    for i, blk in enumerate(vit.blocks):
        if i in LOCS:
            k = keep_counts[LOCS.index(i)]
            idx = torch.rand(x.size(0), x.size(1) - 1, device=x.device).topk(k, 1).indices
            x = torch.cat([x[:, :1], gather_tokens(x[:, 1:], idx)], 1)
        x = blk(x)
    return vit.forward_head(vit.norm(x))


def main():
    torch.manual_seed(0)
    vit = load_backbone()
    N = vit.patch_embed.num_patches
    loader = data.CachedVal(256)
    path = os.path.join(RUNS, 'baselines.json')
    results = json.load(open(path)) if os.path.exists(path) else []
    done = {(r['method'], str(r['setting'])) for r in results}

    def run(method, setting, fn, macs):
        if (method, str(setting)) in done:
            return
        top1, top5 = evaluate(fn, loader)
        r = dict(method=method, setting=setting, gmacs=macs / 1e9, top1=top1, top5=top5, throughput=throughput(fn))
        print(json.dumps(r), flush=True)
        results.append(r)
        with open(path, 'w') as f:
            json.dump(results, f, indent=1)

    from tofe import ToFe
    run('dense', '-', vit, ToFe(vit).dense_flops())
    paper_keep = [round(r * N) for r in (0.49, 0.25, 0.14)]
    for rate in [0.9, 0.8, 0.7, 0.6, 0.5]:  # EViT keep rate per stage (relative to the previous stage)
        keep = [round(N * rate ** (s + 1)) for s in range(3)]
        run('EViT', keep, lambda z, k=keep: evit_forward(vit, z, LOCS, k), evit_flops(vit, LOCS, keep))
    run('EViT', paper_keep, lambda z: evit_forward(vit, z, LOCS, paper_keep), evit_flops(vit, LOCS, paper_keep))
    for r in [4, 8, 12, 14, 16, 18]:
        run('ToMe', r, lambda z, r=r: tome_forward(vit, z, r), tome_flops(vit, r))
    tofe = ToFe(vit, approx='identity')
    rand_macs = tofe.flops(paper_keep) - sum(s.flops(N) for s in tofe.selectors)
    run('Random', paper_keep, lambda z: random_forward(vit, z, paper_keep), rand_macs)


if __name__ == '__main__':
    main()
