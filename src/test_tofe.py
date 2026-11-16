"""Unit tests for the ToFe implementation and the baselines (run: python src/test_tofe.py)."""
import torch

from baselines import evit_flops, evit_forward, tome_flops, tome_forward
from common import DEV, load_backbone
from tofe import ToFe


def keep_everything(model):
    for s in model.selectors:
        last = [m for m in s.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.bias.data[0] += 100


@torch.no_grad()
def main():
    torch.manual_seed(0)
    vit = load_backbone()
    x = torch.randn(4, 3, 224, 224, device=DEV)
    ref = vit(x)
    N = vit.patch_embed.num_patches

    # 1) keep-all must reproduce the dense model on both paths, for every module variant
    for approx in ['mlp', 'identity', 'dwconv', 'block']:
        for selector in ['mlp', 'dynamicvit']:
            m = ToFe(vit, approx=approx, selector=selector).to(DEV).eval()
            keep_everything(m)
            d_inf = (m.forward_infer(x, keep_counts=[N] * 3) - ref).abs().max().item()
            d_tr = (m.forward_train(x, hard_eval=True)[0] - ref).abs().max().item()
            assert d_inf < 1e-3 and d_tr < 1e-3, (approx, selector, d_inf, d_tr)
    print('keep-all == dense for all approximator/selector variants: OK')

    # 2) masked training forward == gather inference forward for the same hard decisions
    for reuse in [True, False]:
        for approx in ['mlp', 'dwconv', 'block']:
            m = ToFe(vit, reuse=reuse, approx=approx).to(DEV).eval()
            for s in m.selectors:  # random but deterministic decisions
                for layer in s.modules():
                    if isinstance(layer, torch.nn.Linear):
                        torch.nn.init.normal_(layer.weight, std=0.5)
            for a in m.approximators:  # non-trivial approximators
                for p in a.parameters():
                    torch.nn.init.normal_(p, std=0.02)
            xb = x[:1]
            out_tr, masks, _ = m.forward_train(xb, hard_eval=True)
            out_inf, info = m.forward_infer(xb, return_info=True)  # instance-adaptive, threshold 0.5
            kept_tr = [int(mk.sum()) for mk in masks]
            kept_inf = [k.size(1) for k in info]
            assert kept_tr == kept_inf, (kept_tr, kept_inf)
            diff = (out_tr - out_inf).abs().max().item()
            assert diff < 2e-3, (reuse, approx, diff)
    print('masked training forward == gather inference forward (reuse on/off, mlp/dwconv/block): OK')

    # 3) no reuse: a token frozen once is never kept again
    m = ToFe(vit, reuse=False).to(DEV).eval()
    _, info = m.forward_infer(x, keep_counts=[96, 49, 27], return_info=True)
    for a, b in zip(info[:-1], info[1:]):
        for i in range(x.size(0)):
            assert set(b[i].tolist()) <= set(a[i].tolist())
    print('no-reuse keeps nested token sets: OK')

    # 4) baselines: zero reduction == dense, FLOPs bookkeeping
    assert (tome_forward(vit, x, 0) - ref).abs().max() < 1e-3
    assert (evit_forward(vit, x, [3, 6, 9], [N] * 3, fuse=False) - ref).abs().max() < 1e-3
    assert abs(tome_flops(vit, 0) - ToFe(vit).dense_flops()) < 1
    print('ToMe(r=0) == dense, EViT(keep all) == dense: OK')
    for r in [8, 12, 13, 14, 16]:
        print(f'  ToMe r={r}: {tome_flops(vit, r) / 1e9:.2f} GMACs')
    print(f'  EViT keep (0.7 rate) [137,96,67]: {evit_flops(vit, [3, 6, 9], [137, 96, 67]) / 1e9:.2f} GMACs')
    print('all tests passed')


if __name__ == '__main__':
    main()
