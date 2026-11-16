"""Quick feasibility test for reproducing ToFe on a single RTX 5070.

T1  Eq. (4) FLOPs vs. an operator-level counter (fvcore)
T2  Correctness: ToFe with "keep all" must equal the dense DeiT-S output
T3  Dense DeiT-S accuracy on the ImageNet subsets (1000-way head)
T4  Throughput: dense vs. ToFe gather inference at the paper's keep ratios
T5  Mini training (frozen backbone): selectors + approximators, 50% FLOPs budget,
    then ToFe vs. training-free EViT at the same token budget
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
from baselines import evit_flops, evit_forward
from common import AMP, DEV, evaluate, load_backbone, throughput
from tofe import ToFe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--loss', choices=['ce', 'kd'], default='ce')
    args = ap.parse_args()
    torch.manual_seed(0)

    vit = load_backbone()
    model = ToFe(vit).to(DEV)
    N = model.num_patches
    paper_keep = [round(r * N) for r in (0.49, 0.25, 0.14)]  # Table VII optimized keep ratio

    # ---------------------------------------------------------------- T1
    print('== T1: FLOPs (GMACs) ==')
    from fvcore.nn import FlopCountAnalysis
    for b in vit.blocks:
        b.attn.fused_attn = False  # fvcore cannot see inside the fused SDPA kernel
    fv = FlopCountAnalysis(vit, torch.randn(1, 3, 224, 224, device=DEV))
    fv.unsupported_ops_warnings(False)
    fv.uncalled_modules_warnings(False)
    fv_total = fv.total() / 1e9
    for b in vit.blocks:
        b.attn.fused_attn = True
    print(f'dense DeiT-S  Eq.(4): {model.dense_flops() / 1e9:.3f}  fvcore: {fv_total:.3f}  (paper: 4.6)')
    print(f'ToFe keep={paper_keep}: {model.flops(paper_keep) / 1e9:.3f}  (paper Table VII: 2.3)')
    print(f'EViT  keep={paper_keep}: {evit_flops(vit, model.locs, paper_keep) / 1e9:.3f}')
    extra = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith('vit.'))
    print(f'params: backbone {sum(p.numel() for p in vit.parameters()) / 1e6:.2f}M, ToFe modules +{extra / 1e6:.3f}M')

    # ---------------------------------------------------------------- T2
    print('\n== T2: correctness (keep all tokens == dense) ==')
    x = torch.randn(4, 3, 224, 224, device=DEV)
    with torch.no_grad():
        ref = vit(x)
        out_inf = model.forward_infer(x, keep_counts=[N] * 3)
        for s in model.selectors:  # force "keep" for the masked path
            s.net[-1].bias.data[0] += 100
        out_tr, masks, _ = model.forward_train(x, hard_eval=True)
        for s in model.selectors:
            s.net[-1].bias.data[0] -= 100
    print(f'max |dense - infer(all)| = {(ref - out_inf).abs().max().item():.2e}')
    print(f'max |dense - train(all)| = {(ref - out_tr).abs().max().item():.2e}, kept={[int(m.sum(1)[0]) for m in masks]}')

    # ---------------------------------------------------------------- T3
    print('\n== T3: dense DeiT-S accuracy (1000-way top-1/top-5) ==')
    tf_eval = data.eval_transform(vit)
    vals = {n: DataLoader(data.build([n], 'val', tf_eval), batch_size=128, num_workers=args.workers,
                          pin_memory=True, persistent_workers=True) for n in data.SUBSETS}
    with torch.no_grad():
        for n, ld in vals.items():
            t = time.perf_counter()
            a1, a5 = evaluate(vit, ld)
            print(f'{n:10s} n={len(ld.dataset)}: top1 {a1:.2f}  top5 {a5:.2f}  ({time.perf_counter() - t:.1f}s)')

    # ---------------------------------------------------------------- T4
    print('\n== T4: throughput, bs=128, bf16 (img/s) ==')
    tp_dense = throughput(vit)
    tp_tofe = throughput(lambda z: model.forward_infer(z, keep_counts=paper_keep))
    tp_evit = throughput(lambda z: evit_forward(vit, z, model.locs, paper_keep))
    print(f'dense {tp_dense:.0f} | ToFe {tp_tofe:.0f} ({100 * (tp_tofe / tp_dense - 1):+.1f}%) | '
          f'EViT {tp_evit:.0f} ({100 * (tp_evit / tp_dense - 1):+.1f}%)   [paper DeiT-S: 2383.9 -> 5477.3]')

    # ---------------------------------------------------------------- T5
    print('\n== T5: mini training, frozen backbone, target = 50% FLOPs ==')
    train_ds = data.build(list(data.SUBSETS), 'train', data.train_transform(vit))
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    params = [p for n_, p in model.named_parameters() if not n_.startswith('vit.')]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), eta_min=2e-5)
    target = 0.5 * model.dense_flops() / 1e9
    lam_cls, lam_apr, lam_flops = 1.0, 2.0, 5.0
    torch.cuda.reset_peak_memory_stats()
    for ep in range(args.epochs):
        model.train()
        t = time.perf_counter()
        kept_sum = torch.zeros(3, device=DEV)
        nb = 0
        for img, y in loader:
            img, y = img.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(**AMP):
                teacher, t_logits = model.teacher_states(img)
                logits, masks, approx = model.forward_train(img)
                if args.loss == 'ce':
                    l_cls = F.cross_entropy(logits.float(), y)
                else:  # distill the dense model's 1000-way prediction instead of subset labels
                    l_cls = F.kl_div(logits.float().log_softmax(-1), t_logits.float().log_softmax(-1),
                                     reduction='batchmean', log_target=True)
                l_apr = 0.
                for m, xa, xt in zip(masks, approx, teacher):
                    frz = 1 - m
                    err = (xa.float() - xt.float()).pow(2).mean(-1)
                    l_apr = l_apr + ((frz * err).sum(1) / frz.sum(1).clamp_min(1)).mean()
                counts = [m.sum(1) for m in masks]
                l_flops = ((model.flops(counts) / 1e9 - target) ** 2).mean()
                loss = lam_cls * l_cls + lam_apr * l_apr + lam_flops * l_flops
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            kept_sum += torch.stack([c.mean() for c in counts]).detach()
            nb += 1
        dt = time.perf_counter() - t
        keep_avg = (kept_sum / nb).tolist()
        print(f'ep{ep}: loss {loss.item():.3f} (cls {l_cls.item():.3f} apr {l_apr.item():.4f} flops {l_flops.item():.4f}) '
              f'keep {[round(k, 1) for k in keep_avg]} -> {model.flops(keep_avg) / 1e9:.2f} GMACs | '
              f'{dt:.0f}s, {nb * args.bs / dt:.0f} img/s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')

    model.eval()
    os.makedirs('runs', exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('vit.')},
               'runs/feasibility_tofe_modules.pt')
    keep_n = [round(k) for k in keep_avg]
    print(f'\nlearned N^s = {keep_n} (ratios {[round(k / N, 2) for k in keep_n]}) -> {model.flops(keep_n) / 1e9:.2f} GMACs; '
          f'paper N^s = {paper_keep}: ToFe {model.flops(paper_keep) / 1e9:.2f} vs EViT '
          f'{evit_flops(vit, model.locs, paper_keep) / 1e9:.2f} GMACs')
    with torch.no_grad():
        for n, ld in vals.items():
            r_learn = evaluate(lambda z: model.forward_infer(z, keep_counts=keep_n), ld)
            r_tofe = evaluate(lambda z: model.forward_infer(z, keep_counts=paper_keep), ld)
            r_evit = evaluate(lambda z: evit_forward(vit, z, model.locs, paper_keep), ld)
            r_prune = evaluate(lambda z: evit_forward(vit, z, model.locs, paper_keep, fuse=False), ld)
            print(f'{n:10s} ToFe(learned N^s) {r_learn[0]:.2f}/{r_learn[1]:.2f} | ToFe(paper N^s) '
                  f'{r_tofe[0]:.2f}/{r_tofe[1]:.2f} | EViT(fuse) {r_evit[0]:.2f}/{r_evit[1]:.2f} | '
                  f'top-k CLS-attn prune {r_prune[0]:.2f}/{r_prune[1]:.2f}')
        # instance-adaptive (B=1): every image chooses its own number of tokens
        ld1 = DataLoader(data.build(['imagewoof'], 'val', tf_eval), batch_size=1, shuffle=True,
                         generator=torch.Generator().manual_seed(0), num_workers=2)
        correct, macs, counts = 0, [], []
        for i, (img, y) in enumerate(ld1):
            if i == 500:
                break
            with torch.autocast(**AMP):
                out, info = model.forward_infer(img.to(DEV), return_info=True)
            k = [t.size(1) for t in info]
            counts.append(k)
            macs.append(model.flops(k) / 1e9)
            correct += int(out.argmax(-1).item() == y.item())
        c = torch.tensor(counts, dtype=torch.float)
        print(f'instance-adaptive (imagewoof, 500 imgs): top1 {correct / 5:.1f} | GMACs mean {sum(macs) / len(macs):.2f} '
              f'(min {min(macs):.2f}, max {max(macs):.2f}) | tokens/stage mean {c.mean(0).tolist()} std {c.std(0).tolist()}')


if __name__ == '__main__':
    main()
