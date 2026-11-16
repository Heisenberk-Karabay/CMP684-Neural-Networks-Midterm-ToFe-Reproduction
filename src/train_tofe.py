"""Train ToFe (selectors + approximators, optionally the backbone) on ImageNet-1k and
evaluate batch-adaptive inference on the full 50k validation set.

Loss (Eq. 15): L = lam_cls * CE + lam_apr * L_apr (Eq. 10) + lam_flops * L_FLOPs (Eq. 14)
Examples:
  python src/train_tofe.py --name tofe_b50                       # main setting, 50% FLOPs
  python src/train_tofe.py --name noreuse_b50 --no-reuse         # Table V
  python src/train_tofe.py --name apx_identity --approx identity # Table VI
  python src/train_tofe.py --name fixed --fixed-keep 0.5,0.25,0.125  # Table VII
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import data
from common import AMP, DEV, RUNS, checkpoint_path, evaluate, load_backbone, throughput
from tofe import ToFe


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--budget', type=float, default=0.5, help='target FLOPs as a fraction of the dense model')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3, help='lr of selectors/approximators')
    ap.add_argument('--backbone-lr', type=float, default=1e-5)
    ap.add_argument('--finetune-epochs', type=int, default=0, help='unfreeze the backbone for the last k epochs')
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--no-reuse', action='store_true')
    ap.add_argument('--approx', default='mlp', choices=['mlp', 'identity', 'dwconv', 'block'])
    ap.add_argument('--selector', default='mlp', choices=['mlp', 'dynamicvit'])
    ap.add_argument('--selector-depth', type=int, default=3)
    ap.add_argument('--fixed-keep', default=None, help='comma separated per-stage keep ratios')
    ap.add_argument('--init', default=None, help='start from the ToFe modules (and backbone) of another run')
    ap.add_argument('--loss', default='ce', choices=['ce', 'kd'])
    ap.add_argument('--lam-cls', type=float, default=1.0)
    ap.add_argument('--lam-apr', type=float, default=2.0)
    ap.add_argument('--lam-flops', type=float, default=50.0,
                    help='paper: 5 with an unspecified FLOPs unit; with GMACs, 5 overshoots the budget')
    ap.add_argument('--train-split', default='train256', help='train256 = shorter side 256 (resize_train.py)')
    ap.add_argument('--train-frac', type=float, default=1.0, help='fraction of the downloaded train images')
    ap.add_argument('--val-limit', type=int, default=None, help='debug only: val is sorted by class')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    out_dir = os.path.join(RUNS, args.name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'log.txt'), 'a')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    log(json.dumps(vars(args)))
    vit = load_backbone()
    teacher = copy.deepcopy(vit) if args.finetune_epochs else vit
    N = vit.patch_embed.num_patches
    fixed = [round(float(r) * N) for r in args.fixed_keep.split(',')] if args.fixed_keep else None
    model = ToFe(vit, reuse=not args.no_reuse, approx=args.approx, selector=args.selector,
                 selector_depth=args.selector_depth, fixed_keep=fixed).to(DEV)
    if args.init:
        state = torch.load(checkpoint_path(args.init), map_location=DEV)['state']
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith('vit.') for k in missing), (missing, unexpected)
        log(f'initialised from {args.init}')

    train_ds = data.ImageNetFiles(args.train_split, data.train_transform(vit))
    if args.train_frac < 1:
        keep = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(0))
        train_ds = Subset(train_ds, keep[:int(args.train_frac * len(train_ds))].tolist())
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=True)
    val_loader = data.CachedVal(256)
    log(f'train {len(train_ds)} images, val {len(val_loader.y)} images')

    modules = [p for n, p in model.named_parameters() if not n.startswith('vit.')]
    groups = [{'params': modules, 'lr': args.lr}]
    if args.finetune_epochs:
        groups.append({'params': list(vit.parameters()), 'lr': args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    total, warm = args.epochs * len(loader), min(200, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / total))))

    dense = model.dense_flops() / 1e9
    target = args.budget * dense
    log(f'dense {dense:.3f} GMACs, target {target:.3f} GMACs, extra params '
        f'{sum(p.numel() for p in modules) / 1e6:.3f}M')
    t_start = time.time()
    for ep in range(args.epochs):
        if args.finetune_epochs and ep == args.epochs - args.finetune_epochs:
            for p in vit.parameters():
                p.requires_grad_(True)
            log('backbone unfrozen')
        model.train()
        vit.eval()  # DeiT-S has no dropout; keep norm/attention behaviour identical
        stats = torch.zeros(4, device=DEV)
        kept = torch.zeros(len(model.locs), device=DEV)
        t, nb = time.time(), 0
        for img, y in loader:
            img, y = img.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(**AMP):
                t_states, t_logits = model.teacher_states(img, teacher)
                logits, masks, approx = model.forward_train(img)
                if args.loss == 'ce':
                    l_cls = F.cross_entropy(logits.float(), y)
                else:
                    l_cls = F.kl_div(logits.float().log_softmax(-1), t_logits.float().log_softmax(-1),
                                     reduction='batchmean', log_target=True)
                l_apr = logits.new_zeros((), dtype=torch.float)
                for m, xa, xt in zip(masks, approx, t_states):
                    frz = 1 - m
                    err = (xa.float() - xt.float()).pow(2).mean(-1)
                    l_apr = l_apr + ((frz * err).sum(1) / frz.sum(1).clamp_min(1)).mean()
                counts = [m.sum(1) for m in masks]
                gmacs = model.flops(counts) / 1e9
                if fixed is None:
                    l_flops = ((gmacs - target) ** 2).mean()
                else:  # Table VII: fixed per-stage keep ratios (DynamicViT-style ratio loss)
                    l_flops = sum(((m.sum(1) - k) / N).pow(2).mean() for m, k in zip(masks, fixed))
                loss = args.lam_cls * l_cls + args.lam_apr * l_apr + args.lam_flops * l_flops
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                stats += torch.stack([l_cls.detach(), l_apr.detach(), l_flops.detach(), gmacs.float().mean()])
                kept += torch.stack([c.mean() for c in counts]).detach()
            nb += 1
        s = (stats / nb).tolist()
        keep_avg = (kept / nb).tolist()
        log(f'ep{ep}: cls {s[0]:.3f} apr {s[1]:.4f} flops {s[2]:.4f} | train GMACs {s[3]:.3f} | '
            f'keep {[round(k, 1) for k in keep_avg]} | {time.time() - t:.0f}s, {nb * args.bs / (time.time() - t):.0f} img/s')
    train_time = time.time() - t_start

    model.eval()
    state = {k: v for k, v in model.state_dict().items() if args.finetune_epochs or not k.startswith('vit.')}
    torch.save({'args': vars(args), 'keep_avg': keep_avg, 'state': state}, os.path.join(out_dir, 'model.pt'))
    keep_n = fixed or [round(k) for k in keep_avg]  # batch-adaptive N^s = average kept during training
    top1, top5 = evaluate(lambda z: model.forward_infer(z, keep_counts=keep_n), val_loader, args.val_limit)
    res = {
        'name': args.name, 'keep_counts': keep_n, 'keep_ratios': [round(k / N, 3) for k in keep_n],
        'gmacs': model.flops(keep_n) / 1e9, 'dense_gmacs': dense, 'top1': top1, 'top5': top5,
        'throughput': throughput(lambda z: model.forward_infer(z, keep_counts=keep_n)),
        'extra_params_M': sum(p.numel() for p in modules) / 1e6, 'train_time_min': train_time / 60,
    }
    log(json.dumps(res))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
