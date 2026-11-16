"""Shared helpers: backbone loading, accuracy evaluation, throughput measurement."""
import os
import time

import timm
import torch

DEV = 'cuda'
AMP = dict(device_type='cuda', dtype=torch.bfloat16)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
BACKBONE = 'deit_small_patch16_224.fb_in1k'
GITHUB_REPO = 'https://github.com/Heisenberk-Karabay/CMP684-Neural-Networks-Midterm-ToFe-Reproduction'
PROJECT_URL = GITHUB_REPO  # the project sits at the root of its own repository


def checkpoint_path(name):
    """runs/<name>/model.pt if the run was trained here, else models/<name>.pt (as in the repository)."""
    path = os.path.join(RUNS, name, 'model.pt')
    return path if os.path.exists(path) else os.path.join(ROOT, 'models', f'{name}.pt')


def load_backbone(name=BACKBONE):
    vit = timm.create_model(name, pretrained=True).to(DEV).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


@torch.no_grad()
def evaluate(fn, loader, limit=None):
    """Top-1 / top-5 accuracy (%) of fn(images) -> logits."""
    c1 = c5 = n = 0
    for img, y in loader:
        img, y = img.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        with torch.autocast(**AMP):
            out = fn(img)
        top5 = out.float().topk(5, -1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += y.numel()
        if limit and n >= limit:
            break
    return 100 * c1 / n, 100 * c5 / n


@torch.no_grad()
def throughput(fn, bs=128, iters=50, warmup=10, res=224):
    """Images/s for fn on random input, batch size bs, bf16 autocast (paper: bs=128)."""
    x = torch.randn(bs, 3, res, res, device=DEV)
    with torch.autocast(**AMP):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()
    return bs * iters / (time.perf_counter() - t)
