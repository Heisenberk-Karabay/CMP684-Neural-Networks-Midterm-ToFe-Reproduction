"""Training-free token reduction baselines running on the same frozen backbone:
EViT-style [CLS]-attention pruning (Liang et al.) and ToMe token merging (Bolya et al.)."""
import math

import torch
import torch.nn.functional as F

from tofe import block_flops, embed, gather_tokens


def _attention(blk, x, bias=None, need_cls=False):
    """MHSA residual branch with the fused SDPA kernel (same speed path as the dense model).
    Optionally also returns the head-averaged [CLS]->patch attention row."""
    B, T, C = x.shape
    a = blk.attn
    qkv = a.qkv(blk.norm1(x)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = a.q_norm(q), a.k_norm(k)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype))
    cls_attn = None
    if need_cls:
        cls_attn = ((q[:, :, :1] * a.scale) @ k.transpose(-2, -1)).softmax(-1)[:, :, 0, 1:].mean(1)
    out = a.proj(a.norm(out.transpose(1, 2).reshape(B, T, C)))
    return x + blk.ls1(out), cls_attn, k


# ----------------------------------------------------------------------- EViT
def evit_block(blk, x, keep, fuse=True):
    """After MHSA keep the `keep` patch tokens with the highest [CLS] attention
    (head-averaged), optionally fuse the rest into one token, then MLP."""
    x, cls_attn, _ = _attention(blk, x, need_cls=True)
    idx = cls_attn.topk(keep, dim=1).indices
    patches = x[:, 1:]
    parts = [x[:, :1], gather_tokens(patches, idx)]
    if fuse:
        w = cls_attn.scatter(1, idx, 0.0).unsqueeze(-1)
        parts.append((w * patches).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1e-6))
    x = torch.cat(parts, 1)
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


@torch.no_grad()
def evit_forward(vit, img, locs, keep_counts, fuse=True):
    x = embed(vit, img)
    for i, blk in enumerate(vit.blocks):
        x = evit_block(blk, x, keep_counts[locs.index(i)], fuse) if i in locs else blk(x)
    return vit.forward_head(vit.norm(x))


def _static(vit):
    pe = vit.patch_embed.proj
    return vit.patch_embed.num_patches * pe.in_channels * pe.kernel_size[0] * pe.kernel_size[1] * vit.embed_dim \
        + vit.embed_dim * vit.head.out_features


def evit_flops(vit, locs, keep_counts, fuse=True):
    D, Dh = vit.embed_dim, vit.blocks[0].mlp.fc1.out_features
    n = vit.patch_embed.num_patches + 1
    total = _static(vit)
    for i in range(len(vit.blocks)):
        if i in locs:
            n_new = 1 + keep_counts[locs.index(i)] + int(fuse)
            total += 4 * n * D * D + 2 * n * n * D + 2 * n_new * D * Dh
            n = n_new
        else:
            total += block_flops(n, D, Dh)
    return total


# ----------------------------------------------------------------------- ToMe
def bipartite_soft_matching(metric, r):
    """ToMe: split tokens into alternating sets A/B, merge the r most similar A tokens
    into their best B match. Token 0 ([CLS]) is protected."""
    t = metric.size(1)
    r = min(r, (t - 1) // 2)
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[:, ::2], metric[:, 1::2]
    scores = a @ b.transpose(-1, -2)
    scores[:, 0, :] = -math.inf
    node_max, node_idx = scores.max(-1)
    edge_idx = node_max.argsort(-1, descending=True)[..., None]
    unm_idx = edge_idx[:, r:].sort(1).values
    src_idx = edge_idx[:, :r]
    dst_idx = node_idx[..., None].gather(1, src_idx)

    def merge(x):
        src, dst = x[:, ::2], x[:, 1::2]
        n, t1, c = src.shape
        unm = src.gather(1, unm_idx.expand(n, t1 - r, c))
        src = src.gather(1, src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(1, dst_idx.expand(n, r, c), src, reduce='sum')
        return torch.cat([unm, dst], 1)

    return merge


def tome_block(blk, x, size, r):
    x, _, k = _attention(blk, x, bias=size.log()[:, None, None, :, 0])  # proportional attention
    if r > 0:
        merge = bipartite_soft_matching(k.mean(1), r)
        x, size = merge(x * size), merge(size)
        x = x / size
    return x + blk.ls2(blk.mlp(blk.norm2(x))), size


@torch.no_grad()
def tome_forward(vit, img, r):
    x = embed(vit, img)
    size = x.new_ones(x.size(0), x.size(1), 1)
    for blk in vit.blocks:
        x, size = tome_block(blk, x, size, r)
    return vit.forward_head(vit.norm(x))


def tome_flops(vit, r):
    D, Dh = vit.embed_dim, vit.blocks[0].mlp.fc1.out_features
    n = vit.patch_embed.num_patches + 1
    total = _static(vit)
    for _ in vit.blocks:
        n_new = n - min(r, (n - 1) // 2)
        total += 4 * n * D * D + 2 * n * n * D + 2 * n_new * D * Dh
        n = n_new
    return total
