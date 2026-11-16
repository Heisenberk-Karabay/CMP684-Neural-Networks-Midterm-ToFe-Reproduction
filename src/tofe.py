"""ToFe: Lagged Token Freezing and Reusing, implemented from the paper text
(Zhang et al., IEEE TNNLS 2026) on top of a timm ViT/DeiT backbone.

Two forward paths:
  * forward_train  - differentiable masked forward used for training (Eq. 16-18).
                     All tokens stay in the tensor; frozen tokens are hidden from
                     attention through the mask G and replaced by X + Approx(X).
  * forward_infer  - real token freezing for inference (Algorithm 1). Kept tokens
                     are gathered and run through the blocks, frozen tokens are
                     approximated and scattered back (Rearrange, Eq. 7).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


def block_flops(n, d, d_hidden):
    """Eq. (4): cost of one ViT block with n tokens (multiply-accumulates)."""
    return 4 * n * d * d + 2 * n * n * d + 2 * n * d * d_hidden


def gather_tokens(x, idx):
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def linear_macs(module):
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


# ------------------------------------------------------------------ token selectors
class TokenSelector(nn.Module):
    """3-layer MLP: input projection, one hidden layer with ReLU, output projection.
    `depth` counts linear layers (Table IV: 3, 5, 7). Outputs keep/freeze logits (index 0 = keep)."""

    def __init__(self, dim, hidden=None, depth=3, init_keep_logit=2.0):
        super().__init__()
        hidden = hidden or dim // 4
        layers = [nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
        # start close to "keep everything" so training begins from the dense model
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([init_keep_logit, 0.0]))

    def forward(self, x):
        return self.net(x)

    def flops(self, n):
        return linear_macs(self) * n


class DynamicViTSelector(nn.Module):
    """Ablation (Table IV): the prediction module of DynamicViT (local + global features)."""

    def __init__(self, dim, init_keep_logit=2.0):
        super().__init__()
        self.in_conv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.out_conv = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim // 4),
                                      nn.GELU(), nn.Linear(dim // 4, 2))
        with torch.no_grad():
            self.out_conv[-1].bias.copy_(torch.tensor([init_keep_logit, 0.0]))

    def forward(self, x):
        x = self.in_conv(x)
        C = x.size(-1)
        local, glob = x[..., :C // 2], x[..., C // 2:].mean(1, keepdim=True)
        return self.out_conv(torch.cat([local, glob.expand_as(local)], -1))

    def flops(self, n):
        return linear_macs(self) * n


# -------------------------------------------------------------- token approximators
class TokenApproximator(nn.Module):
    """Bottleneck 2-layer MLP predicting the change of frozen tokens: X + Approx(X).
    hidden = D/2 reproduces the paper's "2.52% of GFLOPs" (Table VI) for DeiT-S.
    The output layer is zero-initialised, so it starts as plain token reuse."""
    needs_grid = False

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim // 2
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, frozen=None):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))

    def flops(self, n):
        return 2 * self.fc1.in_features * self.fc1.out_features * n


class IdentityApproximator(nn.Module):
    """Ablation (Table VI): reuse frozen tokens as they are."""
    needs_grid = False

    def forward(self, x, frozen=None):
        return x

    def flops(self, n):
        return 0


class DWConvApproximator(nn.Module):
    """Ablation (Table VI): 3x3 depth-wise convolution on the patch grid (local mixing)."""
    needs_grid = True

    def __init__(self, dim, grid):
        super().__init__()
        self.grid = grid
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, frozen=None):
        B, N, D = x.shape
        h = self.norm(x).transpose(1, 2).reshape(B, D, self.grid, self.grid)
        return x + self.conv(h).flatten(2).transpose(1, 2)

    def flops(self, n):
        return 9 * self.conv.out_channels * n


class BlockApproximator(nn.Module):
    """Ablation (Table VI): a full transformer block over the frozen tokens only."""
    needs_grid = False

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.block = Block(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, init_values=1e-5)
        self.dim, self.d_hidden = dim, int(dim * mlp_ratio)

    def forward(self, x, frozen=None):
        # training form: frozen tokens only attend to each other, as at inference
        return self.block(x) if frozen is None else masked_block(self.block, x, frozen)

    def flops(self, n):
        return block_flops(n, self.dim, self.d_hidden)


def masked_block(blk, x, policy, eps=1e-6):
    """timm Block forward with the attention mask of Eq. (17)-(18):
    G_ij = 1 if i == j else policy_j, softmax re-normalised over kept keys."""
    B, T, C = x.shape
    a = blk.attn
    qkv = a.qkv(blk.norm1(x)).reshape(B, T, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = a.q_norm(q), a.k_norm(k)
    attn = ((q * a.scale) @ k.transpose(-2, -1)).float()
    eye = torch.eye(T, device=x.device, dtype=attn.dtype)
    G = policy.float()[:, None, None, :] * (1 - eye) + eye
    attn = (attn - attn.amax(-1, keepdim=True)).exp() * G
    attn = (attn + eps / T) / (attn.sum(-1, keepdim=True) + eps)
    out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, C)
    out = a.proj_drop(a.proj(a.norm(out)))
    x = x + blk.drop_path1(blk.ls1(out))
    x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    return x


def embed(vit, img):
    return vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(img))))


class ToFe(nn.Module):
    def __init__(self, vit, locs=(3, 6, 9), reuse=True, approx='mlp', selector='mlp', selector_depth=3,
                 fixed_keep=None):
        """locs: 0-indexed block where stage s starts (DeiT: before the 4th/7th/10th block).
        fixed_keep: optional per-stage kept-token counts (Table VII): trained with a per-stage
        keep-ratio loss instead of the FLOPs loss and used as N^s at inference."""
        super().__init__()
        self.vit = vit
        self.locs = list(locs)
        self.reuse = reuse
        self.fixed_keep = fixed_keep
        self.depth = len(vit.blocks)
        self.dim = D = vit.embed_dim
        self.d_hidden = vit.blocks[0].mlp.fc1.out_features
        self.num_patches = vit.patch_embed.num_patches
        grid = vit.patch_embed.grid_size[0]
        heads = vit.blocks[0].attn.num_heads
        make_sel = {'mlp': lambda: TokenSelector(D, depth=selector_depth),
                    'dynamicvit': lambda: DynamicViTSelector(D)}[selector]
        make_apx = {'mlp': lambda: TokenApproximator(D), 'identity': IdentityApproximator,
                    'dwconv': lambda: DWConvApproximator(D, grid), 'block': lambda: BlockApproximator(D, heads)}[approx]
        self.selectors = nn.ModuleList(make_sel() for _ in self.locs)
        self.approximators = nn.ModuleList(make_apx() for _ in self.locs[:-1])

    @property
    def stages(self):
        bounds = self.locs + [self.depth]
        return [(bounds[i], bounds[i + 1]) for i in range(len(self.locs))]

    def head(self, x, vit=None):
        vit = vit or self.vit
        return vit.forward_head(vit.norm(x))

    # ------------------------------------------------------------------ FLOPs
    def _static_flops(self):
        pe = self.vit.patch_embed.proj
        return (self.num_patches * pe.in_channels * pe.kernel_size[0] * pe.kernel_size[1] * self.dim
                + self.dim * self.vit.head.out_features)

    def flops(self, keep_counts):
        """Analytical cost (MACs, Eq. 4 + module overheads) for per-stage kept patch-token counts.
        keep_counts may be python numbers or tensors (differentiable, for L_FLOPs)."""
        N, D = self.num_patches, self.dim
        total = self._static_flops() + self.locs[0] * block_flops(N + 1, D, self.d_hidden)
        for s, (lo, hi) in enumerate(self.stages):
            n = keep_counts[s]
            total = total + (hi - lo) * block_flops(n + 1, D, self.d_hidden) + self.selectors[s].flops(N)
            if s < len(self.approximators):
                ap = self.approximators[s]
                total = total + ap.flops(N if ap.needs_grid else N - n)
        return total

    def dense_flops(self):
        return self._static_flops() + self.depth * block_flops(self.num_patches + 1, self.dim, self.d_hidden)

    def _logits(self, s, patches):
        """Selector logits in fp32: under bf16 autocast, saturated keep probabilities
        produce many exact ties and top-k then picks tokens arbitrarily."""
        with torch.autocast('cuda', enabled=False):
            return self.selectors[s](patches.float())

    # --------------------------------------------------------------- training
    def _mask(self, logits, hard_eval):
        if hard_eval:
            return (logits[..., 0] > logits[..., 1]).float()
        return F.gumbel_softmax(logits, tau=1.0, hard=True)[..., 0]

    def forward_train(self, img, hard_eval=False):
        """Masked forward, Eq. (16). Returns logits, keep masks M^s [B,N] and the
        approximated patch tokens at the end of every stage s < S (for L_apr)."""
        x = embed(self.vit, img)
        for blk in self.vit.blocks[:self.locs[0]]:
            x = blk(x)
        cls_keep = x.new_ones(x.size(0), 1, dtype=torch.float)
        prev = None
        masks, approx_out = [], []
        for s, (lo, hi) in enumerate(self.stages):
            m = self._mask(self._logits(s, x[:, 1:]), hard_eval)
            if not self.reuse and prev is not None:
                m = m * prev  # w/o reusing: once frozen, frozen forever
            prev = m
            policy = torch.cat([cls_keep, m], 1)
            x_in = x
            for blk in self.vit.blocks[lo:hi]:
                x = masked_block(blk, x, policy)
            if s < len(self.approximators):
                xa = self.approximators[s](x_in[:, 1:], frozen=1 - m)
                p = m.unsqueeze(-1).to(x.dtype)
                x = torch.cat([x[:, :1], p * x[:, 1:] + (1 - p) * xa], 1)
                approx_out.append(xa)
            masks.append(m)
        return self.head(x), masks, approx_out

    @torch.no_grad()
    def teacher_states(self, img, vit=None):
        """Dense backbone (`vit`, default: own backbone) hidden states at the end of
        every stage s < S, and its logits."""
        vit = vit or self.vit
        x = embed(vit, img)
        ends = {hi - 1 for lo, hi in self.stages[:-1]}
        out = []
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in ends:
                out.append(x[:, 1:])
        return out, self.head(x, vit)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def forward_infer(self, img, keep_counts=None, return_info=False):
        """Algorithm 1. keep_counts=None: instance-adaptive (B=1, keep tokens whose keep
        probability > 0.5). Otherwise batch-adaptive: keep the top-N^s tokens per stage.
        return_info adds the kept patch indices of every stage."""
        x = embed(self.vit, img)
        for blk in self.vit.blocks[:self.locs[0]]:
            x = blk(x)
        B, T, D = x.shape
        alive = None
        info = []
        for s, (lo, hi) in enumerate(self.stages):
            cls, patches = x[:, :1], x[:, 1:]
            logits = self._logits(s, patches)
            score = logits[..., 0] - logits[..., 1]  # same order as the keep probability, no ties
            if alive is not None:
                score = score.masked_fill(~alive, float('-inf'))
            if keep_counts is None:
                assert B == 1, 'instance-adaptive inference needs batch size 1'
                k = int((score > 0).sum())
            else:
                k = int(keep_counts[s])
            order = score.argsort(dim=1, descending=True)
            keep_idx, frz_idx = order[:, :k], order[:, k:]
            if not self.reuse:
                alive = torch.zeros_like(score, dtype=torch.bool).scatter_(1, keep_idx, True)
            if return_info:
                info.append(keep_idx)
            xk = torch.cat([cls, gather_tokens(patches, keep_idx)], 1)
            for blk in self.vit.blocks[lo:hi]:
                xk = blk(xk)
            if s < len(self.approximators):
                ap = self.approximators[s]
                if ap.needs_grid:
                    xf = gather_tokens(ap(patches), frz_idx)
                else:
                    xf = ap(gather_tokens(patches, frz_idx))
                new = torch.empty_like(patches)
                new.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, D), xk[:, 1:])
                new.scatter_(1, frz_idx.unsqueeze(-1).expand(-1, -1, D), xf)
                x = torch.cat([xk[:, :1], new], 1)
            else:
                x = xk  # last stage: frozen tokens can no longer reach the [CLS] token
        out = self.head(x)
        return (out, info) if return_info else out
