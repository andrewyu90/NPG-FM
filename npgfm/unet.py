"""Class-conditional latent U-Net (velocity field v_theta of NPG-FM).

A compact re-implementation of the LDM `openaimodel.UNetModel` configuration used by NPG-FM
(spatial-transformer attention at every level, class tokens as cross-attention context).
Module names match the original so released checkpoints load with `strict=True`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def norm(ch):
    return GroupNorm32(32, ch)


def zero(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, ch, emb_ch, dropout, out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.in_layers = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch))
        self.out_layers = nn.Sequential(norm(out_ch), nn.SiLU(), nn.Dropout(dropout),
                                        zero(nn.Conv2d(out_ch, out_ch, 3, padding=1)))
        self.skip_connection = nn.Identity() if out_ch == ch else nn.Conv2d(ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.in_layers(x) + self.emb_layers(emb).type(x.dtype)[..., None, None]
        return self.skip_connection(x) + self.out_layers(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, context_dim=None, dropout=0.):
        super().__init__()
        inner, context_dim = heads * dim_head, context_dim or dim
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        context = x if context is None else context
        split = lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2)  # b n (h d) -> b h n d
        out = F.scaled_dot_product_attention(split(self.to_q(x)), split(self.to_k(context)), split(self.to_v(context)))
        return self.to_out(out.transpose(1, 2).flatten(2))


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, context_dim, dropout=0.):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads, dim_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(dim, heads, dim_head, context_dim=context_dim, dropout=dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def _forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        return self.ff(self.norm3(x)) + x

    def forward(self, x, context=None):
        if self.training and torch.is_grad_enabled():  # gradient checkpointing saves memory during training
            return checkpoint(self._forward, x, context, use_reentrant=False)
        return self._forward(x, context)


class SpatialTransformer(nn.Module):
    def __init__(self, ch, heads, dim_head, context_dim):
        super().__init__()
        inner = heads * dim_head
        self.norm = nn.GroupNorm(32, ch, eps=1e-6)
        self.proj_in = nn.Conv2d(ch, inner, 1)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner, heads, dim_head, context_dim)])
        self.proj_out = zero(nn.Conv2d(inner, ch, 1))

    def forward(self, x, context):
        b, c, h, w = x.shape
        y = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)
        for blk in self.transformer_blocks:
            y = blk(y, context)
        return x + self.proj_out(y.transpose(1, 2).reshape(b, -1, h, w))


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, dim)

    def forward(self, c):
        return self.embedding(c)


class UNet(nn.Module):
    def __init__(self, in_channels=4, model_channels=256, num_heads=8, channel_mult=(1, 2, 4),
                 num_res_blocks=2, context_dim=512, num_classes=15, dropout=0.4):
        super().__init__()
        mc, emb_ch = model_channels, model_channels * 4
        self.model_channels = mc
        head_dim = model_channels // num_heads  # fixed head width: deeper levels get more heads
        attn = lambda ch: SpatialTransformer(ch, ch // head_dim, head_dim, context_dim)

        self.cond_stage_model = ClassEmbedder(context_dim, num_classes)
        self.time_embed = nn.Sequential(nn.Linear(mc, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, mc, 3, padding=1))])
        chans, ch = [mc], mc
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, emb_ch, dropout, mult * mc), attn(mult * mc)))
                ch = mult * mc
                chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                chans.append(ch)

        self.middle_block = TimestepEmbedSequential(ResBlock(ch, emb_ch, dropout), attn(ch), ResBlock(ch, emb_ch, dropout))

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + chans.pop(), emb_ch, dropout, mc * mult), attn(mc * mult)]
                ch = mc * mult
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(norm(ch), nn.SiLU(), zero(nn.Conv2d(mc, in_channels, 3, padding=1)))

    def forward(self, x, t, cls):
        """x: (B,4,h,w) latent, t: (B,) integer step, cls: (B,) class index."""
        context = self.cond_stage_model(cls[:, None])
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        hs, h = [], x
        for m in self.input_blocks:
            h = m(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for m in self.output_blocks:
            h = m(torch.cat([h, hs.pop()], dim=1), emb, context)
        return self.out(h)


UNET_CONFIGS = {
    'S': dict(model_channels=128, num_heads=4, context_dim=256),
    'M': dict(model_channels=192, num_heads=6, context_dim=384),
    'L': dict(model_channels=256, num_heads=8, context_dim=512),
    'XL': dict(model_channels=320, num_heads=12, context_dim=640),
}
