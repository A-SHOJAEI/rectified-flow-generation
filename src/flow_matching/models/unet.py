"""UNet architecture for flow matching velocity prediction.

Based on DDPM++ / NCSN++ architecture adapted for rectified flow.
Includes time conditioning via sinusoidal embeddings, residual blocks,
self-attention, and GroupNorm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # Each: [B, heads, head_dim, HW]
        q = q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)  # [B, C, HW]
        return x + self.proj(out).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """UNet velocity model for rectified flow on CIFAR-10 32x32.

    Architecture:
    - 4 resolution levels: 32 -> 16 -> 8 -> 4
    - Channel multipliers: [1, 2, 2, 2] * 128 = [128, 256, 256, 256]
    - Self-attention at 16x16 and 8x8
    - 4 residual blocks per level
    - ~62M parameters
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=4,
        attn_resolutions=(16, 8),
        dropout=0.15,
        num_heads=4,
    ):
        super().__init__()
        time_emb_dim = base_channels * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_channels]
        ch = base_channels
        res = 32

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                block = ResBlock(ch, out_ch, time_emb_dim, dropout)
                attn = SelfAttention(out_ch, num_heads) if res in attn_resolutions else None
                level_blocks.append(nn.ModuleDict({"res": block, "attn": nn.Identity() if attn is None else attn}))
                ch = out_ch
                channels.append(ch)
            self.down_blocks.append(level_blocks)

            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(ch))
                channels.append(ch)
                res //= 2
            else:
                self.down_samples.append(None)

        # Middle
        self.mid_res1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch, num_heads)
        self.mid_res2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Build decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            out_ch = base_channels * channel_mults[level]
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block = ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                attn = SelfAttention(out_ch, num_heads) if res in attn_resolutions else None
                level_blocks.append(nn.ModuleDict({"res": block, "attn": nn.Identity() if attn is None else attn}))
                ch = out_ch
            self.up_blocks.append(level_blocks)

            if level > 0:
                self.up_samples.append(Upsample(ch))
                res *= 2
            else:
                self.up_samples.append(None)

        # Output
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self._count = sum(p.numel() for p in self.parameters())

    @property
    def num_params(self):
        return self._count

    def forward(self, x, t):
        """Predict velocity v(x_t, t).

        Args:
            x: [B, C, H, W] interpolated sample x_t
            t: [B] timesteps in [0, 1]
        Returns:
            [B, C, H, W] predicted velocity
        """
        t_emb = timestep_embedding(t * 1000, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        h = self.input_conv(x)
        skips = [h]

        for level_blocks, ds in zip(self.down_blocks, self.down_samples):
            for block_dict in level_blocks:
                h = block_dict["res"](h, t_emb)
                h = block_dict["attn"](h)
                skips.append(h)
            if ds is not None:
                h = ds(h)
                skips.append(h)

        # Middle
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Decoder
        for level_blocks, us in zip(self.up_blocks, self.up_samples):
            for block_dict in level_blocks:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block_dict["res"](h, t_emb)
                h = block_dict["attn"](h)
            if us is not None:
                h = us(h)

        return self.out_conv(F.silu(self.out_norm(h)))
