# Vendored from https://github.com/johnma2006/mamba-minimal (MIT-style, educational)
# Adapted: typing fix, torch.einsum instead of einops.einsum, no LM head.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int = 1
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _compose_ssm_pairs(ar: torch.Tensor, br: torch.Tensor, al: torch.Tensor, bl: torch.Tensor):
    """Compose two affine maps z -> ar*z + br after z -> al*z + bl. Result: z -> (ar*al)*z + (ar*bl + br)."""
    return ar * al, ar * bl + br


def inclusive_scan_affine_recurrence(
    a: torch.Tensor, b: torch.Tensor, dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel inclusive scan for h[t] = a[t] * h[t - 1] + b[t] (element-wise), h[-1] = 0.

    a, b same shape; associative composition on pairs (a, b) per time step.
    After the scan, b_out[t] equals h[t] when the recurrence is scalar along the scanned axis.

    Replaces an O(L) Python loop over sequence length (the main bottleneck in the minimal Mamba port).

    Updates only the ``shift:`` suffix each round and **does not** clone the full tensor per round
    (full clones caused multi GiB spikes and OOM when combined with large image backbones).
    """
    a_out = a.clone()
    b_out = b.clone()
    L = a_out.shape[dim]
    shift = 1
    while shift < L:
        idx_hi = [slice(None)] * a_out.ndim
        idx_hi[dim] = slice(shift, None)
        idx_lo = [slice(None)] * a_out.ndim
        idx_lo[dim] = slice(0, L - shift)
        tup_hi, tup_lo = tuple(idx_hi), tuple(idx_lo)
        ar, br = a_out[tup_hi], b_out[tup_hi]
        al, bl = a_out[tup_lo], b_out[tup_lo]
        ac, bc = _compose_ssm_pairs(ar, br, al, bl)
        # Must not use in-place slice writes here: autograd needs strided views of a_out/b_out
        # for backward through earlier steps; in-place updates corrupt version counters.
        pad = [slice(None)] * a_out.ndim
        pad[dim] = slice(0, shift)
        prefix_a = a_out[tuple(pad)]
        prefix_b = b_out[tuple(pad)]
        a_out = torch.cat([prefix_a, ac], dim=dim)
        b_out = torch.cat([prefix_b, bc], dim=dim)
        shift <<= 1
    return a_out, b_out


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), "n -> d n", d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(torch.einsum("b l d, d n -> b l d n", delta, A))
        deltaB_u = torch.einsum("b l d, b l n, b l d -> b l d n", delta, B, u)
        # Parallel prefix: state x[t] = deltaA[t] * x[t-1] + deltaB_u[t]; collect y[t] = <x[t], C[t]> + u*D
        _, prefix_x = inclusive_scan_affine_recurrence(deltaA, deltaB_u, dim=1)
        y = torch.einsum("b l d n, b l n -> b l d", prefix_x, C)
        return y + u * D


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaStack(nn.Module):
    """Stack of Mamba residual blocks for sequence encoding."""

    def __init__(self, d_model: int, n_layer: int = 2, d_state: int = 16, expand: int = 2):
        super().__init__()
        args = ModelArgs(d_model=d_model, n_layer=n_layer, d_state=d_state, expand=expand)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, N = 3, 127, 16, 8
    a = torch.sigmoid(torch.randn(B, L, D, N))
    b = torch.randn(B, L, D, N)
    x = torch.zeros(B, D, N)
    seq = []
    for i in range(L):
        x = a[:, i] * x + b[:, i]
        seq.append(x.clone())
    seq = torch.stack(seq, dim=1)
    _, par_b = inclusive_scan_affine_recurrence(a, b, dim=1)
    err = (seq - par_b).abs().max().item()
    print("max_abs_err", err)
    assert err < 1e-5, err
    print("ok")
