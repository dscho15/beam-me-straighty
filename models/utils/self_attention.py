import torch
import einops

from models.utils.rope1d import RoPE1D
from torch import nn
from timm.models.vision_transformer import LayerScale, DropPath, Mlp


class SelfAttention1D(nn.Module):

    def __init__(
        self,
        dim,
        use_rope: bool = False,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_freq: int = 100,
    ):

        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        if use_rope:
            self.rope = RoPE1D(rope_freq)
        else:
            self.rope = None

    def forward(
        self, x: torch.Tensor, xpos1d: torch.Tensor, is_causal: bool = True
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = einops.rearrange(
            self.qkv(x), "b n (qkv nh c) -> b nh qkv n c", qkv=3, nh=self.num_heads
        )  # b n 3 nh c

        q, k, v = torch.chunk(qkv, 3, dim=-1)

        if self.rope is not None:

            q = self.rope(q, xpos1d)

            k = self.rope(k, xpos1d)

        attn_bias = torch.zeros(N, N, dtype=torch.float).to(x.device)

        if is_causal:

            temp_mask = torch.ones(N, N, dtype=torch.bool).tril(diagonal=0)

            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

            attn_bias.to(q.dtype)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn += attn_bias

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x


class MHSABlock1D(torch.nn.Module):

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 2,
        init_values: float = 1e-3,
        drop_path: float = 0.1,
        mlp_ratio: int = 2,
        act_layer: torch.nn.Module = torch.nn.GELU,
        proj_drop: float = 0.0,
        use_rope: bool = False,
    ):
        super(MHSABlock1D, self).__init__()

        self.att = SelfAttention1D(dim, num_heads=num_heads, use_rope=use_rope)

        self.norm_cross_domain = torch.nn.LayerNorm(dim)

        self.norm1 = torch.nn.LayerNorm(dim)

        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else torch.nn.Identity()
        )

        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.norm2 = torch.nn.LayerNorm(dim)

        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else torch.nn.Identity()
        )

        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )

    def forward(
        self, x: torch.FloatTensor, is_causal: bool = True
    ) -> torch.FloatTensor:

        z = self.norm1(x)

        xpos1d = torch.arange(z.size(1)).unsqueeze(0).repeat(z.size(0), 1).to(z.device)

        x = x + self.drop_path1(
            self.ls1(
                self.att(
                    z,
                    xpos1d,
                    is_causal=is_causal,
                )
            )
        )

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
