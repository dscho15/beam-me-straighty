import torch
from torch import nn
import einops

from timm.models.vision_transformer import LayerScale, DropPath, Mlp


class CrossAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim**-0.5

        self.to_key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.to_query = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = einops.rearrange(
            self.to_query(x), "b n (1 nh c) -> b nh 1 n c", nh=self.num_heads
        )

        kv = einops.rearrange(
            self.to_key_value(y),
            "b n (kv nh c) -> b nh kv n c",
            kv=2,
            nh=self.num_heads,
        )

        k, v = torch.chunk(kv, 2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x


class MHCABlock(torch.nn.Module):

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 2,
        init_values: float = 1e-3,
        drop_path: float = 0.1,
        mlp_ratio: int = 2,
        act_layer: torch.nn.Module = torch.nn.GELU,
        proj_drop: float = 0.0,
    ):
        super(MHCABlock, self).__init__()

        self.att = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm1 = torch.nn.LayerNorm(dim)
        
        self.norm_cross_domain = torch.nn.LayerNorm(dim)

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

    def forward(self, x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:

        z = self.norm1(x)
        
        y = self.norm_cross_domain(y)

        x = x + self.drop_path1(self.ls1(self.att(query=z, key=y, value=y)[0]))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
