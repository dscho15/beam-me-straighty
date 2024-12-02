import torch
import einops
from torch import nn

from transformers import Dinov2Model

from timm.models.vision_transformer import LayerScale, DropPath, Mlp
from timm.models.vision_transformer import Block as MHSABlock

from models.utils.rope2d import RoPE2D
from models.utils.rope1d import RoPE1D


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb

class SelfAttention1D(nn.Module):

    def __init__(self, dim, use_rope: bool = False, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., rope_freq: int = 100):
        
        super().__init__()
        
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if use_rope:
            self.rope = RoPE1D(rope_freq)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, xpos1d: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, N, C = x.shape

        qkv = einops.rearrange(self.qkv(x), "b n (qkv nh c) -> b nh qkv n c", qkv=3, nh=self.num_heads) # b n 3 nh c        
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

class MHSABlock(torch.nn.Module):

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
        super(MHSABlock, self).__init__()

        self.att = SelfAttention1D(dim, num_heads=num_heads)

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
        self, x: torch.FloatTensor, key_padding_mask: torch.FloatTensor
    ) -> torch.FloatTensor:

        z = self.norm1(x)

        x = x + self.drop_path1(
            self.ls1(
                self.att(
                    query=z,
                    key=z,
                    value=z,
                    is_causal=True,
                    attn_mask=key_padding_mask,
                )[0]
            )
        )

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
    
    
class CrossAttention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 

    def forward(self, x: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        print(self.qkv(x).shape)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
               
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
               
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

    def forward(self, x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:

        x = x + self.drop_path1(
            self.ls1(self.att(query=self.norm1(x), key=y, value=y)[0])
        )

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class DinoFeaturePyramid(torch.nn.Module):

    def __init__(self):

        super(DinoFeaturePyramid, self).__init__()

        self.model = Dinov2Model.from_pretrained(
            "facebook/dinov2-small",
        )

        self.patch_size = 14

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        patch_outputs = self.model(x, return_dict=False)[0][:, 1:]

        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        patch_outputs = einops.rearrange(
            patch_outputs, "b (h w) c -> b h w c", h=h, w=w
        )

        return patch_outputs


class ARStraighter(torch.nn.Module):

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        n_blocks: int = 8,
        n_tokens: int = 8,
        with_pos_embeddings: bool = True,
        discretized_space: int = 301,
    ):

        super(ARStraighter, self).__init__()

        self.backbone = DinoFeaturePyramid()
        self.proj_dino_features = torch.nn.Linear(384, dim)

        self.MHSA_blocks = torch.nn.ModuleList(
            [MHSABlock(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
        )
        self.MHCA_blocks = torch.nn.ModuleList(
            [MHCABlock(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
        )

        self.num_tokens = n_tokens
        self.tokens = torch.nn.Parameter(torch.randn(n_tokens, dim))
        torch.nn.init.normal_(self.tokens, std=0.02)

        self.output_tokens = torch.nn.Linear(dim, discretized_space)
        torch.nn.init.normal_(self.output_tokens.weight, std=0.02)

        self.embed_tokens = torch.nn.Sequential(
            torch.nn.Linear(1, dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim // 2, dim),
        )

        self.with_pos_embeddings = with_pos_embeddings

    def forward(
        self, x: tuple[torch.FloatTensor, torch.FloatTensor]
    ) -> torch.FloatTensor:

        (image, tokens) = x

        if len(image.shape) != 4:
            raise ValueError("Image shape must be (b, c, h, w)")

        # image dims
        b, c, h, w = image.shape

        # check num of tokens
        b, n, e = tokens.shape
        
        tokens_embedded = self.embed_tokens(tokens)
        
        token_mask = torch.triu(torch.ones(b, n, n), diagonal=1).to(image.device)

        # extract dino features
        with torch.no_grad():
            
            feats = self.backbone(image)

        feats = einops.rearrange(feats, "b h w c -> b (h w) c")
        
        feats = self.proj_dino_features(feats)

        for mhsa, mhca in zip(self.MHSA_blocks, self.MHCA_blocks):

            tokens = mhsa(tokens_embedded, token_mask)

            tokens = mhca(tokens, feats)

        tokens = self.output_tokens(tokens)

        return tokens


if __name__ == "__main__":

    self_attn = SelfAttention1D(384)
    
    x = torch.randn(16, 4, 384) # b
    self_attn(x, None, True)

    # arm = ARStraighter()

    # x = torch.randn(1, 3, 256, 256)

    # tokens = torch.randint(0, 301, (1, 8, 1)) * 1.0

    # out = arm((x, tokens))
    # print(out.shape)
