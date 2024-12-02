import torch
import einops
from torch import nn

from transformers import Dinov2Model

from timm.models.vision_transformer import LayerScale, DropPath, Mlp

from models.utils.rope2d import RoPE2D
from models.utils.rope1d import RoPE1D

from models.utils.cross_attention import MHCABlock
from models.utils.self_attention import MHSABlock1D


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
        dim: int = 384,
        abs_pos_emb: bool = False,
        num_heads: int = 4,
        n_blocks: int = 8,
        n_tokens: int = 8,
        with_pos_embeddings: bool = True,
        discretized_space: int = 301,
    ):

        super(ARStraighter, self).__init__()

        self.backbone = DinoFeaturePyramid()
        self.proj_dino_features = torch.nn.Linear(384, dim)
        self.abs_pos_emb = abs_pos_emb

        self.MHSA_blocks = torch.nn.ModuleList(
            [MHSABlock1D(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
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
        self.pos_emb_mlp = nn.Linear(dim, dim)

    def forward(
        self, x: tuple[torch.FloatTensor, torch.FloatTensor]
    ) -> torch.FloatTensor:

        (image, tokens) = x

        if len(image.shape) != 4:
            raise ValueError("Image shape must be (b, c, h, w)")

        # image dims
        b, c, h, w = image.shape

        # check num of tokens
        b, n, _ = tokens.shape

        # embed tokens
        tokens = self.embed_tokens(tokens)

        if self.abs_pos_emb:
            tokens += get_1d_sincos_pos_embed_from_grid(
                tokens.size(-1), torch.arange(n)
            )
            tokens = self.pos_emb_mlp(tokens)

        # extract dino features
        feats = self.backbone(image)

        feats = einops.rearrange(feats, "b h w c -> b (h w) c")

        feats = self.proj_dino_features(feats)

        for mhsa, mhca in zip(self.MHSA_blocks, self.MHCA_blocks):

            tokens = mhsa(tokens)

            tokens = mhca(tokens, feats)

        tokens = self.output_tokens(tokens)

        return tokens


if __name__ == "__main__":

    # self_attn = SelfAttention1D(384)

    # x = torch.randn(16, 4, 384) # b
    # self_attn(x, None, True)

    arm = ARStraighter()

    x = torch.randn(1, 3, 256, 256)

    tokens = torch.randint(0, 302, (1, 8, 1)) * 1.0

    out = arm((x, tokens))
    print(out.shape)
