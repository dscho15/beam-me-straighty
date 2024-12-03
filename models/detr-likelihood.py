import torch
import einops

from transformers import Dinov2Model

from timm.models.vision_transformer import LayerScale, DropPath, Mlp
from timm.models.vision_transformer import Block as MHSABlock
from models.utils.cross_attention import MHCABlock


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


class DETRStraighter(torch.nn.Module):

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        n_blocks: int = 8,
        n_tokens: int = 8,
        with_pos_embeddings: bool = True,
        discretized_space: int = 301,
    ):

        super(DETRStraighter, self).__init__()

        self.backbone = DinoFeaturePyramid()
        self.proj_dino_features = torch.nn.Linear(384, dim)

        self.MHSA_blocks = torch.nn.ModuleList(
            [MHSABlock(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
        )
        self.MHCA_blocks = torch.nn.ModuleList(
            [MHCABlock(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
        )

        self.tokens = torch.nn.Parameter(torch.randn(n_tokens, dim))
        torch.nn.init.normal_(self.tokens, std=0.02)

        self.output_tokens = torch.nn.Linear(dim, discretized_space)
        torch.nn.init.normal_(self.output_tokens.weight, std=0.02)

        self.with_pos_embeddings = with_pos_embeddings

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        features = self.backbone(x)

        features = einops.rearrange(features, "b h w c -> b (h w) c")

        features = self.proj_dino_features(features)

        tokens = self.tokens.unsqueeze(0).repeat(features.shape[0], 1, 1)

        if self.with_pos_embeddings:

            grid_h, grid_w = torch.meshgrid(
                torch.arange(2), torch.arange(2), indexing="ij"
            )
            
            grid = torch.stack([grid_h.ravel(), grid_w.ravel()], axis=0)  # Shape: (2, H*W)

            pos_embeds = get_2d_sincos_pos_embed_from_grid(tokens.shape[-1], grid)
            pos_embeds = pos_embeds.unsqueeze(1).repeat(1, 2, 1)
            pos_embeds = einops.rearrange(pos_embeds, "h w c -> (h w) c")
            pos_embeds = pos_embeds.to(tokens.device)

            tokens += pos_embeds.unsqueeze(0)

        for mhsa, mhca in zip(self.MHSA_blocks, self.MHCA_blocks):

            tokens = mhsa(tokens)

            tokens = mhca(tokens, features)

        tokens = self.output_tokens(tokens)

        return tokens


if __name__ == "__main__":

    arm = DETRStraighter()

    x = torch.randn(1, 3, 336, 336)

    out = arm(x)
    print(out.shape)

    print(count_parameters(arm))
