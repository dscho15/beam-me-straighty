import torch
import einops
import kornia

from transformers import Dinov2Model

from models.utils.cross_attention import MHCABlock
from timm.models.vision_transformer import Block as MHSABlock, Mlp


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, dtype=torch.float, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        
    return pe.type(dtype)


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


class PosUpdateBlock(torch.nn.Module):

    def __init__(self, dim: int, start: float, end: float, n_bins: int):
        super(PosUpdateBlock, self).__init__()

        self.weighting_function = torch.linspace(start, end, n_bins)
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim, n_bins),
            torch.nn.Softmax(dim=-1),
        )

    def forward(
        self, x: torch.FloatTensor, return_distribution: bool = False
    ) -> torch.FloatTensor:

        weighting_function = einops.rearrange(
            self.weighting_function.to(x.device), "d -> 1 1 d"
        )

        x = self.model(x)
        z = torch.sum(weighting_function * x, dim=-1, keepdim=True)

        if return_distribution:
            return z, x
        else:
            return z

class PosUpdateBlockSimple(torch.nn.Module):

    def __init__(self, dim: int, start: float, end: float):
        super().__init__()

        assert start == -end
        self.scale = end
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim, 1),
            torch.nn.Tanh(),
        )

    def forward(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:

        x = self.model(x)
        return x * self.scale
        
def generate_grid(
    x_steps: int = 3, y_steps: int = 3, x_range: tuple = (0, 1), y_range: tuple = (0, 1)
):
    """
    Generates a grid of evenly spaced points.

    Parameters:
        x_steps (int): Number of points along the x-axis.
        y_steps (int): Number of points along the y-axis.
        x_range (tuple): Range for x-axis values (min, max).
        y_range (tuple): Range for y-axis values (min, max).

    Returns:
        list of tuple: List of (x, y) coordinates.
    """
    x_values = torch.linspace(x_range[0], x_range[1], x_steps)
    y_values = torch.linspace(y_range[0], y_range[1], y_steps)

    (gy, gx) = torch.meshgrid(
            y_values,
            x_values,
            indexing="ij",
    )
        
    grid = torch.stack((gx, gy), dim=-1).reshape(-1, 2)

    return grid


class DETRStraighter(torch.nn.Module):

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        n_blocks: int = 8,
        n_tokens_row_cols: int = 2,
        with_pos_embeddings: bool = False,
        n_bins: int = 301,
        pos_ranges: list = [(-1, 1) for _ in range(8)],
    ):
        super(DETRStraighter, self).__init__()
        if len(pos_ranges) == 1:
            r = pos_ranges[0]
            pos_ranges_ = []
            for i in range(n_blocks):
                pos_ranges_.append((r[0] * (0.5)**i, r[1] * (0.5)**i))
            pos_ranges = pos_ranges_

        assert n_blocks == len(pos_ranges)
        
        self.backbone = DinoFeaturePyramid()
        
        self.proj_dino_features = torch.nn.Linear(384, dim)
        
        self.n_blocks = n_blocks

        self.MHSABlocks = torch.nn.ModuleList(
            [MHSABlock(dim=dim, num_heads=num_heads, drop_path=0.1) for _ in range(n_blocks)]
        )

        self.MHCABlocks = torch.nn.ModuleList(
            [MHCABlock(dim=dim, num_heads=num_heads, drop_path=0.1) for _ in range(n_blocks)]
        )

        self.tokens = torch.nn.Parameter(torch.randn(n_tokens_row_cols**2 * 2, dim))

        self.with_pos_embeddings = with_pos_embeddings

        self.PosUpdateBlocks = torch.nn.ModuleList(
            [PosUpdateBlockSimple(dim, start, end) for (start, end) in pos_ranges]
        )

        self.fixed_positions = torch.tensor(
            generate_grid(n_tokens_row_cols, n_tokens_row_cols, (-1, 1), (-1, 1)) # parameter to specify
        )

        self.proj = Mlp(in_features=dim, hidden_features=dim, out_features=dim, drop=0.1, norm_layer=torch.nn.LayerNorm)

    def forward_backbone(self, x: torch.FloatTensor) -> torch.FloatTensor:
        features = self.backbone(x)
        
        _, dh, dw, _ = features.shape
        
        features = einops.rearrange(features, "b h w c -> b (h w) c")
        
        features += posemb_sincos_2d(dh, dw, features.size(-1)).to(features.device)
        
        features = self.proj_dino_features(features)
        
        return features, dh, dw
    
    def warp_grid(self, points1: torch.FloatTensor, points2: torch.FloatTensor, dh: int, dw: int) -> torch.FloatTensor:
        device = points1.device

        H = kornia.geometry.homography.find_homography_dlt(
                points1, points2, solver="svd"
            ).unsqueeze(1)

        (gy, gx) = torch.meshgrid(
            torch.linspace(-1, 1, dh, device=device),
            torch.linspace(-1, 1, dw, device=device),
            indexing="ij",
        )
        
        grid = torch.stack((gx, gy), dim=-1)[None]

        warped_grid = kornia.geometry.transform.warp_grid(grid, H).clamp(-1, 1)
        
        return warped_grid

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bs = x.size(0)
        n_tokens = self.tokens.size(0)

        features, dh, dw = self.forward_backbone(x)

        tokens = self.tokens.unsqueeze(0).repeat(bs, 1, 1)
        
        if self.with_pos_embeddings:
        
            grid = torch.arange(tokens.size(1), device=features.device).unsqueeze(0)
        
            pos_embeds = get_1d_sincos_pos_embed_from_grid(
                tokens.size(-1), grid
            ).unsqueeze(1)
        
            pos_embeds = einops.rearrange(pos_embeds, "h w c -> (h w) c")
            pos_embeds = pos_embeds.to(tokens.device)
        
            tokens += pos_embeds.unsqueeze(0)
            
        delta_offsets = torch.zeros(
            bs, self.n_blocks, n_tokens, 1
        ).to(tokens.device)

        dino_features = features

        features = einops.rearrange(features, "b (h w) c -> b c h w", h=dh, w=dw)

        for i, (mhca, mhsa, pib) in enumerate(
            zip(self.MHCABlocks, self.MHSABlocks, self.PosUpdateBlocks)
        ):

            tokens = mhca(tokens, dino_features)

            tokens = mhsa(tokens)
            
            d_o = pib(tokens) 

            delta_offsets[:, i, ...] = (d_o if i == 0 else d_o + delta_offsets[:, i - 1, ...])

            points1 = self.fixed_positions.unsqueeze(0).repeat(bs, 1, 1).to(tokens.device)

            points2 = points1 + einops.rearrange(
                delta_offsets[:, i, ...],
                "b (n xy) 1 -> b n xy",
                n=points1.size(1),
                xy=2,
            )

            warped_grid = self.warp_grid(points2, points1, dh, dw)
            
            dino_features = torch.nn.functional.grid_sample(
                features, 
                warped_grid, 
                align_corners=True,
                padding_mode="zeros"
            )

            dino_features = einops.rearrange(
                dino_features, "b c h w -> b (h w) c", h=dh, w=dw
            )

            dino_features += self.proj(dino_features)

        return delta_offsets.squeeze(-1)


if __name__ == "__main__":

    m = DETRStraighter(
        dim=256,
        n_blocks=8,
        n_tokens_row_cols=2,
        pos_ranges=[(-1, 1)]
    )
    
    x = torch.randn(1, 3, 224, 224)
    z = m(x)
    
    print(z.shape) # (B, n_blocks, n_tokens_row_cols**2 * 2)
    
    # print(z)
    
    einops.rearrange(
        torch.arange(6 * 2).unsqueeze(0), "b (n xy) -> b n (xy)", xy=2, n=6
    )
