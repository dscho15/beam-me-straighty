import torch
import einops
import kornia

from transformers import Dinov2Model

from models.utils.cross_attention import MHCABlock
from timm.models.vision_transformer import Block as MHSABlock

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
    
class PosUpdateBlock(torch.nn.Module):
    
    def __init__(self, dim: int, start: float, end: float, n_bins: int):
        super(PosUpdateBlock, self).__init__()
        
        self.weighting_function = torch.linspace(-start, end, n_bins)
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(dim, n_bins),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.FloatTensor, return_distribution: bool = False) -> torch.FloatTensor:
        
        weighting_function = einops.rearrange(self.weighting_function.to(x.device), "d -> 1 1 d")

        x = self.linear(x)
        
        z = torch.mean(weighting_function * x, dim=-1, keepdim=True)
                        
        if return_distribution:
            return z, x
        else:
            return z


def generate_grid(x_steps: int = 3, y_steps: int = 3, x_range: tuple = (0, 1), y_range: tuple = (0, 1)):
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
    
    grid = [(x, y) for x in x_values for y in y_values]
    
    return grid
    
    



class DETRStraighter(torch.nn.Module):

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        n_blocks: int = 8,
        n_tokens_row_cols: int = 2,
        with_pos_embeddings: bool = True,
        n_bins: int = 301,
        pos_ranges: list = [(-1, 1), (-0.5, 0.5), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25)]
    ):
        super(DETRStraighter, self).__init__()

        self.backbone = DinoFeaturePyramid()
        self.proj_dino_features = torch.nn.Linear(384, dim)
        self.n_blocks = n_blocks

        self.MHSABlocks = torch.nn.ModuleList(
            [MHSABlock(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
        )
        
        self.MHCABlocks = torch.nn.ModuleList(
            [MHCABlock(dim=dim, num_heads=num_heads) for _ in range(n_blocks)]
        )
        
        self.discretize_blocks = torch.nn.ModuleList(
            [torch.nn.Linear(dim, n_bins) for _ in range(n_blocks)]
        )

        self.tokens = torch.nn.Parameter(torch.randn(n_tokens_row_cols ** 2 * 2, dim))
        self.pos_tokens = torch.zeros(n_tokens_row_cols ** 2 * 2)
        torch.nn.init.normal_(self.tokens, std=0.02)
        

        self.output_tokens = torch.nn.Linear(dim, n_bins)
        
        torch.nn.init.normal_(self.output_tokens.weight, std=0.02)

        self.with_pos_embeddings = with_pos_embeddings
        
        if self.with_pos_embeddings:
            
            self.proj_pos_embeddings = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
            )
            
        self.pos_ranges = pos_ranges
        self.PosUpdateBlocks = torch.nn.ModuleList(
            [PosUpdateBlock(dim, start, end, n_bins) for start, end in pos_ranges]
        )
        
        self.fixed_positions = torch.tensor(generate_grid(n_tokens_row_cols, n_tokens_row_cols, (0, 1), (0, 1)))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        features = self.backbone(x)
        
        _, dh, dw, _ = features.shape

        features = einops.rearrange(features, "b h w c -> b (h w) c")

        features = self.proj_dino_features(features)

        tokens = self.tokens.unsqueeze(0).repeat(features.shape[0], 1, 1)

        if self.with_pos_embeddings:

            grid = torch.arange(tokens.size(1), device=features.device).unsqueeze(0)
            
            pos_embeds = get_1d_sincos_pos_embed_from_grid(tokens.size(-1), grid)
            
            pos_embeds = pos_embeds.unsqueeze(1)
            
            pos_embeds = einops.rearrange(pos_embeds, "h w c -> (h w) c")
            
            pos_embeds = pos_embeds.to(tokens.device)

            tokens += pos_embeds.unsqueeze(0)
            
            tokens = self.proj_pos_embeddings(tokens)
            
        delta_offsets = torch.zeros(tokens.shape[0], self.n_blocks, tokens.shape[1], 1).to(tokens.device)
                            
        for i, (mhca, mhsa, pib) in enumerate(zip(self.MHCABlocks, self.MHSABlocks, self.PosUpdateBlocks)):

            tokens = mhca(tokens, features)

            tokens = mhsa(tokens)
            
            d_o = pib(tokens)

            if i == 0:
                delta_offsets[:, i, ...] = d_o
            else:
                delta_offsets[:, i, ...] = d_o + delta_offsets[:, i-1, ...]
            
            points1 = self.fixed_positions.unsqueeze(0).repeat(tokens.shape[0], 1, 1) # b n 2
            points2 = points1 + einops.rearrange(delta_offsets[:, i, ...], "b (n xy) 1 -> b n xy", n=points1.size(1), xy=2)
            
            H = kornia.geometry.homography.find_homography_dlt(points1, points2).unsqueeze(1) # (B 1 3 3)
            
            

            features = einops.rearrange(features, "b (h w) c -> b h w c", h=dh, w=dw)
        
        delta_offsets = torch.cumsum(delta_offsets, dim=1)
        
        return delta_offsets
    
if __name__ == "__main__":
    
    m = DETRStraighter(256)
    x = torch.randn(2, 3, 224, 224)
    z = m(x)