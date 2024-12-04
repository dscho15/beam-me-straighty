import torch
from timm.models.vision_transformer import LayerScale, DropPath, Mlp

class MHCABlock(torch.nn.Module):

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 2,
        init_values: float = 1e-3,
        drop_path: float = 0.0,
        mlp_ratio: int = 4,
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
        
        x = x + self.drop_path1(self.ls1(self.att(query=z, key=y, value=y, need_weights=False)[0]))
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x
