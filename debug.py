import numpy as np

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    
    return emb

if __name__ == "__main__":

    image_size = 16  # Image size is 16x16
    embed_dim = 64   # Total embedding dimension (can be any even number)

    # Create the grid for a 16x16 image
    grid_h, grid_w = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

    # Flatten the grid
    grid = np.stack([grid_h.ravel(), grid_w.ravel()], axis=0)  # Shape: (2, H*W)
    
    # Generate positional embeddings
    embeddings = get_2d_sincos_pos_embed_from_grid(8, grid)
    
    # plot the embeddings with matplotlib
    import matplotlib.pyplot as plt
    
    plt.imshow(embeddings, aspect='auto')
    plt.savefig('positional_embeddings.png')