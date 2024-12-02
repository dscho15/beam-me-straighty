import torch


class RoPE2D(torch.nn.Module):

    def __init__(self, freq: float = 100.0, F0: float = 1.0):
        super().__init__()

        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(
        self, D: int, seq_len: int, device: torch.device, dtype
    ) -> torch.Tensor:

        if (D, seq_len, device, dtype) not in self.cache:

            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, D, 2).float().to(device) / D)
            )

            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)

            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)

            freqs = torch.cat((freqs, freqs), dim=-1)

            cos = freqs.cos()  # (Seq, Dim)

            sin = freqs.sin()

            self.cache[D, seq_len, device, dtype] = (cos, sin)

        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:

        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(
        self,
        tokens: torch.Tensor,
        pos1d: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:

        assert pos1d.ndim == 2

        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]

        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(
        self, tokens: torch.FloatTensor, positions: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert (
            tokens.size(3) % 2 == 0
        ), "number of dimensions should be a multiple of two"

        D = tokens.size(3) // 2

        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2

        cos, sin = self.get_cos_sin(
            D, int(positions.max()) + 1, tokens.device, tokens.dtype
        )
        # split features into two along the feature dimension, and apply rope1d on each half

        y, x = tokens.chunk(2, dim=-1)

        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)

        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)

        tokens = torch.cat((y, x), dim=-1)

        return tokens
