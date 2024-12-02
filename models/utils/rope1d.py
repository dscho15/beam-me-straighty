import torch


class RoPE1D(torch.nn.Module):

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

        x1, x2 = torch.chunk(x, 2, dim=-1)

        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(
        self,
        tokens: torch.Tensor,
        pos1d: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:

        # print(pos1d.shape)
        assert pos1d.ndim == 2, "pos1d should be 2D"

        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]

        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(
        self, tokens: torch.FloatTensor, positions: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens (x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        # assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"

        D = tokens.size(3)

        assert positions.ndim == 2  # Batch, Seq

        cos, sin = self.get_cos_sin(
            D, int(positions.max()) + 1, tokens.device, tokens.dtype
        )

        return self.apply_rope1d(tokens, positions, cos, sin)


if __name__ == "__main__":

    rope = RoPE1D()

    b, nh, ntokens, dim = 1, 1, 4, 4

    tokens = torch.eye(ntokens).reshape(1, 1, ntokens, ntokens)
    positions = torch.arange(0, ntokens).reshape(1, ntokens).repeat(b, 1)

    rope(tokens, positions)
