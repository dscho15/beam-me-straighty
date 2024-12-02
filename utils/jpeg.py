import torch
from torchvision import io
from torchvision.transforms.functional import convert_image_dtype


def jpeg_compression(x: torch.Tensor, min_quality: int = 70, max_quality: int = 100):
    batch = x.ndim > 3
    if batch:
        x = x.squeeze(0)

    x = convert_image_dtype(x, dtype=torch.uint8)
    quality = torch.randint(min_quality, max_quality + 1, (1,)).item()
    x = io.decode_jpeg(io.encode_jpeg(x, quality=quality)) / 255.0
    if batch:
        x = x.unsqueeze(0)
    return x


class JPEGCompression(torch.nn.Module):
    def __init__(
        self, min_quality: int = 70, max_quality: int = 100, probability: float = 1.0
    ):
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.probability = probability

    def forward(self, x: torch.Tensor):
        if torch.rand(1).item() < self.probability:
            return jpeg_compression(x, self.min_quality, self.max_quality)
        return x


if __name__ == "__main__":
    from utils.save import save_image

    img = io.read_image("./input.jpg").unsqueeze(0) / 255.0
    print(img.shape, img.dtype)

    img = jpeg_compression(img, min_quality=20, max_quality=22)
    print(img.shape, img.dtype)

    save_image(img, "compressed.jpg")
