import os
from pathlib import Path

import torch
from torchvision import io
from torchvision.transforms.functional import convert_image_dtype
import imageio.v2 as imageio
from typing import List
import uuid


def save_image(image: torch.Tensor, filename: Path, quality: int = None):
    filename = Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim > 3:
        image = image.squeeze(0)
    image = convert_image_dtype(image, dtype=torch.uint8)
    if image.device != torch.device("cpu"):
        image = image.cpu()
    if filename.suffix == ".png":
        quality = 0 if quality is None else quality
        io.write_png(image, str(filename), compression_level=quality)
    elif filename.suffix in [".jpg", ".jpeg"]:
        quality = 95 if quality is None else quality
        io.write_jpeg(image, str(filename), quality=quality)


def make_gif(images: List[torch.Tensor], filename: Path, duration: float = 0.5):
    try:
        # make sure directory exists
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # write files to disk temporarily
        tmp_filenames = []
        for image in images:
            tmp_filename = f"{uuid.uuid4()}.jpg"
            tmp_filenames.append(tmp_filename)
            save_image(image, tmp_filename)

        # make gif
        images = [imageio.imread(f) for f in tmp_filenames]
        imageio.mimsave(filename, images, duration=duration)

        # cleanup
        for f in tmp_filenames:
            Path(f).unlink()

    except KeyboardInterrupt as e:
        for f in tmp_filenames:
            Path(f).unlink()
        raise e
