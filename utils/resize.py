from torchvision.transforms.functional import resize, InterpolationMode


def resize_max_edge(
    image,
    size: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = True,
):
    height, width = image.shape[-2:]
    scale = min(size / height, size / width)
    new_height = round(scale * height)
    new_width = round(scale * width)
    return resize(
        image,
        size=(new_height, new_width),
        interpolation=interpolation,
        antialias=antialias,
    )
