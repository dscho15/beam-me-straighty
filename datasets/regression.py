import json
from typing import Sequence, Tuple, Union
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms.functional import center_crop
from kornia.geometry.transform import warp_perspective, get_perspective_transform
from functools import reduce
import kornia.augmentation as K
from einops import repeat

from utils.jpeg import JPEGCompression

from utils.resize import resize_max_edge


class BaseDataset(Dataset):
    def __init__(self,
                 dataframe_path: Union[str, Sequence[str]],
                 max_edge: int,
                 normalize_offsets: bool = True,
                 augment: bool = False,
                 normalize: bool = False,
                 mean: Sequence[float] = (0.5, 0.5, 0.5),
                 std: Sequence[float] = (0.5, 0.5, 0.5),
                 return_crop_size: bool = False,
                 p_jpeg: float = 0.5,
                 p_jiggle: float = 0.5,
                 jpeg_quality_min: float = 70,
                 jpeg_quality_max: float = 95,
                 seed: int = None,
                 n: int = -1
                 ):
        super().__init__()
        if isinstance(dataframe_path, str):
            self.dataframe = pd.read_csv(dataframe_path)
        else:
            self.dataframe = pd.concat([pd.read_csv(path) for path in dataframe_path])
        if n > 0:
            self.dataframe = self.dataframe.sample(n)

        self.max_edge = max_edge
        self.normalize_offsets = normalize_offsets
        self.augment = augment
        self.normalize = normalize
        self.mean, self.std = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
        self.return_crop_size = return_crop_size
        self.jpeg_compression = JPEGCompression(min_quality=jpeg_quality_min,
                                                max_quality=jpeg_quality_max,
                                                probability=p_jpeg)
        self.color_jiggle = K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=p_jiggle, keepdim=True)

        self.seed = seed
        self.reset_rng()
    
    def reset_rng(self):
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.dataframe)
    
    def load_item(self, idx: int):
        row = self.dataframe.iloc[idx]
        image = io.read_image(row['ORIGINAL'], io.ImageReadMode.RGB).unsqueeze(0) / 255.0
        if max(image.shape) != row['MAX_EDGE']:
            image = resize_max_edge(image, row['MAX_EDGE'])
        transform = torch.tensor(json.loads(row['TRANSFORM'])).reshape(1, 3, 3)
        return image, transform
    
    def normalization(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std
    
    def denormalization(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.std + self.mean
    
    def adjust_transform(self, 
                          transform: torch.Tensor, 
                          original_size: Tuple[int, int] = None, 
                          new_size: Tuple[int, int] = None,
                          scale: Tuple[float, float] = None,
                          padding_x: float = 0,
                          padding_y: float = 0) -> torch.Tensor:
        """Adjusts the transform matrix to account for resizing

        Args:
            transform (torch.Tensor): The original transform matrix
            original_size (Tuple[int, int]): The original size of the image (height, width)
            new_size (Tuple[int, int]): The new size of the image (height, width)
            scale (Tuple[float, float], optional): The scale factor (height, width).

        Returns:
            torch.Tensor: The adjusted transform matrix
        """
        if original_size is not None or new_size is not None:
            assert original_size is not None, "original_size must be provided if new_size is provided"
            assert new_size is not None, "new_size must be provided if original_size is provided"
            assert scale is None, "scale must not be provided if original_size is provided"

        # scale
        if scale is None:
            oh, ow = original_size
            nh, nw = new_size
            sx, sy = ow / nw, oh / nh
        else:
            sy, sx = scale
        scale = torch.tensor([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ], dtype=transform.dtype, device=transform.device)

        # shift
        shift = torch.tensor([
            [1, 0, -padding_x],
            [0, 1, -padding_y],
            [0, 0, 1]
        ], dtype=transform.dtype, device=transform.device)

        transform = torch.linalg.inv(scale) @ transform @ scale
        transform = torch.linalg.inv(shift) @ transform @ shift

        return transform       

    def get_src_corners(self,
                      image_height: int,
                      image_width: int,
                      homogenous: bool = False,
                      dtype = None) -> torch.Tensor:
        corners = torch.tensor([
            [0, 0],                             # top-left
            [image_width - 1, 0],               # top-right
            [0, image_height - 1],              # bottom-left
            [image_width - 1, image_height - 1],# bottom-right
        ], dtype=torch.float32) # (4, 2) (x, y)

        if homogenous:
            corners = torch.cat([corners, torch.ones(4, 1, dtype=dtype)], dim=-1)
        return corners
    
    def get_offsets(self, height: int, width: int, transform: torch.Tensor) -> torch.Tensor:
    
        src_corners = self.get_src_corners(height,
                                           width,
                                           homogenous=True,
                                           dtype=transform.dtype)
        
        dst_corners = (src_corners @ transform.squeeze(0).T)
        dst_corners = dst_corners / dst_corners[..., -1].unsqueeze(-1)
        offsets = (dst_corners[..., :-1] - src_corners[..., :-1]).reshape(1, 4, 2)

        return offsets

    def __getitem__(self, idx: int):
        image, transform = self.load_item(idx)
        shape = image.shape # (1, 3, H, W)

        # Resize while keeping the aspect ratio
        image = resize_max_edge(image, self.max_edge)
        height, width = image.shape[-2:]

        # Augmentations before padding and normalization
        flipped = False
        if self.augment:
            if self.rng.choice([True, False]):
                image = image.flip(-1)
                flipped = True
            image = self.color_jiggle(image)
            image = self.jpeg_compression(image)
        
        if self.normalize:
            image = self.normalization(image)

        # Pad to achieve the desired size
        image = center_crop(image, self.max_edge)

        # Adjust transformation matrix to account for resizing and padding
        s = max(shape[-2:]) / self.max_edge
        padding_y = (self.max_edge - height) / 2
        padding_x = (self.max_edge - width) / 2
        transform = self.adjust_transform(transform, scale=(s, s), padding_x=padding_x, padding_y=padding_y)

        # Compute offsets and account for flipping
        offsets = self.get_offsets(self.max_edge, self.max_edge, transform) # (1, 4, 2), raster order
        if flipped:
            offsets = offsets * torch.tensor([-1, 1], dtype=offsets.dtype, device=offsets.device).reshape(1, 1, 2) # flip x component
            offsets = torch.stack([offsets[:, 1], offsets[:, 0], offsets[:, 3], offsets[:, 2]], dim=1) # tl <-> tr, bl <-> br
        
        # Normalize offsets to account for image coordinate range of [-1, 1]
        if self.normalize_offsets:
            offsets = 2 * offsets / self.max_edge
        
        image = image.reshape(3, *image.shape[-2:]).contiguous()
        
        offsets = offsets.reshape(-1).contiguous() # (1, 4, 2) -> (8)
        
        if self.return_crop_size:
            return image, (offsets, (height, width))
        
        return image, offsets

if __name__ == '__main__':
    from utils.save import save_image

    max_edge = 512
    dataset = BaseDataset(dataframe_path='data/interior/ps-straighten-ok-train-interior.csv',
                             max_edge=max_edge,
                             augment=True,
                             normalize=False,
                             return_crop_size=True,)
    image, (offsets, (height, width)) = dataset[120] # 205, 120

    save_image(image, 'input.png')

    offsets = offsets.reshape(4, 2) * (max_edge / 2)
    src_corners = dataset.get_src_corners(max_edge, max_edge, homogenous=False)
    dst_corners = src_corners + offsets

    if src_corners.dtype != dst_corners.dtype:
        src_corners = src_corners.to(dst_corners.dtype)
    estimated_transform = get_perspective_transform(src_corners.unsqueeze(0), dst_corners.unsqueeze(0))

    # # Apply
    image = warp_perspective(image.unsqueeze(0), estimated_transform, (max_edge, max_edge))
    save_image(image, 'output.png')
    