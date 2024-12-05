from pathlib import Path
from typing import Callable

import cv2
import torch
from frogbox.utils import load_model_checkpoint
from kornia.geometry.transform import warp_perspective, get_perspective_transform
from torchvision import io
from torchvision.transforms.functional import center_crop
from torchvision.utils import flow_to_image

from datasets.dataset import BaseDataset
from utils.resize import resize_max_edge
from utils.save import make_gif, save_image
import numpy as np
from einops import rearrange

# Paths
CKPT_PATH = "#"
TEST_SET = "#"

# Settings
DEVICE = "cuda:0"
MAX_EDGE = 1536
INFERENCE_MAX_EDGE = None # None to use training size

# Output
OUTPUT = ["GIF", "CONF"] # GIF, IMAGE, FLOW, CONF

    
def prepare_input(normalize: Callable,
                  image: torch.Tensor,
                  max_edge: int):
    input = resize_max_edge(image, max_edge)
    h, w = input.shape[-2:]
    input = normalize(input)
    return center_crop(input, max_edge), (h, w)


def main():
    output_dir = Path("data/inference")
    output_dir.mkdir(exist_ok=True, parents=True)

    # model
    model, config = load_model_checkpoint(CKPT_PATH)
    device = torch.device(DEVICE)
    model = model.eval().to(device)
    
    # test set
    params = config.datasets['val'].params
    params['dataframe_path'] = TEST_SET
    dataset = BaseDataset(**params)
    inference_size = dataset.max_edge
    n_bins = dataset.n_bins
    bin_max = dataset.bin_max
    if MAX_EDGE is not None:
        assert inference_size <= MAX_EDGE, "MAX_EDGE is too small"

    for i in range(len(dataset)):

        row = dataset.dataframe.iloc[i]
        original_image = io.read_image(row['TARGET'], io.ImageReadMode.RGB).unsqueeze(0) / 255.0
        if MAX_EDGE is not None:
            original_image = resize_max_edge(original_image, MAX_EDGE)
        
        input = original_image.clone()
        outputs = [original_image]

        input, (i_h, i_w) = prepare_input(dataset.normalization, input, inference_size)
        input = input.to(device)
        with torch.inference_mode():
            logits = model(input)
        input = input.cpu().detach().squeeze(0)
        logits = logits.cpu().detach().squeeze(0) # (8, n_bins)
        
        pred_indices = logits.argmax(dim=-1) # (8,)
        offsets = (-bin_max + pred_indices * (2 * bin_max / (n_bins - 1))).reshape(-1, 2) # (8, 2)

        # Prepare source and destination corners
        src_corners = dataset.get_src_corners(i_h, i_w)
        dst_corners = src_corners + offsets
        dst_corners = dst_corners.to(src_corners.dtype)

        # Compute transform
        transform = get_perspective_transform(src_corners.unsqueeze(0), dst_corners.unsqueeze(0))
        transform = dataset.adjust_transform(transform, (i_h, i_w), original_image.shape[-2:])

        # apply
        warped = warp_perspective(original_image.clone(), transform, dsize=original_image.shape[-2:])
        outputs.append(warped)

        # Plot curves
        probs = torch.softmax(logits, dim=-1)
        fig, ax = plt.subplots(1, 1)
        for j in range(8):
            ax.plot(probs[j].numpy())
        ax.set_title("Confidence")
        ax.set_xlabel("Bins")
        ax.set_ylabel("Probability")
        plt.tight_layout()


        # Save
        name = Path(row['TARGET']).stem
        if "GIF" in OUTPUT:
            make_gif(outputs, output_dir / "ps" / f"{name}.gif")
        if "IMAGE" in OUTPUT:
            save_image(outputs[-1], output_dir / "ps" / f"{name}.jpg")
        plt.savefig(output_dir / "ps" / f"{name}_conf.jpg")
        plt.close()
        
        print(f"Saved {name}")

if __name__ == "__main__":
    main()