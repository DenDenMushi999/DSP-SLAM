import os
import glob
import re
from pathlib import Path

import argparse
import numpy as np
import cv2
from reconstruct.optimizer import MeshExtractor
from reconstruct.utils import get_configs, get_decoder, write_mesh_to_ply

def frames_sort_key(fname):
    # pattern .../000123.png
    return int(re.findall(r'\d+', str(fname))[-1])

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames_dir', type=str, required=True, help='path to frames directory')
    parser.add_argument('-m', '--masks_dir', type=str, required=True, help='path to masks directory')
    parser.add_argument('-o', '--save_dir', type=str, required=True, help='path to save directory')
    return parser

def draw_mask(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    # colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    # colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def draw_masks(frames_dir, masks_dir, save_dir):
    frame_paths = sorted(glob.glob(str(frames_dir/'*.png')), key=frames_sort_key)
    mask_paths = sorted(glob.glob(str(frames_dir/'*.png')), key=frames_sort_key)
    assert len(frame_paths) == len(mask_paths), f'The number of images and masks are not equal: {len(frame_paths)}, {len(mask_paths)}'
    
    for frame_path, mask_path in zip(frame_paths, mask_paths):
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path)
        frame_id = re.findall(r'\d+', str(frame_path))[-1]
        assert frame_id is not None

        assert (frame is not None) and (mask is not None)
        assert len(frame.shape) == 3
        if len(mask.shape) == 2:
            print('adding dim to mask')
            mask = np.expand_dims(mask, 2).repeat(3, axis=2)
        frame_shape = frame.shape
        mask_shape = mask.shape
        d_h = frame_shape[0] - mask_shape[0]
        d_w = frame_shape[1] - mask_shape[1]
        assert (not d_w%2) and (not d_h%2), f'Mask has incorrect shape {frame.shape, mask.shape}'
        if d_h and d_w:
            print('padding')
            mask = np.pad(mask, ((d_h//2,d_h//2), (d_w//2,d_w//2),(0,0)) )
        print(mask.max())
        cv2.imwrite(str(save_dir/(frame_id+'-Masked-frame.png')), draw_mask(frame, mask, color=(0,255,0), alpha=0.3))

if __name__ == "__main__":

    parser = args_parser()
    args = parser.parse_args()

    draw_masks(Path(args.frames_dir), Path(args.masks_dir), Path(args.save_dir))