import os
from PIL import Image, UnidentifiedImageError
import numpy as np


def validate_input(path: str) -> bool:
    """Check file is valid BMP before processing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if not path.lower().endswith('.bmp'):
        raise ValueError("Input file must have .bmp extension")

    try:
        with Image.open(path) as img:
            if img.format != 'BMP':
                raise ValueError("Input file is not BMP format")
            img.verify()
    except UnidentifiedImageError as e:
        raise UnidentifiedImageError(f"Unable to read BMP: {e}")

    return True


def load_bmp(path: str) -> tuple[np.ndarray, int, int]:
    """Load BMP file. Returns (rgb_array, width, height)."""
    validate_input(path)
    # print(f"Loading BMP: {path}")
    with Image.open(path) as img:
        
        
        if img.format != 'BMP':
            raise ValueError("Input file is not BMP format")
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]

    return arr, w, h


def save_bmp(array: np.ndarray, path: str, original_size: tuple):
    """Save output as BMP."""
    out_h, out_w = array.shape[:2]
    orig_w, orig_h = original_size

    assert out_w == orig_w and out_h == orig_h, \
        f"Output size mismatch {out_w}x{out_h} != {orig_w}x{orig_h}"

    unique_colors = np.unique(array.reshape(-1, 3), axis=0)
    assert 2 <= unique_colors.shape[0] <= 3, \
        f"Output color count must be 2 or 3, got {unique_colors.shape[0]}"

    out_img = Image.fromarray(array.astype(np.uint8), mode='RGB')
    out_img.save(path, format='BMP')
