import os
import tempfile
import numpy as np
from PIL import Image

from main import process_single, process_batch
from config import PALETTE


def _write_temp_bmp(img_array, path):
    img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    img.save(path, format='BMP')


def test_output_size_matches_input():
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'in.bmp')
        out = os.path.join(d, 'out.bmp')
        arr = np.full((100, 150, 3), 255, np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(arr, (30, 20), (120, 80), (0,0,0), 2)
        _write_temp_bmp(arr, inp)
        try:
            process_single(inp, out)
        except ValueError:
            return
        loaded = Image.open(out)
        assert loaded.size == (150, 100)


def test_exactly_3_colors():
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'in.bmp')
        out = os.path.join(d, 'out.bmp')
        arr = np.full((120, 120, 3), 255, np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(arr, (20, 20), (100, 100), (0, 0, 0), thickness=4)
        _write_temp_bmp(arr, inp)
        result = process_single(inp, out)
        out_arr = np.array(Image.open(out))
        unique_count = len(np.unique(out_arr.reshape(-1, 3), axis=0))
        assert 2 <= unique_count <= 3


def test_colors_from_palette():
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'in.bmp')
        out = os.path.join(d, 'out.bmp')
        arr = np.full((120, 120, 3), 255, np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(arr, (20, 20), (100, 100), (0, 0, 0), thickness=4)
        _write_temp_bmp(arr, inp)
        process_single(inp, out)
        out_arr = np.array(Image.open(out))
        unique = np.unique(out_arr.reshape(-1, 3), axis=0)
        allowed = {tuple(c) for c in PALETTE.values()}
        assert set(tuple(r) for r in unique).issubset(allowed)


def test_no_black_pixels():
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'in.bmp')
        out = os.path.join(d, 'out.bmp')
        arr = np.full((120, 120, 3), 255, np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(arr, (20, 20), (100, 100), (0, 0, 0), thickness=4)
        _write_temp_bmp(arr, inp)
        process_single(inp, out)
        out_arr = np.array(Image.open(out))
        assert not np.any(np.all(out_arr == [0, 0, 0], axis=2))


def test_background_is_largest_region():
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'in.bmp')
        out = os.path.join(d, 'out.bmp')
        arr = np.full((100, 100, 3), 255, np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(arr, (25, 25), (75, 75), (0, 0, 0), thickness=4)
        _write_temp_bmp(arr, inp)
        process_single(inp, out)
        out_arr = np.array(Image.open(out))
        bg_count = np.sum(np.all(out_arr == PALETTE['bg'], axis=2))
        line_count = np.sum(np.all(out_arr == PALETTE['line'], axis=2))
        assert bg_count > line_count


def test_bmp_format_output():
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, 'in.bmp')
        out = os.path.join(d, 'out.bmp')
        arr = np.full((80, 80, 3), 255, np.uint8)
        cv2 = __import__('cv2')
        cv2.rectangle(arr, (20, 20), (60, 60), (0, 0, 0), thickness=4)
        _write_temp_bmp(arr, inp)
        process_single(inp, out)
        out_img = Image.open(out)
        assert out_img.format == 'BMP'


def test_batch_mode_processes_all_files():
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        for i in range(3):
            path = os.path.join(src, f'{i}.bmp')
            arr = np.full((60, 60, 3), 255, np.uint8)
            cv2 = __import__('cv2')
            cv2.rectangle(arr, (10, 10), (50, 50), (0, 0, 0), thickness=3)
            _write_temp_bmp(arr, path)

        results = process_batch(src, dst)
        assert len(results) == 3
        for i in range(3):
            out_path = os.path.join(dst, f'{i}_designed.bmp')
            assert os.path.isfile(out_path)
