"""CLI entrypoint for saree_processor pipeline."""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import UnidentifiedImageError

from processor.loader import load_bmp, save_bmp, validate_input
from processor.preprocess import detect_background, detect_noise_level, apply_clahe, apply_nlm_denoising
from processor.binarise import select_and_combine
from processor.morphology import dynamic_noise_removal, scale_aware_gap_closing, repair_broken_contours, scale_aware_bold_dilation
from processor.fill import smart_flood_fill
from processor.composer import compose_output, verify_exactly_3_colors
from processor.validator import check_structure_deviation, check_output_size, full_validation_report
from config import REPAIR_GAP_MULTIPLIER


def process_single(input_path: str, output_path: str, palette: dict = None) -> dict:
    if not input_path.lower().endswith('.bmp') or not output_path.lower().endswith('.bmp'):
        raise ValueError('Input and output must be BMP paths')

    validate_input(input_path)
    arr, W, H = load_bmp(input_path)

    if W <= 0 or H <= 0:
        raise ValueError('Invalid image size')

    short_side = min(W, H)
    long_side = max(W, H)

    if long_side > 4000:
        print(f"[WARN] Very large image ({W}x{H}) - NLM may be slow")

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    source_all_black = np.all(gray == 0)
    source_all_white = np.all(gray == 255)

    if source_all_black:
        gray = cv2.bitwise_not(gray)

    if source_all_white:
        # continue pipeline - it may be empty
        pass

    is_binary_source = len(np.unique(gray)) <= 2

    bg_brightness, is_dark_bg = detect_background(gray)

    if is_binary_source:
        gray_eq = gray.copy()
        denoised = gray.copy()
        noise_level = detect_noise_level(gray_eq)
    else:
        gray_eq = apply_clahe(gray)
        denoised = apply_nlm_denoising(gray_eq)
        noise_level = detect_noise_level(gray_eq)

    bin_mask = select_and_combine(denoised, noise_level, bg_brightness, is_dark_bg, short_side, W * H)

    if np.count_nonzero(bin_mask) == 0:
        raise ValueError('No design strokes detected in image')

    if short_side < 200:
        forced_noise_thresh = 5
    else:
        forced_noise_thresh = None

    clean, noise_threshold = dynamic_noise_removal(bin_mask, short_side)
    if forced_noise_thresh is not None:
        noise_threshold = max(noise_threshold, forced_noise_thresh)

    gap_size = max(1, int(short_side * 0.005)) if short_side < 200 else max(3, int(short_side * 0.005))
    closed = scale_aware_gap_closing(clean, short_side)
    repaired = repair_broken_contours(closed, max_gap=gap_size * REPAIR_GAP_MULTIPLIER)

    stroke_density = float(np.count_nonzero(bin_mask)) / float(W * H)
    bold, bold_size = scale_aware_bold_dilation(repaired, stroke_density, short_side)
    if short_side < 200:
        bold_size = max(bold_size, 2)

    fill_mask = smart_flood_fill(bold, W, H, W * H)

    output = compose_output(bold, fill_mask, H, W)
    verify_exactly_3_colors(output)

    dev, passed = check_structure_deviation(gray, output)
    check_output_size(output, W, H)

    report = full_validation_report(output, gray, W, H, {
        'noise_threshold': noise_threshold,
        'gap_size': gap_size,
        'bold_size': bold_size,
        'primary_method': 'adaptive_gaussian' if noise_level > 40 else 'sauvola' if bg_brightness < 200 else 'otsu',
        'fill_pixels': int(np.count_nonzero(fill_mask)),
        'line_pixels': int(np.count_nonzero(np.all(output == [90, 20, 50], axis=2))),
        'bg_pixels': int(np.count_nonzero(np.all(output == [245, 240, 225], axis=2)))
    })

    save_bmp(output, output_path, (W, H))

    report.update({'deviation_pct': dev, 'deviation_pass': passed, 'overall_pass': report.get('overall_pass', False)})
    return report


def process_batch(input_dir: str, output_dir: str) -> list[dict]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.bmp')]
    results = []

    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + '_designed.bmp'
        out_path = os.path.join(output_dir, out_name)

        try:
            result = process_single(in_path, out_path)
            results.append({'file': fname, 'status': 'ok', **result})
        except Exception as exc:
            print(f"[ERROR] {fname}: {exc}")
            results.append({'file': fname, 'status': 'error', 'error': str(exc)})

    passed = sum(1 for r in results if r.get('overall_pass'))
    print(f"[SUMMARY] {passed}/{len(results)} files passed")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saree Design Processor')
    parser.add_argument('input', help='Input BMP path or folder')
    parser.add_argument('output', help='Output BMP path or folder')
    parser.add_argument('--batch', action='store_true', help='Batch mode for folders')

    args = parser.parse_args()

    if args.batch:
        process_batch(args.input, args.output)
    else:
        process_single(args.input, args.output)
