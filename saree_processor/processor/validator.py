import cv2
import numpy as np
from config import MAX_STRUCTURE_DEVIATION, PALETTE


def check_structure_deviation(original_gray: np.ndarray,
                               output: np.ndarray) -> tuple[float, bool]:
    _, orig_ref = cv2.threshold(original_gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    orig_struct = orig_ref > 0
    LINE = np.array(PALETTE['line'])
    out_line = np.all(output == LINE, axis=2)

    missed = np.logical_and(orig_struct, ~out_line).sum()
    deviation = missed / (orig_struct.sum() + 1e-9) * 100.0
    passed = deviation < MAX_STRUCTURE_DEVIATION
    return deviation, passed


def check_output_size(output: np.ndarray,
                       original_W: int, original_H: int) -> bool:
    out_H, out_W = output.shape[:2]
    assert out_W == original_W, f"Width mismatch: {out_W} != {original_W}"
    assert out_H == original_H, f"Height mismatch: {out_H} != {original_H}"
    return True


def full_validation_report(output: np.ndarray,
                            original_gray: np.ndarray,
                            original_W: int, original_H: int,
                            metadata: dict) -> dict:
    size_match = check_output_size(output, original_W, original_H)
    unique = np.unique(output.reshape(-1, 3), axis=0)

    deviation, deviation_pass = check_structure_deviation(original_gray, output)
    fill_px = int(metadata.get('fill_pixels', 0))
    line_px = int(metadata.get('line_pixels', 0))
    bg_px = int(metadata.get('bg_pixels', 0))

    return {
        'size_match': size_match,
        'color_count': int(len(unique)),
        'deviation_pct': float(deviation),
        'deviation_pass': bool(deviation_pass),
        'noise_threshold': metadata.get('noise_threshold'),
        'gap_size': metadata.get('gap_size'),
        'bold_size': metadata.get('bold_size'),
        'primary_method': metadata.get('primary_method'),
        'fill_pixels': fill_px,
        'line_pixels': line_px,
        'bg_pixels': bg_px,
        'overall_pass': size_match and deviation_pass and len(unique) == 3
    }
