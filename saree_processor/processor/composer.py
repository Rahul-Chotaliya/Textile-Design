import numpy as np
from config import PALETTE


def compose_output(bold: np.ndarray,
                   fill_mask: np.ndarray,
                   H: int, W: int) -> np.ndarray:
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[:] = PALETTE['bg']

    out[fill_mask == 1] = PALETTE['fill']
    out[bold == 255] = PALETTE['line']

    line_px = np.all(out == PALETTE['line'], axis=2)
    fill_px = np.all(out == PALETTE['fill'], axis=2)
    out[~line_px & ~fill_px] = PALETTE['bg']

    return out


def verify_exactly_3_colors(out: np.ndarray) -> bool:
    unique = np.unique(out.reshape(-1, 3), axis=0)
    assert 2 <= len(unique) <= 3, f"Expected 2 or 3 unique colors, got {len(unique)}"
    allowed = [tuple(PALETTE['bg']), tuple(PALETTE['line']), tuple(PALETTE['fill'])]
    for row in unique:
        assert tuple(row) in allowed, f"Color {tuple(row)} not in palette"
    return True
