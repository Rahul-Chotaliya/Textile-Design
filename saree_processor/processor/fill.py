import cv2
import numpy as np
from config import FILL_SEAL_KERNEL, FILL_SEAL_ITERATIONS, FILL_MIN_ABSOLUTE, FILL_MIN_FACTOR


def smart_flood_fill(bold: np.ndarray,
                     W: int, H: int,
                     area: int) -> np.ndarray:
    seal_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FILL_SEAL_KERNEL)
    sealed = cv2.morphologyEx(bold, cv2.MORPH_CLOSE, seal_k, iterations=FILL_SEAL_ITERATIONS)

    inv = cv2.bitwise_not(sealed)
    canvas = inv.copy()

    for x in range(W):
        if canvas[0, x] == 255:
            cv2.floodFill(canvas, None, (x, 0), 128)
        if canvas[H-1, x] == 255:
            cv2.floodFill(canvas, None, (x, H-1), 128)
    for y in range(H):
        if canvas[y, 0] == 255:
            cv2.floodFill(canvas, None, (0, y), 128)
        if canvas[y, W-1] == 255:
            cv2.floodFill(canvas, None, (W-1, y), 128)

    interior = (canvas == 255).astype(np.uint8) * 255

    min_fill = max(FILL_MIN_ABSOLUTE, int(area * FILL_MIN_FACTOR))
    num_f, labf, statsf, _ = cv2.connectedComponentsWithStats(interior, 8)
    fill_mask = np.zeros((H, W), np.uint8)
    for lbl in range(1, num_f):
        if statsf[lbl, cv2.CC_STAT_AREA] >= min_fill:
            fill_mask[labf == lbl] = 1

    return fill_mask
