import cv2
import numpy as np
from skimage.morphology import skeletonize
from config import (NOISE_LOSS_BUDGET, NOISE_CAP_FACTOR,
                    GAP_SIZE_FACTOR, GAP_MIN_SIZE, GAP_ITERATIONS,
                    REPAIR_GAP_MULTIPLIER,
                    BOLD_SPARSE_THRESHOLD, BOLD_MEDIUM_THRESHOLD,
                    BOLD_SPARSE_FACTOR, BOLD_MEDIUM_FACTOR,
                    BOLD_DENSE_FACTOR, BOLD_MIN_SPARSE,
                    BOLD_MIN_MEDIUM, BOLD_MIN_DENSE,
                    BOLD_ITERATIONS)


def dynamic_noise_removal(bin_mask: np.ndarray, short_side: int) -> tuple[np.ndarray, int]:
    num_l, lab, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    all_areas = sorted([stats[l, cv2.CC_STAT_AREA] for l in range(1, num_l)])
    total_px = int((bin_mask > 0).sum())

    thresh = 1
    cap = max(1, int(short_side * NOISE_CAP_FACTOR))
    for candidate in range(1, cap):
        lost = sum(a for a in all_areas if a <= candidate)
        if lost / (total_px + 1e-9) > NOISE_LOSS_BUDGET:
            break
        thresh = candidate

    clean = np.zeros_like(bin_mask)
    for lbl in range(1, num_l):
        if stats[lbl, cv2.CC_STAT_AREA] > thresh:
            clean[lab == lbl] = 255

    return clean, thresh


def scale_aware_gap_closing(clean: np.ndarray, short_side: int) -> np.ndarray:
    gap_size = max(GAP_MIN_SIZE, int(short_side * GAP_SIZE_FACTOR))
    gap_size = max(1, gap_size)
    gap_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, gap_k, iterations=GAP_ITERATIONS)
    return closed


def repair_broken_contours(mask: np.ndarray, max_gap: int) -> np.ndarray:
    skel = skeletonize((mask // 255).astype(np.uint8)).astype(np.uint8) * 255
    k3 = np.ones((3, 3), np.float32)
    nc = cv2.filter2D((skel > 0).astype(np.float32), -1, k3)
    ep = np.argwhere((skel > 0) & (nc <= 2))

    repaired = mask.copy()
    for i in range(len(ep)):
        r1, c1 = ep[i]
        if i + 1 >= len(ep):
            break
        dists = np.hypot(ep[i+1:,0] - r1, ep[i+1:,1] - c1)
        near = np.where(dists <= max_gap)[0]
        for j in near:
            r2, c2 = ep[i+1+j]
            cv2.line(repaired, (c1, r1), (c2, r2), 255, 1)

    return repaired


def scale_aware_bold_dilation(closed: np.ndarray,
                               stroke_density: float,
                               short_side: int) -> tuple[np.ndarray, int]:
    if stroke_density < BOLD_SPARSE_THRESHOLD:
        bold_size = max(BOLD_MIN_SPARSE, int(short_side * BOLD_SPARSE_FACTOR))
    elif stroke_density < BOLD_MEDIUM_THRESHOLD:
        bold_size = max(BOLD_MIN_MEDIUM, int(short_side * BOLD_MEDIUM_FACTOR))
    else:
        bold_size = max(BOLD_MIN_DENSE, int(short_side * BOLD_DENSE_FACTOR))

    bold_size = max(1, bold_size)
    bold_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bold_size, bold_size))
    bold = cv2.dilate(closed, bold_k, iterations=BOLD_ITERATIONS)
    return bold, bold_size
