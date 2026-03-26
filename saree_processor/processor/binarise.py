import cv2
import numpy as np
from config import ADAPTIVE_C_VALUE, CANNY_LOW, CANNY_HIGH


def method_otsu(denoised: np.ndarray, is_dark_bg: bool) -> np.ndarray:
    _, mask = cv2.threshold(denoised, 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if is_dark_bg:
        mask = cv2.bitwise_not(mask)
    return mask


def method_adaptive_gaussian(denoised: np.ndarray, short_side: int) -> np.ndarray:
    block = 5
    mask = cv2.adaptiveThreshold(denoised, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 blockSize=block,
                                 C=ADAPTIVE_C_VALUE)
    return mask


def method_sauvola(denoised: np.ndarray,
                   window: int = 25,
                   k: float = 0.2,
                   R: float = 128.0) -> np.ndarray:
    img_f = denoised.astype(np.float64)
    mean = cv2.boxFilter(img_f, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(img_f ** 2, -1, (window, window), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
    thresh = mean * (1.0 + k * (std / R - 1.0))
    mask = ((img_f < thresh) * 255).astype(np.uint8)
    return mask


def method_canny_edge(denoised: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(denoised, CANNY_LOW, CANNY_HIGH)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(edges, ek, iterations=1)
    return mask


def score_binarisation(mask: np.ndarray, area: int) -> int:
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    scores = [stats[l, cv2.CC_STAT_AREA] for l in range(1, num)]
    score = sum(1 for a in scores if 50 < a < area * 0.1)
    return score


def select_and_combine(denoised: np.ndarray,
                       noise_level: float,
                       bg_brightness: float,
                       is_dark_bg: bool,
                       short_side: int,
                       area: int) -> np.ndarray:
    if noise_level > 40:
        primary = method_adaptive_gaussian(denoised, short_side)
        primary_name = 'adaptive_gaussian'
    elif bg_brightness < 200:
        primary = method_sauvola(denoised)
        primary_name = 'sauvola'
    else:
        primary = method_otsu(denoised, is_dark_bg)
        primary_name = 'otsu'

    edge_mask = method_canny_edge(denoised)
    bin_mask = cv2.bitwise_or(primary, edge_mask)
    return bin_mask
