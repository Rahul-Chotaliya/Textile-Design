import cv2
import numpy as np
from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, NLM_H, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW


def detect_background(gray: np.ndarray) -> tuple[float, bool]:
    c = 20
    corners = [gray[:c, :c], gray[:c, -c:], gray[-c:, :c], gray[-c:, -c:]]
    corner_medians = [np.median(x) for x in corners]
    bg_brightness = float(np.median(corner_medians))
    is_dark_bg = bg_brightness < 128
    return bg_brightness, is_dark_bg


def detect_noise_level(gray_eq: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_eq, cv2.CV_64F)
    noise_level = float(np.std(lap))
    return noise_level


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_GRID)
    gray_eq = clahe.apply(gray)
    return gray_eq


def apply_nlm_denoising(gray_eq: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray_eq,
                                        h=NLM_H,
                                        templateWindowSize=NLM_TEMPLATE_WINDOW,
                                        searchWindowSize=NLM_SEARCH_WINDOW)
    return denoised
