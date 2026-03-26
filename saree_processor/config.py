# ── PALETTE (3 colors — fixed for all outputs) ─────────────────────
PALETTE = {
    "bg":   [245, 240, 225],   # Cream  — saree background
    "line": [ 90,  20,  50],   # Deep Burgundy — all design lines
    "fill": [195, 145,  30],   # Gold — interior decorative regions
}

# ── BINARISATION ───────────────────────────────────────────────────
CLAHE_CLIP_LIMIT        = 0.5     # CLAHE contrast limit
CLAHE_TILE_GRID         = (16,16) # CLAHE grid size
NLM_H                   = 10      # NLM filter strength (higher=more denoise)
NLM_TEMPLATE_WINDOW     = 7       # NLM template window size
NLM_SEARCH_WINDOW       = 21      # NLM search window size
ADAPTIVE_C_VALUE        = 15      # Adaptive threshold constant C
CANNY_LOW               = 50      # Canny lower threshold
CANNY_HIGH              = 150     # Canny upper threshold

# ── NOISE REMOVAL ──────────────────────────────────────────────────
NOISE_LOSS_BUDGET       = 0.01   # Max 1.5% of line pixels can be lost
NOISE_CAP_FACTOR        = 0.05    # Max noise threshold = short_side * 0.05

# ── GAP CLOSING ────────────────────────────────────────────────────
GAP_SIZE_FACTOR         = 0.001  # gap_kernel = short_side * factor
GAP_MIN_SIZE            = 3       # minimum gap kernel size (px)
GAP_ITERATIONS          = 0      # morphologyEx iterations
REPAIR_GAP_MULTIPLIER   = 1.5       # endpoint repair = gap_size * this

# ── BOLD DILATION ──────────────────────────────────────────────────
BOLD_SPARSE_THRESHOLD   = 0.05    # stroke_density < this = sparse
BOLD_MEDIUM_THRESHOLD   = 0.15    # stroke_density < this = medium
BOLD_SPARSE_FACTOR  = 0.001
BOLD_MEDIUM_FACTOR  = 0.002
BOLD_DENSE_FACTOR   = 0.003
BOLD_MIN_SPARSE         = 1       # minimum bold kernel sizes
BOLD_MIN_MEDIUM         = 1
BOLD_MIN_DENSE          = 1
BOLD_ITERATIONS         = 1       # dilation iterations

# ── FILL ───────────────────────────────────────────────────────────
FILL_MIN_FACTOR         = 0.00005 # min fill region = area * factor
FILL_MIN_ABSOLUTE       = 1      # absolute minimum fill region (px)
FILL_SEAL_KERNEL        = (9, 9)  # seal open contours before fill
FILL_SEAL_ITERATIONS    = 3

# ── VALIDATION ─────────────────────────────────────────────────────
MAX_STRUCTURE_DEVIATION = 3.0     # max allowed % deviation
CONNECTIVITY            = 8       # always 8-connected components
