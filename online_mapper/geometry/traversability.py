"""Traversability map from a camera-frame point cloud.

Input: points_camera (H,W,3) np.float32 in camera frame
       (x right, y down, z forward; meters).
Output: per-pixel traversability score map (H,W) in [0, 1]:
   1.0 = ground plane
   0.0 = obstacle (wall / pillar / chair / person / tall object)
  ~0.5 = unknown (missing depth or too far)
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Thresholds assume ~1.2-1.5 m camera height (indoor + outdoor mix):
GROUND_BAND_M = 0.30     # |Δy - y_ground| ≤ GROUND_BAND → traversable
OBSTACLE_ABOVE_M = 0.25  # y < y_ground - this → obstacle (taller than ground)
MAX_DEPTH_M = 12.0       # beyond this, depth reliability drops → unknown
MIN_DEPTH_M = 0.3        # too close, depth unreliable

SCORE_TRAVERSABLE = 1.0
SCORE_UNKNOWN = 0.4
SCORE_OBSTACLE = 0.0


def estimate_ground_y(points_camera: np.ndarray, image_bottom_frac: float = 0.33) -> float | None:
    """Estimate ground y-coordinate (meters, camera frame, y-down).

    Samples points from the bottom `image_bottom_frac` of the image where
    ground is most likely visible. Uses a robust high-percentile of y
    (ground is the **lowest** physical plane, so **largest** y in y-down).

    Returns None if not enough valid samples.
    """
    if points_camera is None or points_camera.ndim != 3 or points_camera.shape[2] != 3:
        return None
    H, W, _ = points_camera.shape
    y_start = int(H * (1.0 - image_bottom_frac))
    strip = points_camera[y_start:, :, :]         # (Hs, W, 3)
    y = strip[..., 1]
    z = strip[..., 2]
    # valid mask: finite + reasonable depth range
    valid = np.isfinite(y) & np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    if valid.sum() < 100:
        return None
    y_valid = y[valid]
    # ground is largest y (y-down). Take 75th percentile to avoid outliers
    # (rare points *below* ground in noisy VGGT output) but still robust to
    # table/seat surfaces (smaller y).
    return float(np.percentile(y_valid, 75))


def compute_traversability_map(points_camera: np.ndarray,
                                y_ground: float | None = None) -> np.ndarray:
    """Build per-pixel traversability score map.

    If y_ground is None, auto-estimate from the bottom of the image.

    Returns (H,W) float32 in [0, 1].
    """
    if points_camera is None or points_camera.ndim != 3 or points_camera.shape[2] != 3:
        raise ValueError(f"points_camera must be HxWx3, got shape {None if points_camera is None else points_camera.shape}")
    H, W, _ = points_camera.shape
    if y_ground is None:
        y_ground = estimate_ground_y(points_camera)
    if y_ground is None:
        # Cannot estimate ground → return unknown everywhere
        return np.full((H, W), SCORE_UNKNOWN, dtype=np.float32)

    y = points_camera[..., 1]
    z = points_camera[..., 2]
    valid_depth = np.isfinite(y) & np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)

    dy = y - y_ground  # positive = below ground (physically lower)
                       # negative = above ground (obstacle)
    # Traversable: within ground band (approximately on ground plane)
    is_ground = valid_depth & (np.abs(dy) <= GROUND_BAND_M)
    # Obstacle: significantly above ground (y smaller = higher in y-down)
    is_obstacle = valid_depth & (dy < -OBSTACLE_ABOVE_M)

    trav = np.full((H, W), SCORE_UNKNOWN, dtype=np.float32)
    trav[is_obstacle] = SCORE_OBSTACLE
    trav[is_ground] = SCORE_TRAVERSABLE
    return trav


def validate_point(trav_map: np.ndarray, cx: int, cy: int,
                   radius: int | None = None,
                   min_traversable_frac: float = 0.55,
                   max_obstacle_frac: float = 0.25) -> bool:
    """Check if (cx, cy) lies in a traversable crop region.

    Samples a square window of side ~2*radius around (cx,cy).
    Passes if BOTH:
      - traversable fraction ≥ min_traversable_frac (enough ground visible)
      - obstacle fraction ≤ max_obstacle_frac (no dominant wall/pillar/person)

    radius defaults to ~10% of trav_map width (big crop proxy at VGGT scale).
    """
    H, W = trav_map.shape[:2]
    if radius is None:
        radius = max(12, int(W * 0.10))
    x0 = max(0, cx - radius); x1 = min(W, cx + radius + 1)
    y0 = max(0, cy - radius); y1 = min(H, cy + radius + 1)
    patch = trav_map[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    traversable_frac = float((patch >= SCORE_TRAVERSABLE - 1e-3).sum()) / patch.size
    obstacle_frac = float((patch <= SCORE_OBSTACLE + 1e-3).sum()) / patch.size
    return (traversable_frac >= min_traversable_frac
            and obstacle_frac <= max_obstacle_frac)


def detect_vertical_obstacle_columns(points_camera: np.ndarray,
                                      y_ground: float | None = None,
                                      min_col_coverage: float = 0.15,
                                      bottom_frac: float = 0.5) -> np.ndarray | None:
    """Per-column mask of vertical obstacles (pillars, walls, chair backs).

    A column is flagged if a significant fraction of pixels in the BOTTOM half
    of the image sit meaningfully above the ground plane. Limiting to the
    bottom half excludes ceilings (indoor scenes): a ceiling at 2.5m fills the
    top half of every column with above-ground pixels, which would otherwise
    flag every column as an obstacle. Pillars/walls/chairs still qualify
    because their base reaches down into the bottom half.
    """
    if points_camera is None or points_camera.ndim != 3 or points_camera.shape[2] != 3:
        return None
    H, W, _ = points_camera.shape
    if y_ground is None:
        y_ground = estimate_ground_y(points_camera)
    if y_ground is None:
        return None
    row_start = int(H * (1.0 - bottom_frac))
    y = points_camera[row_start:, :, 1]
    z = points_camera[row_start:, :, 2]
    valid = np.isfinite(y) & np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    above = valid & ((y_ground - y) > OBSTACLE_ABOVE_M)
    col_coverage = above.sum(axis=0) / (H - row_start)
    return col_coverage >= min_col_coverage


def resolve_crop_point(trav_map: np.ndarray,
                       preferred_cx: int,
                       target_y_frac: float = 0.48,
                       edge_margin_frac: float = 0.10,
                       max_cx_offset_frac: float = 0.30,
                       points_camera: np.ndarray | None = None,
                       y_ground: float | None = None) -> tuple[int, int] | None:
    """Pointer-preserve crop point resolver.

    与 find_best_traversable_point 不同, 此函数优先保留 pointer 的 cx 信号,
    TRAV 只做:
      1. cy 强制固定到 target_y_frac 行 (地面带)
      2. cx 仅在落边缘或柱子列时 veto, 找最近可通行列替代
      3. 所有候选列都不通时返回 None

    目的: 让不同 pointer (qwen/gdino/geom/molmo/gsam2) 的 cx 差异真正体现在最终
    crop, 避免 find_best_traversable_point 的"widest segment center"把所有 pointer
    结果拉到同样几个热点列 (上一轮对比发现三方 cx 被离散到 577/664/730/1378 等).
    """
    H, W = trav_map.shape[:2]
    row = int(H * target_y_frac)
    edge_px = int(W * edge_margin_frac)
    max_cx_offset = int(W * max_cx_offset_frac)
    col_obstacle = None
    if points_camera is not None:
        col_obstacle = detect_vertical_obstacle_columns(points_camera, y_ground)
        if col_obstacle is not None and col_obstacle.shape[0] != W:
            col_obstacle = None

    def _cx_usable(cx: int) -> bool:
        if cx < edge_px or cx >= W - edge_px:
            return False
        if col_obstacle is not None and col_obstacle[cx]:
            return False
        return True

    if _cx_usable(preferred_cx):
        return preferred_cx, row

    # veto triggered: search nearest usable cx within max_cx_offset
    lo = max(edge_px, preferred_cx - max_cx_offset)
    hi = min(W - edge_px, preferred_cx + max_cx_offset + 1)
    candidates = [cx for cx in range(lo, hi) if _cx_usable(cx)]
    if not candidates:
        return None
    best_cx = min(candidates, key=lambda c: abs(c - preferred_cx))
    return best_cx, row


def find_best_traversable_point(trav_map: np.ndarray,
                                 preferred_cx: int | None = None,
                                 target_y_frac: float = 0.48,
                                 x_search_band: int | None = None,
                                 edge_margin_frac: float = 0.10,
                                 max_offset_frac: float = 0.30,
                                 points_camera: np.ndarray | None = None,
                                 y_ground: float | None = None) -> tuple[int, int] | None:
    """Pick the best traversable (cx, cy) on the map.

    Strategy:
      - target_y_frac: row height hint (matches auto_sub_image_extractor.TARGET_Y_PCT)
      - preferred_cx: if given, try to stay near this column (Qwen hint)
      - edge_margin_frac: ignore left/right edge columns (VGGT panorama border
        noise)
      - max_offset_frac: reject replacement that moves cx by > this*W from
        preferred_cx (safer than far-off edge artifact)
      - points_camera: if provided, columns covered by vertical obstacles
        (pillars, walls) are masked out even when the target-row pixel alone
        tests as traversable — fixes the pillar-column false positive.
      - Returns None if no safe traversable pixel found; caller keeps qwen.
    """
    H, W = trav_map.shape[:2]
    row = int(H * target_y_frac)
    search = trav_map[max(0, row - 10):min(H, row + 11), :]
    if search.size == 0:
        return None
    col_score = search.max(axis=0)

    edge_px = int(W * edge_margin_frac)
    if edge_px > 0:
        col_score[:edge_px] = SCORE_OBSTACLE
        col_score[W - edge_px:] = SCORE_OBSTACLE

    if points_camera is not None:
        col_obstacle = detect_vertical_obstacle_columns(points_camera, y_ground)
        if col_obstacle is not None and col_obstacle.shape[0] == W:
            col_score[col_obstacle] = SCORE_OBSTACLE

    traversable_cols = np.where(col_score >= SCORE_TRAVERSABLE - 1e-3)[0]
    if traversable_cols.size == 0:
        return None

    max_offset = int(W * max_offset_frac) if preferred_cx is not None else W

    # Strategy 1: nearest traversable column within preferred_cx ± x_search_band
    if preferred_cx is not None and x_search_band:
        lo = max(edge_px, preferred_cx - x_search_band)
        hi = min(W - edge_px, preferred_cx + x_search_band + 1)
        in_band = traversable_cols[(traversable_cols >= lo) & (traversable_cols < hi)]
        if in_band.size > 0:
            cx = int(in_band[np.argmin(np.abs(in_band - preferred_cx))])
            return cx, row

    # Strategy 2: widest contiguous segment center, bounded by max_offset
    gaps = np.diff(traversable_cols)
    segments = []
    seg_start_idx = 0
    for i, g in enumerate(gaps):
        if g > 5:
            segments.append((seg_start_idx, i))
            seg_start_idx = i + 1
    segments.append((seg_start_idx, len(traversable_cols) - 1))
    segments.sort(key=lambda s: -(s[1] - s[0]))  # widest first

    for s_lo, s_hi in segments:
        cand_cx = int(traversable_cols[(s_lo + s_hi) // 2])
        if preferred_cx is None or abs(cand_cx - preferred_cx) <= max_offset:
            return cand_cx, row

    # All segments too far from preferred_cx → caller should keep qwen
    return None
