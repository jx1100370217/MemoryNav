"""Traversability map from VGGT point cloud (P3 of baseline-optim).

Input: points_camera (H,W,3) np.float32 in camera frame
       (x right, y down, z forward; meters).
Output: traversability score map (H,W) 0-1:
   1.0 = ground plane (可通行地面)
   0.0 = obstacle (墙/柱/椅/人/高物)
  ~0.5 = unknown (深度缺失 / 太远)

Used by ConnectionBuilder to validate / replace Qwen point-grounding.
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Thresholds tuned for MemoryNav (1.2-1.5m camera height, indoor+outdoor):
GROUND_BAND_M = 0.30   # |Δy - y_ground| ≤ GROUND_BAND → traversable
OBSTACLE_ABOVE_M = 0.25  # y < y_ground - 0.25 (higher than ground by 0.25m) → obstacle
MAX_DEPTH_M = 12.0     # z > MAX_DEPTH → unknown (reliability drops)
MIN_DEPTH_M = 0.3      # z < MIN_DEPTH → too close, unreliable

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


def find_best_traversable_point(trav_map: np.ndarray,
                                 preferred_cx: int | None = None,
                                 target_y_frac: float = 0.48,
                                 x_search_band: int | None = None,
                                 edge_margin_frac: float = 0.10,
                                 max_offset_frac: float = 0.30) -> tuple[int, int] | None:
    """Pick the best traversable (cx, cy) on the map.

    Strategy:
      - target_y_frac: row height hint (matches auto_sub_image_extractor.TARGET_Y_PCT)
      - preferred_cx: if given, try to stay near this column (Qwen hint)
      - edge_margin_frac: ignore left/right edge columns (avoid VGGT noise
        at panorama borders where cx=0 traversable illusions came from)
      - max_offset_frac: reject replacement that moves cx by > this*W from
        preferred_cx (safer than far-off edge artifact)
      - Returns None if no safe traversable pixel found; caller keeps qwen.
    """
    H, W = trav_map.shape[:2]
    row = int(H * target_y_frac)
    search = trav_map[max(0, row - 10):min(H, row + 11), :]
    if search.size == 0:
        return None
    col_score = search.max(axis=0)

    # Hard-mask edge columns: these are where VGGT panorama borders produce
    # spurious "traversable" noise (observed in P3 round 1: cx=0 replacements)
    edge_px = int(W * edge_margin_frac)
    if edge_px > 0:
        col_score[:edge_px] = SCORE_OBSTACLE
        col_score[W - edge_px:] = SCORE_OBSTACLE

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
