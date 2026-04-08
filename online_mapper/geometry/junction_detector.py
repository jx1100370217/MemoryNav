"""JunctionDetector — multi-camera depth-based junction classifier.

For each of 4 cameras around the robot, takes the central column median
depth. A camera is "open" if its median depth exceeds OPEN_DEPTH_THRESH.

Mapping (4 cameras = 4 cardinal directions):
  4 open  -> CROSS
  3 open  -> T_JUNCTION
  2 open with opposite cameras -> CORRIDOR
  2 open with adjacent cameras -> CROSS-ish (tagged as T)
  <=1 open -> DEAD_END

Cheap: 4 depth inferences per call. We call this only when a frame is a
keyframe candidate, not every frame.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from online_mapper.semantics.node_category import JunctionKind

logger = logging.getLogger(__name__)


@dataclass
class JunctionInfo:
    kind: JunctionKind
    open_cams: list   # ['camera_1', 'camera_3', ...]
    median_depths: Dict[str, float]
    n_open: int


class JunctionDetector:
    OPEN_DEPTH_THRESH = 1.8   # meters (relative-depth × scale)
    CENTER_W_FRAC = 0.30
    CENTER_H_FRAC = 0.30

    # Camera id -> assumed yaw offset (radians) from robot forward.
    # camera_1 = front, camera_2 = right, camera_3 = rear, camera_4 = left
    CAMERA_YAW = {
        'camera_1': 0.0,
        'camera_2': -1.5708,  # -90 deg
        'camera_3':  3.1416,
        'camera_4':  1.5708,
    }

    def __init__(self, depth_estimator):
        self.depth = depth_estimator
        self.stats = {"calls": 0, "kinds": {}}

    def _central_median(self, depth_map: np.ndarray) -> float:
        if depth_map is None or depth_map.size == 0:
            return 0.0
        h, w = depth_map.shape[:2]
        ww = max(1, int(w * self.CENTER_W_FRAC))
        hh = max(1, int(h * self.CENTER_H_FRAC))
        x0 = (w - ww) // 2
        y0 = (h - hh) // 2
        roi = depth_map[y0:y0 + hh, x0:x0 + ww]
        roi = roi[(roi > 0.05) & (roi < 60.0)]
        if roi.size < 50:
            return 0.0
        return float(np.median(roi))

    def classify(self, frame_cameras: Dict[str, str]) -> JunctionInfo:
        """frame_cameras: {camera_id: image_path}"""
        if self.depth is None or not self.depth.available:
            return JunctionInfo(JunctionKind.UNKNOWN, [], {}, 0)
        self.stats["calls"] += 1
        medians: Dict[str, float] = {}
        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            p = frame_cameras.get(cam_id)
            if not p:
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            try:
                d = self.depth.estimate_stateless(img)
            except Exception as e:
                logger.debug(f"junction depth fail {cam_id}: {e}")
                continue
            medians[cam_id] = self._central_median(d)
        opens = [c for c, m in medians.items() if m >= self.OPEN_DEPTH_THRESH]
        n_open = len(opens)
        if n_open >= 4:
            kind = JunctionKind.CROSS
        elif n_open == 3:
            kind = JunctionKind.T_JUNCTION
        elif n_open == 2:
            # opposite cameras (1+3 or 2+4) -> straight corridor
            o = set(opens)
            if o == {'camera_1', 'camera_3'} or o == {'camera_2', 'camera_4'}:
                kind = JunctionKind.CORRIDOR
            else:
                kind = JunctionKind.T_JUNCTION  # adjacent opens = corner
        elif n_open == 1:
            kind = JunctionKind.DEAD_END
        else:
            kind = JunctionKind.DEAD_END
        self.stats["kinds"][kind.value] = self.stats["kinds"].get(kind.value, 0) + 1
        return JunctionInfo(kind, opens, medians, n_open)
