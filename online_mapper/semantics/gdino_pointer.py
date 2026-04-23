"""Grounding-DINO 的 target 方向打点器 — Qwen35PointGrounder 的替换方案.

提供与 Qwen35PointGrounder 对齐的 predict(image, prompt) API, 内部用
GroundingDINO 对"可通行走廊/过道/地面/路径"类 text query 做检测, 选最大
bbox 作为方向锚点. cx = bbox 中心, cy = bbox 从上起 60% 位置 (偏下靠
近地面). 设计意图: 在 Qwen 对大堂/广场等开放场景整体失锚 (4 cam 都
返回 cx≈0.5) 时, GD 的几何 grounding 更稳定.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


_WALKABLE_QUERIES: List[str] = [
    "walkway", "corridor", "passage", "hallway",
    "open ground", "floor", "path",
]


class GDinoPointer:
    def __init__(self, open_set_detector=None, cfg=None,
                 queries: Optional[List[str]] = None,
                 min_top_y_frac: float = 0.25):
        if open_set_detector is not None:
            self._gd = open_set_detector
        else:
            from online_mapper.semantics.open_set_detector import OpenSetDetector
            if cfg is None:
                raise ValueError("cfg is required when open_set_detector is None")
            self._gd = OpenSetDetector(cfg)
        self._queries = queries or _WALKABLE_QUERIES
        self._min_top_y_frac = min_top_y_frac
        logger.info(f"[GDinoPointer] 就绪, queries={self._queries}, "
                    f"min_top_y_frac={self._min_top_y_frac}")

    def start(self):
        # No-op: OpenSetDetector 在构造时已懒加载.
        pass

    def stop(self):
        pass

    def is_ready(self) -> bool:
        return getattr(self._gd, "gd_available", False)

    def predict(self, image: np.ndarray, landmark_name: str = "") -> Dict:
        if not self.is_ready():
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "GD not available"}
        queries = list(self._queries)
        if landmark_name and landmark_name.strip() and landmark_name not in queries:
            queries.append(landmark_name.strip())
        try:
            dets = self._gd.detect(image, queries=queries)
        except Exception as e:
            return {"success": False, "point": None, "confidence": 0.0, "error": str(e)}
        if not dets:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "no detections"}
        H, W = image.shape[:2]
        y_cut = H * self._min_top_y_frac
        candidates = [d for d in dets if d["bbox"][1] >= y_cut]
        if not candidates:
            candidates = dets
        best = max(candidates, key=lambda d: d.get("area", 0.0))
        x1, y1, x2, y2 = best["bbox"]
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + (y2 - y1) * 0.60) / H
        return {
            "success": True,
            "point": (cx, cy),
            "confidence": float(best.get("score", 0.0)),
            "label": best.get("label", ""),
            "bbox_norm": [x1 / W, y1 / H, x2 / W, y2 / H],
        }
