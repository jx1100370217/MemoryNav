"""GDino bbox 下部中位 pointer — Grounded-SAM2 的无 SAM 简化版.

真正的 Grounded-SAM2 (GDino + SAM2 mask 重心) 需要 transformers 4.53+ 或
Meta 官方 sam2 包, 当前 transformers 4.51 不支持 Sam2Processor/Sam2Model.
为保持对比实验可控 (不升级 transformers 引入兼容风险, 不装 sam2 的 hydra
等额外依赖), 此版本降级为:

GDino 找 walkable bbox → 取 bbox 面积最大且 top 不越过画面上 25% 的那个
→ cx 取 bbox 水平中线, cy 取 bbox 从顶起 83% (下 1/3 行中位) → 归一化点.

相对 GDinoPointer (cy = bbox 从顶起 60%) 更靠近地面, 近似 mask 重心效果
(walkable 区域的质心通常偏 bbox 下半). 不等同 SAM2 像素级 mask, 但零
新依赖且与 GDinoPointer 可独立并行对比.
"""
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


_WALKABLE_QUERIES: List[str] = [
    "walkway", "corridor", "passage", "hallway",
    "open ground", "floor", "path",
]


class GSAM2Pointer:
    def __init__(self, open_set_detector=None, cfg=None,
                 queries: Optional[List[str]] = None,
                 min_top_y_frac: float = 0.25,
                 bottom_ratio: float = 0.83,
                 **kwargs):
        if open_set_detector is not None:
            self._gd = open_set_detector
        else:
            from online_mapper.semantics.open_set_detector import OpenSetDetector
            if cfg is None:
                raise ValueError("cfg required when open_set_detector is None")
            self._gd = OpenSetDetector(cfg)
        self._queries = queries or _WALKABLE_QUERIES
        self._min_top_y_frac = min_top_y_frac
        self._bottom_ratio = bottom_ratio
        logger.info(f"[GSAM2Pointer] 就绪 (无 SAM 简化版, "
                    f"bottom_ratio={bottom_ratio}, "
                    f"queries={self._queries})")

    def start(self):
        pass

    def stop(self):
        pass

    def is_ready(self) -> bool:
        return getattr(self._gd, "gd_available", False)

    def predict(self, image: np.ndarray, landmark_name: str = "") -> dict:
        if not self.is_ready():
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "GD not available"}
        try:
            dets = self._gd.detect(image, queries=self._queries)
        except Exception as e:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": f"GD detect failed: {e}"}
        if not dets:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "no GD detections"}
        H, W = image.shape[:2]
        y_cut = H * self._min_top_y_frac
        candidates = [d for d in dets if d["bbox"][1] >= y_cut]
        if not candidates:
            candidates = dets
        best = max(candidates, key=lambda d: d.get("area", 0.0))
        x1, y1, x2, y2 = best["bbox"]
        cx_px = (x1 + x2) / 2.0
        cy_px = y1 + (y2 - y1) * self._bottom_ratio
        return {"success": True,
                "point": (cx_px / W, cy_px / H),
                "confidence": float(best.get("score", 0.0)),
                "method": "gsam2_simple",
                "label": best.get("label", "")}
