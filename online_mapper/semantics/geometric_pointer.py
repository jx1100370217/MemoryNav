"""纯几何 pointer: 用 VGGT traversability map 找画面下半可通行列中位数.

零模型推理, 仅复用 depth_estimator + traversability. 失败时 (trav 不可靠,
如电梯厅反光) 显式返回 success=False, 让上层 fallback 处理.

与 Qwen / GDino pointer 并列, 提供"当 VLM 失锚时的几何兜底"选项, 或作为
对比 baseline.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GeometricPointer:
    def __init__(self, depth_estimator=None, target_y_frac: float = 0.48,
                 **kwargs):
        self._depth = depth_estimator
        self._target_y_frac = target_y_frac
        if self._depth is None:
            logger.warning("[GeometricPointer] 没有 depth_estimator, 所有 predict 将失败")
        else:
            logger.info(f"[GeometricPointer] 就绪 (target_y_frac={target_y_frac})")

    def start(self):
        pass

    def stop(self):
        pass

    def is_ready(self) -> bool:
        return self._depth is not None

    def predict(self, image: np.ndarray, landmark_name: str = "") -> dict:
        if self._depth is None:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "no depth estimator"}
        try:
            out = self._depth.estimate_stateless_with_points(image)
        except Exception as e:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": f"depth failed: {e}"}
        if out is None or "points_camera" not in out:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "no points_camera"}
        pts = out["points_camera"]
        from online_mapper.geometry import traversability as _trav
        trav = _trav.compute_traversability_map(pts)
        h_pts, w_pts = trav.shape
        best = _trav.find_best_traversable_point(
            trav, target_y_frac=self._target_y_frac,
            points_camera=pts, edge_margin_frac=0.10)
        if best is None:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "no traversable column"}
        cx_t, cy_t = best
        cx_norm = cx_t / float(w_pts)
        cy_norm = cy_t / float(h_pts)
        return {"success": True,
                "point": (cx_norm, cy_norm),
                "confidence": 1.0,
                "method": "geom"}
