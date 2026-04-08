"""单目深度估计 — Depth-Anything-V2-Small (HF cached)"""
import logging, numpy as np
logger = logging.getLogger(__name__)


class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device="cuda:0"):
        self.device = device
        self.model_id = model_id
        self._pipe = None
        self.available = False
        try:
            from transformers import pipeline
            import torch
            self._pipe = pipeline("depth-estimation", model=model_id, device=device)
            self.available = True
            logger.info(f"DepthEstimator loaded {model_id} on {device}")
        except Exception as e:
            logger.warning(f"DepthEstimator unavailable, using uniform depth fallback: {e}")

    def estimate(self, bgr_image) -> np.ndarray:
        """返回相对深度图 (HxW float)。失败时返回均匀 1.0。"""
        if not self.available:
            h, w = bgr_image.shape[:2]
            return np.ones((h, w), dtype=np.float32)
        try:
            from PIL import Image
            import cv2
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            out = self._pipe(pil)
            depth = np.array(out["depth"], dtype=np.float32)
            # 归一化到 [0.5, 5.0] 米的伪 metric 范围
            d_min, d_max = float(depth.min()), float(depth.max())
            if d_max - d_min < 1e-6:
                return np.ones_like(depth) * 1.0
            depth = (depth - d_min) / (d_max - d_min)
            depth = 0.5 + depth * 4.5
            return depth
        except Exception as e:
            logger.warning(f"Depth estimate failed: {e}")
            h, w = bgr_image.shape[:2]
            return np.ones((h, w), dtype=np.float32)
