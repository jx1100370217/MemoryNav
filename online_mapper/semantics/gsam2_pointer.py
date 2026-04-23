"""Grounded-SAM2 pointer: Grounding-DINO 找 walkable bbox -> SAM2 像素级 mask ->
mask 重心作为点.

相比纯 GDino (bbox 中心), 精度提升到像素级 — 避免 bbox 覆盖过大导致的"居中
陷阱". 用 HuggingFace transformers 内置 SAM2 支持 (4.47+), 无需额外 pip 包.

模型: facebook/sam2-hiera-large-hf + GDino (复用 OpenSetDetector).
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

_SAM2_LOCAL_DEFAULT = "pretrained/sam2-hiera-large-hf"
_SAM2_HF_DEFAULT = "facebook/sam2-hiera-large-hf"


def _resolve_sam2_id(model_id: Optional[str]) -> str:
    if model_id:
        return model_id
    if os.path.isdir(_SAM2_LOCAL_DEFAULT):
        return _SAM2_LOCAL_DEFAULT
    return _SAM2_HF_DEFAULT


class GSAM2Pointer:
    def __init__(self, open_set_detector=None, cfg=None,
                 sam2_model_id: Optional[str] = None,
                 device: str = "cuda:0",
                 queries: Optional[List[str]] = None,
                 min_top_y_frac: float = 0.25,
                 **kwargs):
        sam2_model_id = _resolve_sam2_id(sam2_model_id)
        if open_set_detector is not None:
            self._gd = open_set_detector
        else:
            from online_mapper.semantics.open_set_detector import OpenSetDetector
            if cfg is None:
                raise ValueError("cfg required when open_set_detector is None")
            self._gd = OpenSetDetector(cfg)
        self._queries = queries or _WALKABLE_QUERIES
        self._min_top_y_frac = min_top_y_frac
        self._sam2_model_id = sam2_model_id
        self._device = device
        self._sam_processor = None
        self._sam_model = None

    def _load_sam2(self):
        if self._sam_model is not None:
            return
        try:
            from transformers import Sam2Processor, Sam2Model
            logger.info(f"[GSAM2Pointer] 加载 SAM2 {self._sam2_model_id} ...")
            self._sam_processor = Sam2Processor.from_pretrained(self._sam2_model_id)
            self._sam_model = Sam2Model.from_pretrained(self._sam2_model_id).to(self._device).eval()
            logger.info("[GSAM2Pointer] SAM2 就绪")
        except Exception as e:
            logger.error(f"[GSAM2Pointer] SAM2 加载失败: {e}")

    def start(self):
        self._load_sam2()

    def stop(self):
        pass

    def is_ready(self) -> bool:
        return (getattr(self._gd, "gd_available", False) and self._sam_model is not None)

    def predict(self, image: np.ndarray, landmark_name: str = "") -> dict:
        self._load_sam2()
        if not getattr(self._gd, "gd_available", False):
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "GD not available"}
        if self._sam_model is None:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "SAM2 not available"}
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
        try:
            import torch
            import cv2
            from PIL import Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inputs = self._sam_processor(
                images=pil,
                input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._sam_model(**inputs)
            masks = self._sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())[0][0].numpy()
            mask = masks[np.argmax(outputs.iou_scores[0, 0].cpu().numpy())] \
                   if masks.ndim == 3 else masks
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                return {"success": False, "point": None, "confidence": 0.0,
                        "error": "empty mask"}
            cy_px = float(np.median(ys))
            cx_px = float(np.median(xs))
        except Exception as e:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": f"SAM2 failed: {e}"}
        return {"success": True,
                "point": (cx_px / W, cy_px / H),
                "confidence": float(best.get("score", 0.0)),
                "method": "gsam2",
                "label": best.get("label", "")}
