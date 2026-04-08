"""开放集检测 — Grounding-DINO + 可选 Qwen 命名"""
import logging, numpy as np
logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "door plate", "room number sign", "printer", "trash can",
    "white chair", "stool", "elevator", "fire extinguisher",
    "potted plant", "vending machine", "sofa", "table", "monitor"
]


class OpenSetDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gd_available = False
        self._processor = None
        self._model = None
        if cfg.enable_grounding_dino:
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                import torch
                self._processor = AutoProcessor.from_pretrained(cfg.grounding_dino_model_id)
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    cfg.grounding_dino_model_id).to(cfg.depth_device).eval()
                self.gd_available = True
                logger.info(f"GroundingDINO loaded {cfg.grounding_dino_model_id}")
            except Exception as e:
                logger.warning(f"GroundingDINO unavailable: {e}")

    def detect(self, bgr_image, queries=None) -> list:
        if not self.gd_available:
            return []
        try:
            from PIL import Image
            import cv2, torch
            queries = queries or DEFAULT_QUERIES
            text = ". ".join(queries) + "."
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self.cfg.depth_device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.30, text_threshold=0.25,
                target_sizes=[pil.size[::-1]],
            )[0]
            dets = []
            H, W = bgr_image.shape[:2]
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                dets.append({
                    "label": str(label),
                    "score": float(score),
                    "bbox": [x1, y1, x2, y2],
                    "norm_bbox": [x1 / W, y1 / H, x2 / W, y2 / H],
                    "area": (x2 - x1) * (y2 - y1),
                })
            return dets
        except Exception as e:
            logger.warning(f"GD detect failed: {e}")
            return []
