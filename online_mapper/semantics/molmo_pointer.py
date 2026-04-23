"""Molmo pointing pointer: 用 AllenAI Molmo-7B 原生 pointing API 直接产出像素点.

Molmo 原生支持 "Point to <target>" prompt, 输出 <point x="pct" y="pct">name</point>
格式 (x,y 是 0-100 百分比). 相比 Qwen VLM 推理方式, Molmo 专门训练过 pointing
任务, 对开放词表空间定位更稳定.

模型: allenai/Molmo-7B-D-0924 (~14GB, bf16).
"""
import logging
import os
import re
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


_POINT_RE = re.compile(r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"[^>]*>([^<]*)</point>',
                        re.IGNORECASE)

_LOCAL_DEFAULT = "pretrained/Molmo-7B-D-0924"
_HF_DEFAULT = "allenai/Molmo-7B-D-0924"


def _resolve_model_id(model_id: Optional[str]) -> str:
    if model_id:
        return model_id
    if os.path.isdir(_LOCAL_DEFAULT):
        return _LOCAL_DEFAULT
    return _HF_DEFAULT


class MolmoPointer:
    def __init__(self, model_id: Optional[str] = None,
                 device: str = "cuda:0", dtype: str = "bf16",
                 prompt: str = "Point to the walkable path ahead.",
                 **kwargs):
        model_id = _resolve_model_id(model_id)
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._prompt = prompt
        self._processor = None
        self._model = None
        logger.info(f"[MolmoPointer] 懒加载模式 (model_id={model_id}, "
                    f"device={device}, dtype={dtype})")

    def _load(self):
        if self._model is not None or getattr(self, "_load_failed", False):
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
            td = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[self._dtype]
            logger.info(f"[MolmoPointer] 加载 {self._model_id} ...")
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, trust_remote_code=True, torch_dtype=td)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, trust_remote_code=True, torch_dtype=td).to(self._device).eval()
            self._gen_cfg = GenerationConfig(
                max_new_tokens=128, stop_strings="<|endoftext|>")
            logger.info(f"[MolmoPointer] 就绪")
        except Exception as e:
            self._load_failed = True
            logger.error(f"[MolmoPointer] 加载失败 (后续 predict 静默返回 fail): {e}")

    def start(self):
        try:
            self._load()
        except Exception as e:
            logger.error(f"[MolmoPointer] 加载失败: {e}")

    def stop(self):
        pass

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, image: np.ndarray, landmark_name: str = "") -> dict:
        self._load()
        if self._model is None:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": "model not loaded"}
        try:
            import torch
            if image.ndim == 3 and image.shape[2] == 3:
                import cv2
                pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil = Image.fromarray(image)
            inputs = self._processor.process(images=[pil], text=self._prompt)
            inputs = {k: (v.to(self._device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                       for k, v in inputs.items()}
            with torch.no_grad():
                output = self._model.generate_from_batch(
                    inputs, self._gen_cfg, tokenizer=self._processor.tokenizer)
            gen_tokens = output[0, inputs["input_ids"].size(1):]
            text = self._processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        except Exception as e:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": f"molmo inference failed: {e}"}

        m = _POINT_RE.search(text)
        if not m:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": f"no point tag in output: {text[:100]}"}
        try:
            x_pct = float(m.group(1))
            y_pct = float(m.group(2))
        except ValueError as e:
            return {"success": False, "point": None, "confidence": 0.0,
                    "error": f"parse failed: {e}, text={text[:100]}"}
        cx_norm = x_pct / 100.0
        cy_norm = y_pct / 100.0
        return {"success": True,
                "point": (cx_norm, cy_norm),
                "confidence": 1.0,
                "method": "molmo",
                "label": m.group(3) if m.group(3) else ""}
