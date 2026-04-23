#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人行横道红绿灯检测器 (YOLO11n + HSV 颜色判别 + 时序平滑)

检测 camera_1/camera_2 画面中的人行横道红绿灯，判断红/绿状态。
红灯时让机器人原地等待，绿灯或无灯则正常导航。

颜色判别: bbox 内两段式 (上红下绿) HSV 分析，适配人行横道灯。
时序平滑: 防止闪烁导致的误判 —— 红灯立即生效，绿灯需连续确认。
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained", "yolo11n.pt"
)

TRAFFIC_LIGHT_CLASS_ID = 9

_DEFAULT_AZIMUTHS = {
    'camera_1': 39.42,
    'camera_2': -35.84,
    'camera_3': -142.04,
    'camera_4': 143.52,
}
_DEFAULT_FOV = 180


@dataclass
class TrafficLightResult:
    """单帧红绿灯检测结果（原始，未经时序平滑）"""
    detected: bool = False
    color: str = "none"          # "red" / "green" / "none"
    confidence: float = 0.0
    bbox: Optional[List[int]] = None
    elapsed_ms: float = 0.0
    camera_name: str = ""
    global_angle_deg: float = 0.0  # 红绿灯在机器人坐标系下的全局角度（度，0=正前方）

    def to_dict(self) -> Dict:
        return {
            "detected": self.detected,
            "color": self.color,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "camera_name": self.camera_name,
            "global_angle_deg": round(self.global_angle_deg, 1),
        }


class TrafficLightDetector:
    """人行横道红绿灯检测器

    YOLO11n 检测 traffic light bbox → HSV 两段式颜色判别 → 时序平滑状态机
    """

    def __init__(self,
                 device: str = "cuda:0",
                 confidence_threshold: float = 0.25,
                 model_path: str = None,
                 no_detect_clear_frames: int = 5,
                 max_angle_deg: float = 30.0,
                 max_bbox_height: int = 120):
        """
        Args:
            max_angle_deg: 只接受全局角度在 [-max_angle_deg, +max_angle_deg] 范围内
                           的红绿灯（0°=正前方），过滤侧面/后方的无关灯
            max_bbox_height: bbox 高度上限(像素)，超过视为近处灯忽略
        """
        self._device = device
        self.confidence_threshold = confidence_threshold
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._model = None
        self._max_angle_deg = max_angle_deg
        self._max_bbox_height = max_bbox_height

        self._no_detect_clear_frames = no_detect_clear_frames

        self._last_color: str = "none"
        self._no_detect_counter: int = 0

    def preload(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            logger.info(f"[TrafficLightDetector] 加载 {self._model_path} (device={self._device}) ...")
            self._model = YOLO(self._model_path)
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model.predict(dummy, device=self._device, verbose=False)
            logger.info("[TrafficLightDetector] 模型加载完成")
        except Exception as e:
            logger.error(f"[TrafficLightDetector] 模型加载失败: {e}")
            raise

    def reset_state(self):
        """重置时序平滑状态（新导航任务时调用）"""
        self._last_color = "none"
        self._no_detect_counter = 0

    # ------------------------------------------------------------------
    # 单帧检测
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_global_angle(bbox: List[int], img_width: int, camera_name: str) -> float:
        """计算 bbox 中心在机器人坐标系下的全局角度（度，0°=正前方）"""
        cx = (bbox[0] + bbox[2]) / 2.0
        x_norm = cx / img_width
        azimuth_deg = _DEFAULT_AZIMUTHS.get(camera_name, 0.0)
        theta_h_deg = (0.5 - x_norm) * _DEFAULT_FOV
        return azimuth_deg + theta_h_deg

    def detect_all(self, image: np.ndarray, camera_name: str = "") -> List[TrafficLightResult]:
        """对单张图检测所有红绿灯 bbox，返回全部结果列表"""
        t0 = time.perf_counter()
        self.preload()

        results = self._model.predict(
            image,
            device=self._device,
            conf=self.confidence_threshold,
            verbose=False,
        )

        all_results: List[TrafficLightResult] = []
        img_w = image.shape[1]

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id != TRAFFIC_LIGHT_CLASS_ID:
                        continue
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    global_angle = self._bbox_global_angle(bbox, img_w, camera_name)
                    color = self._classify_color(image, bbox)
                    all_results.append(TrafficLightResult(
                        detected=True, color=color, confidence=conf,
                        bbox=bbox, camera_name=camera_name,
                        global_angle_deg=global_angle,
                    ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in all_results:
            r.elapsed_ms = elapsed_ms

        return all_results

    def _is_valid_detection(self, r: TrafficLightResult) -> bool:
        """检查检测结果是否通过角度和尺寸过滤"""
        if abs(r.global_angle_deg) > self._max_angle_deg:
            return False
        if r.bbox and (r.bbox[3] - r.bbox[1]) > self._max_bbox_height:
            return False
        return True

    def detect(self, image: np.ndarray, camera_name: str = "") -> TrafficLightResult:
        """对单张图检测红绿灯，返回通过角度+尺寸过滤后置信度最高的结果"""
        t0 = time.perf_counter()
        all_results = self.detect_all(image, camera_name)
        elapsed = (time.perf_counter() - t0) * 1000
        if not all_results:
            return TrafficLightResult(detected=False, camera_name=camera_name, elapsed_ms=elapsed)

        candidates = [r for r in all_results if self._is_valid_detection(r)]
        if not candidates:
            return TrafficLightResult(detected=False, camera_name=camera_name, elapsed_ms=elapsed)
        best = max(candidates, key=lambda r: r.confidence)
        best.elapsed_ms = elapsed
        return best

    def detect_multi(self,
                     camera_images: Dict[str, np.ndarray],
                     cam_ids: List[str]) -> TrafficLightResult:
        """对多路相机检测，过滤角度/尺寸不符合的，返回置信度最高的结果"""
        best: Optional[TrafficLightResult] = None
        total_ms = 0.0

        for cam_id in cam_ids:
            img = camera_images.get(cam_id)
            if img is None:
                continue
            r = self.detect(img, cam_id)
            total_ms += r.elapsed_ms
            if r.detected:
                if best is None or r.confidence > best.confidence:
                    best = r

        if best is not None:
            best.elapsed_ms = total_ms
            return best

        return TrafficLightResult(
            detected=False, color="none", confidence=0.0,
            bbox=None, elapsed_ms=total_ms, camera_name="",
        )

    # ------------------------------------------------------------------
    # HSV 两段式颜色判别（人行横道灯：上红下绿）
    # ------------------------------------------------------------------

    def _classify_color(self, image: np.ndarray, bbox: List[int]) -> str:
        x1, y1, x2, y2 = bbox
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "none"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_h = y2 - y1
        h_half = roi_h // 2
        if h_half < 2:
            return "none"

        top = hsv[:h_half]
        bottom = hsv[h_half:]

        red_bright = self._count_color_pixels(top, "red")
        green_bright = self._count_color_pixels(bottom, "green")

        top_total = top.shape[0] * top.shape[1]
        bottom_total = bottom.shape[0] * bottom.shape[1]
        if top_total == 0 or bottom_total == 0:
            return "none"

        red_ratio = red_bright / top_total
        green_ratio = green_bright / bottom_total

        min_ratio = 0.05

        if red_ratio >= min_ratio and red_ratio > green_ratio:
            return "red"
        elif green_ratio >= min_ratio and green_ratio > red_ratio:
            return "green"

        return "none"

    @staticmethod
    def _count_color_pixels(hsv_region: np.ndarray, color: str) -> int:
        """统计 HSV 区域中指定颜色的高亮像素数"""
        h, s, v = cv2.split(hsv_region)

        brightness_mask = v > 150
        saturation_mask = s > 80

        if color == "red":
            hue_mask = (h <= 10) | (h >= 160)
        elif color == "green":
            hue_mask = (h >= 35) & (h <= 85)
        else:
            return 0

        mask = brightness_mask & saturation_mask & hue_mask
        return int(np.count_nonzero(mask))

    # ------------------------------------------------------------------
    # 时序平滑状态机
    # ------------------------------------------------------------------

    def update_state(self, raw: TrafficLightResult) -> str:
        """根据原始检测结果更新时序状态，返回平滑后的颜色决策

        规则:
          - 无 bbox → 连续 N 帧后清除状态
          - 有 bbox + color="none" → 保持上一次颜色
          - 有 bbox + color="red" → 立即切红
          - 有 bbox + color="green" → 立即切绿
        """
        if not raw.detected:
            self._no_detect_counter += 1
            if self._no_detect_counter >= self._no_detect_clear_frames:
                self._last_color = "none"
            return self._last_color

        self._no_detect_counter = 0

        if raw.color == "none":
            return self._last_color

        self._last_color = raw.color
        return raw.color
