#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉遮挡检测器 (YOLOv8n)

用于 MemoryNav 导航过程中检测相机画面是否被遮挡（行人、手、大面积前景物体等）。
当 VPR 匹配失败 + 子图匹配失败时，判断是否因为视觉遮挡导致，
若是遮挡则让机器人原地等待，直到遮挡消除后继续导航。

核心逻辑:
1. 使用 YOLOv8n 检测画面中的前景物体
2. 计算遮挡类别物体的 bbox 面积占画面总面积的比例
3. 面积比超过阈值 → 判定为遮挡
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 默认模型路径
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained", "yolov8n.pt"
)

# ============================================================================
# 遮挡检测结果
# ============================================================================

@dataclass
class OcclusionResult:
    """遮挡检测结果"""
    occluded: bool                     # 是否判定为遮挡
    max_area_ratio: float = 0.0        # 最大单个遮挡物的面积占比
    total_area_ratio: float = 0.0      # 所有遮挡物的总面积占比（含重叠）
    detections: List[Dict] = field(default_factory=list)  # 检测到的遮挡物列表
    elapsed_ms: float = 0.0            # 推理耗时 (ms)
    camera_name: str = ""              # 检测的相机名称
    reason: str = ""                   # 判定理由

    def to_dict(self) -> Dict:
        return {
            'occluded': self.occluded,
            'max_area_ratio': round(self.max_area_ratio, 4),
            'total_area_ratio': round(self.total_area_ratio, 4),
            'detections': self.detections,
            'elapsed_ms': round(self.elapsed_ms, 1),
            'camera_name': self.camera_name,
            'reason': self.reason,
        }


# ============================================================================
# COCO 类别中可能造成相机遮挡的物体
# ============================================================================

# COCO 80类 → 可能在近距离遮挡机器人相机的类别
OCCLUDING_CLASS_IDS = {
    0,   # person — 最常见的遮挡源
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    28,  # suitcase
    # 可根据实际场景扩展
}

# COCO 类别名称映射 (只列出遮挡相关的)
OCCLUDING_CLASS_NAMES = {
    0: 'person',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    28: 'suitcase',
}


# ============================================================================
# 遮挡检测器
# ============================================================================

class OcclusionDetector:
    """视觉遮挡检测器 (YOLOv8n)

    使用 YOLOv8n 检测画面中的近距离前景物体，
    当遮挡类物体的 bbox 面积占比超过阈值时判定为遮挡。
    """

    def __init__(self,
                 device: str = "cuda:0",
                 area_threshold: float = 0.35,
                 confidence_threshold: float = 0.4,
                 model_name: str = None):
        """
        Args:
            device: 推理设备
            area_threshold: 单个遮挡物 bbox 面积占画面比例 >= 此值判定为遮挡
            confidence_threshold: YOLO 检测置信度阈值
            model_name: YOLO 模型名称 (首次使用会自动下载 ~6MB)
        """
        self._device = device
        self.area_threshold = area_threshold
        self.confidence_threshold = confidence_threshold
        self._model_name = model_name or _DEFAULT_MODEL_PATH
        self._model = None

    def preload(self):
        """预加载 YOLO 模型"""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            logger.info(f"[OcclusionDetector] 加载 {self._model_name} (device={self._device}) ...")
            self._model = YOLO(self._model_name)
            # 预热推理 (用小图)
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model.predict(dummy, device=self._device, verbose=False)
            logger.info(f"[OcclusionDetector] 模型加载完成")
        except Exception as e:
            logger.error(f"[OcclusionDetector] 模型加载失败: {e}")
            raise

    def detect(self, image: np.ndarray, camera_name: str = "") -> OcclusionResult:
        """对单张相机图执行遮挡检测

        Args:
            image: 相机图像 (BGR, H×W×3)
            camera_name: 相机名称 (用于日志)

        Returns:
            OcclusionResult
        """
        t0 = time.perf_counter()

        # 确保模型已加载
        self.preload()

        h, w = image.shape[:2]
        total_area = h * w

        # YOLO 推理
        results = self._model.predict(
            image,
            device=self._device,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        max_area_ratio = 0.0
        total_occluding_area = 0.0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                    # 只关注遮挡类别
                    if cls_id not in OCCLUDING_CLASS_IDS:
                        continue

                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_ratio = bbox_area / total_area

                    det = {
                        'class_id': cls_id,
                        'class_name': OCCLUDING_CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                        'confidence': round(conf, 4),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area_ratio': round(area_ratio, 4),
                    }
                    detections.append(det)

                    if area_ratio > max_area_ratio:
                        max_area_ratio = area_ratio
                    total_occluding_area += bbox_area

        total_area_ratio = total_occluding_area / total_area if total_area > 0 else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 判定遮挡: 任何一个遮挡物面积占比 >= 阈值
        occluded = bool(max_area_ratio >= self.area_threshold)

        if occluded:
            # 找出最大的遮挡物
            largest = max(detections, key=lambda d: d['area_ratio']) if detections else {}
            reason = (f"{largest.get('class_name', '?')} 占画面 {max_area_ratio*100:.1f}% "
                      f"(>= {self.area_threshold*100:.0f}% 阈值)")
        else:
            if detections:
                reason = f"检测到 {len(detections)} 个前景物体，但面积占比均低于阈值"
            else:
                reason = "未检测到遮挡类物体"

        result = OcclusionResult(
            occluded=occluded,
            max_area_ratio=max_area_ratio,
            total_area_ratio=total_area_ratio,
            detections=detections,
            elapsed_ms=elapsed_ms,
            camera_name=camera_name,
            reason=reason,
        )

        log_fn = logger.warning if occluded else logger.info
        log_fn(f"🔍 [OcclusionDetector] camera={camera_name}: "
               f"occluded={occluded}, max_area={max_area_ratio:.4f}, "
               f"detections={len(detections)}, {reason} ({elapsed_ms:.1f}ms)")

        return result

    def detect_with_visualization(self, image: np.ndarray,
                                   camera_name: str = "") -> Tuple[OcclusionResult, np.ndarray]:
        """遮挡检测 + 可视化标注

        Returns:
            (OcclusionResult, annotated_image)
        """
        result = self.detect(image, camera_name)

        # 在图像上画检测框
        vis_img = image.copy()
        h, w = vis_img.shape[:2]

        for det in result.detections:
            x1, y1, x2, y2 = det['bbox']
            area_pct = det['area_ratio'] * 100
            is_occluding = det['area_ratio'] >= self.area_threshold

            # 遮挡物: 红色; 非遮挡: 黄色
            color = (0, 0, 255) if is_occluding else (0, 200, 255)
            thickness = 3 if is_occluding else 2

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)

            # 标签
            label = f"{det['class_name']} {det['confidence']:.2f} ({area_pct:.1f}%)"
            font_scale = 0.6
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(vis_img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis_img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # 顶部状态栏
        status_color = (0, 0, 200) if result.occluded else (0, 160, 0)
        status_text = f"OCCLUDED ({result.reason})" if result.occluded else f"CLEAR ({result.reason})"
        cv2.rectangle(vis_img, (0, 0), (w, 36), status_color, -1)
        cv2.putText(vis_img, status_text, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result, vis_img
