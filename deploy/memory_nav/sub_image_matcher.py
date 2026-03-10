#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav - 子图匹配模块

基于 SuperPoint + LightGlue 的子图定位，用于在导航过程中
在当前相机图中定位记忆中的注意力子图（crop）。

代码源自 SubImageLocator 项目，已内嵌到 MemoryNav 中，
不再依赖外部代码库。

模型生命周期：启动时加载一次，后续复用。
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================================
# 全局模型缓存（启动时加载一次，后续匹配复用）
# ============================================================================
_device = "cuda" if torch.cuda.is_available() else "cpu"
_extractor = None
_matcher = None


def _load_models():
    """加载并缓存 SuperPoint + LightGlue 模型（仅首次调用时加载）。"""
    global _extractor, _matcher
    if _extractor is not None:
        return _extractor, _matcher

    from lightglue import LightGlue, SuperPoint

    logger.info("[SubImageMatcher] 正在加载 SuperPoint ...")
    _extractor = SuperPoint(max_num_keypoints=2048).eval().to(_device)
    logger.info("[SubImageMatcher] 正在加载 LightGlue ...")
    _matcher = LightGlue(features="superpoint").eval().to(_device)
    logger.info(f"[SubImageMatcher] 模型加载完成 (device={_device})")
    return _extractor, _matcher


# ============================================================================
# 子图匹配结果
# ============================================================================

@dataclass
class SubImageMatchResult:
    """子图匹配结果

    区域描述与 SubImageLocator WebUI 一致：
    - top_left_pct:  左上角百分比 (x%, y%)
    - bottom_right_pct: 右下角百分比 (x%, y%)
    """
    found: bool                   # 是否匹配成功
    confidence: float             # 匹配置信度 [0, 1]

    # 像素坐标 bounding box
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0

    # 百分比 bounding box —— 与 WebUI 格式一致
    # 左上角 (x_min_pct, y_min_pct) / 右下角 (x_max_pct, y_max_pct)
    x_min_pct: float = 0.0       # 左上角 x%
    y_min_pct: float = 0.0       # 左上角 y%
    x_max_pct: float = 0.0       # 右下角 x%
    y_max_pct: float = 0.0       # 右下角 y%

    # 耗时 & 方法
    elapsed_ms: float = 0.0
    method: str = ""

    def to_dict(self) -> Dict:
        return {
            'found': self.found,
            'confidence': round(self.confidence, 4),
            'top_left_pct': {
                'x': round(self.x_min_pct / 100.0, 4),
                'y': round(self.y_min_pct / 100.0, 4),
            },
            'bottom_right_pct': {
                'x': round(self.x_max_pct / 100.0, 4),
                'y': round(self.y_max_pct / 100.0, 4),
            },
            'bbox_pixel': {
                'x_min': self.x_min, 'y_min': self.y_min,
                'x_max': self.x_max, 'y_max': self.y_max,
            },
            'elapsed_ms': round(self.elapsed_ms, 1),
            'method': self.method,
        }


# ============================================================================
# 底层特征匹配（源自 SubImageLocator/matchers/feature_matcher.py）
# ============================================================================

def _extract_features(extractor, image: np.ndarray) -> dict:
    """提取 SuperPoint 特征。"""
    from lightglue.utils import numpy_image_to_torch
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    tensor = numpy_image_to_torch(gray).to(_device)
    with torch.no_grad():
        feats = extractor.extract(tensor)
    return feats


def _match_features(
    image: np.ndarray,
    template: np.ndarray,
    min_matches: int = 8,
    confidence_threshold: float = 0.3,
) -> SubImageMatchResult:
    """
    使用 SuperPoint + LightGlue 在 image 中定位 template。

    返回与 SubImageLocator WebUI 一致的百分比描述。
    """
    t0 = time.perf_counter()

    extractor, matcher = _load_models()
    img_h, img_w = image.shape[:2]

    # 提取特征
    feats0 = _extract_features(extractor, image)
    feats1 = _extract_features(extractor, template)

    # 匹配
    with torch.no_grad():
        matches_result = matcher({"image0": feats0, "image1": feats1})

    kpts0 = feats0["keypoints"][0].cpu().numpy()
    kpts1 = feats1["keypoints"][0].cpu().numpy()
    matches0 = matches_result["matches0"][0].cpu().numpy()

    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    n_matches = len(mkpts0)

    if n_matches < min_matches:
        return SubImageMatchResult(
            found=False, confidence=0.0,
            elapsed_ms=round(elapsed_ms, 1),
            method=f"SuperPoint+LightGlue ({n_matches} matches)",
        )

    # 用单应矩阵定位模板区域
    H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.USAC_MAGSAC, 5.0)

    if H is None:
        return SubImageMatchResult(
            found=False, confidence=0.0,
            elapsed_ms=round(elapsed_ms, 1),
            method=f"SuperPoint+LightGlue ({n_matches} matches, H failed)",
        )

    tpl_h, tpl_w = template.shape[:2]
    corners = np.float32([
        [0, 0], [tpl_w, 0], [tpl_w, tpl_h], [0, tpl_h]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x_min = max(0, int(projected[:, 0].min()))
    y_min = max(0, int(projected[:, 1].min()))
    x_max = min(img_w, int(projected[:, 0].max()))
    y_max = min(img_h, int(projected[:, 1].max()))

    inlier_ratio = mask.sum() / len(mask) if mask is not None else 0
    confidence = float(inlier_ratio)
    found = confidence >= confidence_threshold

    return SubImageMatchResult(
        found=found,
        confidence=round(confidence, 4),
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        # 百分比与 WebUI 一致：左上角 / 右下角
        x_min_pct=round(x_min / img_w * 100, 2),
        y_min_pct=round(y_min / img_h * 100, 2),
        x_max_pct=round(x_max / img_w * 100, 2),
        y_max_pct=round(y_max / img_h * 100, 2),
        elapsed_ms=round(elapsed_ms, 1),
        method=f"SuperPoint+LightGlue ({n_matches} matches, "
               f"{int(inlier_ratio*100)}% inliers)",
    )


# ============================================================================
# 子图匹配器（对外接口）
# ============================================================================

class SubImageMatcher:
    """
    子图匹配器

    模型在首次调用 match() 或 preload() 时加载到 GPU，
    后续所有匹配复用同一模型实例。
    """

    def __init__(self, device: str = "cuda:0",
                 min_matches: int = 8,
                 confidence_threshold: float = 0.3):
        """
        Args:
            device: 推理设备（模型缓存为全局单例）
            min_matches: 最小特征匹配数
            confidence_threshold: 置信度阈值
        """
        global _device
        _device = device if torch.cuda.is_available() else "cpu"
        self.min_matches = min_matches
        self.confidence_threshold = confidence_threshold

    def preload(self):
        """预加载模型（可在启动时调用，避免首次匹配延迟）。"""
        _load_models()

    def match(self, camera_image: np.ndarray,
              crop_image: np.ndarray) -> SubImageMatchResult:
        """
        在相机图中定位子图。

        Args:
            camera_image: 当前相机图像 (BGR, H×W×3)
            crop_image: 记忆中的 crop 子图 (BGR, H×W×3)

        Returns:
            SubImageMatchResult（包含左上角/右下角百分比）
        """
        try:
            return _match_features(
                camera_image, crop_image,
                min_matches=self.min_matches,
                confidence_threshold=self.confidence_threshold,
            )
        except Exception as e:
            logger.error(f"[SubImageMatcher] 匹配异常: {e}")
            return SubImageMatchResult(
                found=False, confidence=0.0,
                method=f"SuperPoint+LightGlue (异常: {e})"
            )

    def match_from_path(self, camera_image: np.ndarray,
                        crop_image_path: str) -> SubImageMatchResult:
        """
        从文件路径加载 crop 图后匹配。

        Args:
            camera_image: 当前相机图像 (BGR)
            crop_image_path: crop 子图文件路径

        Returns:
            SubImageMatchResult
        """
        if not os.path.exists(crop_image_path):
            logger.warning(f"[SubImageMatcher] crop 文件不存在: {crop_image_path}")
            return SubImageMatchResult(
                found=False, confidence=0.0,
                method=f"文件不存在: {crop_image_path}"
            )

        crop_image = cv2.imread(crop_image_path)
        if crop_image is None:
            logger.warning(f"[SubImageMatcher] crop 图像读取失败: {crop_image_path}")
            return SubImageMatchResult(
                found=False, confidence=0.0,
                method=f"图像读取失败: {crop_image_path}"
            )

        return self.match(camera_image, crop_image)
