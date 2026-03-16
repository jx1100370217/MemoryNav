#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav - 子图匹配模块

基于 SuperPoint + LightGlue 的子图定位，用于在导航过程中
在当前相机图中定位记忆中的注意力子图（crop）。

包含 Homography 退化检测，避免输出垃圾 bbox。

模型生命周期：启动时加载一次，后续复用。
"""

import os
import logging
import time
import math
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

# 匹配分辨率（提取特征前 resize 到此分辨率，坐标映射回原图）
_MATCH_W = 1280
_MATCH_H = 960


def _load_models():
    """加载并缓存 SuperPoint + LightGlue 模型（仅首次调用时加载）。"""
    global _extractor, _matcher
    if _extractor is not None:
        return _extractor, _matcher

    from lightglue import LightGlue, SuperPoint

    logger.info("[SubImageMatcher] 正在加载 SuperPoint ...")
    _extractor = SuperPoint(max_num_keypoints=4096).eval().to(_device)
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

    # 中心百分比（从 bounding box 自动计算）
    @property
    def center_x_pct(self) -> float:
        return (self.x_min_pct + self.x_max_pct) / 2.0

    @property
    def center_y_pct(self) -> float:
        return (self.y_min_pct + self.y_max_pct) / 2.0

    # 耗时 & 方法
    elapsed_ms: float = 0.0
    method: str = ""

    def to_dict(self) -> Dict:
        top_left = {
            'x': round(self.x_min_pct / 100.0, 4),
            'y': round(self.y_min_pct / 100.0, 4),
        }
        bottom_right = {
            'x': round(self.x_max_pct / 100.0, 4),
            'y': round(self.y_max_pct / 100.0, 4),
        }
        center = {
            'x': round((top_left['x'] + bottom_right['x']) / 2, 4),
            'y': round((top_left['y'] + bottom_right['y']) / 2, 4),
        }
        return {
            'found': self.found,
            'confidence': round(self.confidence, 4),
            'top_left_pct': top_left,
            'bottom_right_pct': bottom_right,
            'center_pct': center,
            'bbox_pixel': {
                'x_min': self.x_min, 'y_min': self.y_min,
                'x_max': self.x_max, 'y_max': self.y_max,
            },
            'elapsed_ms': round(self.elapsed_ms, 1),
            'method': self.method,
        }


# ============================================================================
# Homography 质量检测
# ============================================================================

def _is_homography_degenerate(H, img_shape, tpl_shape, projected_corners):
    """
    检测 Homography 是否退化。

    Args:
        H: 3x3 Homography 矩阵
        img_shape: 搜索图像 (H, W, ...)
        tpl_shape: 模板图像 (H, W, ...)
        projected_corners: 投影后的四角坐标 (4, 2)

    Returns:
        True 表示退化（不可信）
    """
    # 1. 行列式检查
    det = np.linalg.det(H[:2, :2])
    if det < 0.05 or det > 20:
        return True

    # 2. 投影区域面积 vs 图像面积
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    proj_area = cv2.contourArea(projected_corners.astype(np.float32))
    area_ratio = proj_area / img_area
    if area_ratio > 0.3:  # 占图超过 30% → 退化
        return True
    if area_ratio < 0.005:  # 占图不到 0.5% → 太小
        return True
    if proj_area < 100:  # 绝对面积太小
        return True

    # 2b. 投影面积 vs 模板面积的缩放比
    tpl_h_local, tpl_w_local = tpl_shape[:2]
    tpl_area = tpl_h_local * tpl_w_local
    if tpl_area > 0:
        scale_ratio = proj_area / tpl_area
        if scale_ratio < 0.1 or scale_ratio > 10.0:  # 缩放超过10倍 → 退化
            return True

    # 3. 凸性检查：非凸说明 fold
    if not cv2.isContourConvex(projected_corners.astype(np.int32)):
        return True

    # 4. 宽高比与模板偏差检查
    tpl_h, tpl_w = tpl_shape[:2]
    tpl_aspect = tpl_w / max(tpl_h, 1)
    br = projected_corners.max(axis=0)
    tl = projected_corners.min(axis=0)
    proj_w = br[0] - tl[0]
    proj_h = br[1] - tl[1]
    proj_aspect = proj_w / max(proj_h, 1)
    if tpl_aspect > 0:
        aspect_ratio = proj_aspect / tpl_aspect
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return True

    return False


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
    confidence_threshold: float = 0.55,
) -> SubImageMatchResult:
    """
    使用 SuperPoint + LightGlue 在 image 中定位 template。

    返回与 SubImageLocator WebUI 一致的百分比描述。
    """
    t0 = time.perf_counter()

    extractor, matcher = _load_models()
    orig_h, orig_w = image.shape[:2]
    img_h, img_w = orig_h, orig_w  # 百分比计算基于原始尺寸

    # Resize 到匹配分辨率以获取更多特征点
    scale_x = orig_w / _MATCH_W
    scale_y = orig_h / _MATCH_H
    image_resized = cv2.resize(image, (_MATCH_W, _MATCH_H), interpolation=cv2.INTER_LINEAR)
    # 模板按相同比例缩放（保持宽高比）
    tpl_h_orig, tpl_w_orig = template.shape[:2]
    tpl_w_resized = max(1, int(tpl_w_orig / scale_x))
    tpl_h_resized = max(1, int(tpl_h_orig / scale_y))
    template_resized = cv2.resize(template, (tpl_w_resized, tpl_h_resized), interpolation=cv2.INTER_LINEAR)

    # 提取特征（在 resize 后的图像上）
    feats0 = _extract_features(extractor, image_resized)
    feats1 = _extract_features(extractor, template_resized)

    # 匹配
    with torch.no_grad():
        matches_result = matcher({"image0": feats0, "image1": feats1})

    kpts0 = feats0["keypoints"][0].cpu().numpy()
    kpts1 = feats1["keypoints"][0].cpu().numpy()
    matches0 = matches_result["matches0"][0].cpu().numpy()

    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]

    # 将关键点坐标映射回原始分辨率
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y

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

    tpl_h, tpl_w = tpl_h_orig, tpl_w_orig  # 使用原始模板尺寸
    corners = np.float32([
        [0, 0], [tpl_w, 0], [tpl_w, tpl_h], [0, tpl_h]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    inlier_ratio = mask.sum() / len(mask) if mask is not None else 0
    n_inliers = int(mask.sum()) if mask is not None else 0
    # confidence = sqrt(inlier_ratio * inlier_quality)
    # inlier_quality 惩罚低 inlier 数量的情况（防止少量匹配点碰巧得高 ratio）
    _MIN_INLIERS = 20
    inlier_quality = min(1.0, n_inliers / _MIN_INLIERS)
    confidence = float(math.sqrt(inlier_ratio * inlier_quality))

    # Homography 退化检测
    if _is_homography_degenerate(H, image.shape, template.shape, projected):
        logger.debug(f"[SubImageMatcher] Homography 退化: "
                     f"det={np.linalg.det(H[:2,:2]):.3f}, "
                     f"inliers={int(inlier_ratio*100)}%")
        return SubImageMatchResult(
            found=False, confidence=round(confidence, 4),
            elapsed_ms=round(elapsed_ms, 1),
            method=f"SuperPoint+LightGlue ({n_matches} matches, "
                   f"{n_inliers} inliers, {int(inlier_ratio*100)}%, H degenerate)",
        )

    x_min = max(0, int(projected[:, 0].min()))
    y_min = max(0, int(projected[:, 1].min()))
    x_max = min(img_w, int(projected[:, 0].max()))
    y_max = min(img_h, int(projected[:, 1].max()))

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
               f"{n_inliers} inliers, {int(inlier_ratio*100)}%)",
    )


# ============================================================================
# 子图匹配器（对外接口）
# ============================================================================

class SubImageMatcher:
    """
    子图匹配器

    模型在首次调用 match() 或 preload() 时加载到 GPU，
    后续所有匹配复用同一模型实例。

    包含 Homography 退化检测，自动拦截退化的匹配结果
    （投影面积过大、非凸、宽高比异常等）。
    """

    def __init__(self, device: str = "cuda:0",
                 min_matches: int = 8,
                 confidence_threshold: float = 0.55):
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
