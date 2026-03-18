#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav - 子图匹配模块 (DINOv3 方案)

支持 DINOv3 密集特征匹配，成功匹配阈值: 0.6
"""

import os
import logging
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 子图匹配结果 (统一输出格式)
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
# 基类：子图匹配策略
# ============================================================================

class BaseSubImageMatchStrategy(ABC):
    """子图匹配策略基类"""

    @abstractmethod
    def name(self) -> str:
        """方案名称"""
        ...

    @abstractmethod
    def preload(self):
        """预加载模型"""
        ...

    @abstractmethod
    def match(self, camera_image: np.ndarray, crop_image: np.ndarray,
              min_matches: int = 50,
              confidence_threshold: float = 0.6) -> SubImageMatchResult:
        """
        在相机图中定位子图。

        Args:
            camera_image: 当前相机图像 (BGR, H×W×3)
            crop_image: 记忆中的 crop 子图 (BGR, H×W×3)
            min_matches: 最小特征匹配数 (仅部分方案使用)
            confidence_threshold: 置信度阈值 (默认 0.6)

        Returns:
            SubImageMatchResult
        """
        ...


# ============================================================================
# 方案: DINOv3 密集特征匹配
# ============================================================================

class DINOv3Strategy(BaseSubImageMatchStrategy):
    """基于 DINOv3 密集 patch 特征的子图匹配 (通过 timm 加载)

    原理：
    1. 使用 DINOv3 (ViT-B/16) 提取两张图的 patch token 特征 (密集特征图)
    2. 在相机图特征图上滑动窗口，计算与子图特征图的余弦相似度
    3. 找到相似度最高的位置作为匹配结果

    DINOv3 相比 DINOv2 在密集特征质量上有显著提升。
    """

    def __init__(self, device: str = "cuda:0",
                 model_name: str = "vit_base_patch16_dinov3"):
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model_name = model_name
        self._model = None
        self._patch_size = 16  # DINOv3 ViT-B/16
        self._feature_dim = 768  # ViT-B

    def name(self) -> str:
        return "dinov3"

    def preload(self):
        if self._model is not None:
            return
        import timm
        logger.info(f"[DINOv3] 正在通过 timm 加载 {self._model_name} ...")
        self._model = timm.create_model(
            self._model_name,
            pretrained=True,
            num_classes=0,  # 去掉分类头
        )
        self._model = self._model.eval().to(self._device)

        # 获取 patch_size 和 feature_dim
        if hasattr(self._model, 'patch_embed'):
            ps = getattr(self._model.patch_embed, 'patch_size', None)
            if ps is not None:
                self._patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        if hasattr(self._model, 'embed_dim'):
            self._feature_dim = self._model.embed_dim

        logger.info(f"[DINOv3] 模型加载完成 (device={self._device}, "
                     f"patch_size={self._patch_size}, dim={self._feature_dim})")

    def _prepare_image(self, image: np.ndarray, target_size: int = 518):
        """预处理图像，确保尺寸为 patch_size 的整数倍。

        Returns:
            (tensor [1,3,H,W], resized_h, resized_w)
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 确保是 patch_size 的整数倍
        new_h = max(self._patch_size, (new_h // self._patch_size) * self._patch_size)
        new_w = max(self._patch_size, (new_w // self._patch_size) * self._patch_size)

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std

        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        return tensor, new_h, new_w

    def _extract_patch_features(self, image: np.ndarray, target_size: int = 518):
        """提取 DINOv3 patch 特征。

        Returns:
            (features [n_patches_h * n_patches_w, dim], n_patches_h, n_patches_w)
        """
        tensor, resized_h, resized_w = self._prepare_image(image, target_size)
        n_patches_h = resized_h // self._patch_size
        n_patches_w = resized_w // self._patch_size

        with torch.no_grad():
            features = self._model.forward_features(tensor)
            # DINOv3 输出: [1, num_prefix_tokens + n_patches, dim]
            # num_prefix_tokens = CLS (1) + register tokens (DINOv3 通常有4个)
            n_patches = n_patches_h * n_patches_w
            n_prefix = features.shape[1] - n_patches
            if n_prefix > 0:
                patch_tokens = features[:, n_prefix:, :]
            else:
                patch_tokens = features

        return patch_tokens[0], n_patches_h, n_patches_w  # [n_patches, dim]

    def match(self, camera_image: np.ndarray, crop_image: np.ndarray,
              min_matches: int = 50,
              confidence_threshold: float = 0.6) -> SubImageMatchResult:
        t0 = time.perf_counter()
        self.preload()

        orig_h, orig_w = camera_image.shape[:2]

        # 提取特征
        cam_feats, cam_ph, cam_pw = self._extract_patch_features(camera_image, target_size=518)
        crop_h, crop_w = crop_image.shape[:2]
        crop_target = max(self._patch_size * 2,
                         min(280, int(518 * max(crop_h, crop_w) / max(orig_h, orig_w))))
        crop_target = max(crop_target, self._patch_size * 2)
        crop_feats, crop_ph, crop_pw = self._extract_patch_features(crop_image, target_size=crop_target)

        # reshape 为空间网格
        dim = cam_feats.shape[-1]
        cam_grid = cam_feats.reshape(cam_ph, cam_pw, dim)
        crop_grid = crop_feats.reshape(crop_ph, crop_pw, dim)

        # 归一化
        cam_grid = torch.nn.functional.normalize(cam_grid, dim=-1)
        crop_grid = torch.nn.functional.normalize(crop_grid, dim=-1)

        # crop 网格比 cam 网格大，无法匹配
        if crop_ph > cam_ph or crop_pw > cam_pw:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SubImageMatchResult(
                found=False, confidence=0.0,
                elapsed_ms=round(elapsed_ms, 1),
                method=f"DINOv3 (crop patches {crop_ph}x{crop_pw} > cam {cam_ph}x{cam_pw})",
            )

        # 滑动窗口匹配 (unfold 加速)
        cam_grid_4d = cam_grid.permute(2, 0, 1).unsqueeze(0)  # [1, dim, cam_ph, cam_pw]
        crop_flat = crop_grid.reshape(-1, dim)  # [crop_ph * crop_pw, dim]

        out_h = cam_ph - crop_ph + 1
        out_w = cam_pw - crop_pw + 1

        windows = cam_grid_4d.unfold(2, crop_ph, 1).unfold(3, crop_pw, 1)
        windows = windows[0].permute(1, 2, 3, 4, 0)  # [out_h, out_w, crop_ph, crop_pw, dim]
        windows = windows.reshape(out_h, out_w, -1, dim)

        crop_flat_expanded = crop_flat.unsqueeze(0).unsqueeze(0)
        sim = (windows * crop_flat_expanded).sum(dim=-1).mean(dim=-1)  # [out_h, out_w]

        sim_np = sim.cpu().numpy()
        best_idx = np.unravel_index(sim_np.argmax(), sim_np.shape)
        best_y, best_x = best_idx
        best_score = float(sim_np[best_y, best_x])

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # patch 坐标 → 原图像素坐标
        cam_resized_h = cam_ph * self._patch_size
        cam_resized_w = cam_pw * self._patch_size
        scale_back_x = orig_w / cam_resized_w
        scale_back_y = orig_h / cam_resized_h

        x_min_px = max(0, min(int(best_x * self._patch_size * scale_back_x), orig_w))
        y_min_px = max(0, min(int(best_y * self._patch_size * scale_back_y), orig_h))
        x_max_px = max(0, min(int((best_x + crop_pw) * self._patch_size * scale_back_x), orig_w))
        y_max_px = max(0, min(int((best_y + crop_ph) * self._patch_size * scale_back_y), orig_h))

        confidence = max(0.0, min(1.0, best_score))
        found = confidence >= confidence_threshold

        return SubImageMatchResult(
            found=found,
            confidence=round(confidence, 4),
            x_min=x_min_px, y_min=y_min_px,
            x_max=x_max_px, y_max=y_max_px,
            x_min_pct=round(x_min_px / orig_w * 100, 2),
            y_min_pct=round(y_min_px / orig_h * 100, 2),
            x_max_pct=round(x_max_px / orig_w * 100, 2),
            y_max_pct=round(y_max_px / orig_h * 100, 2),
            elapsed_ms=round(elapsed_ms, 1),
            method=f"DINOv3 (score={best_score:.4f}, grid={cam_ph}x{cam_pw}, "
                   f"crop={crop_ph}x{crop_pw})",
        )


# ============================================================================
# 方案注册表 (仅 DINOv3)
# ============================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    "dinov3": DINOv3Strategy,
}

# 方案显示名称
STRATEGY_DISPLAY_NAMES: Dict[str, str] = {
    "dinov3": "DINOv3",
}


def list_strategies() -> List[str]:
    """返回所有可用方案名称"""
    return list(STRATEGY_REGISTRY.keys())


# ============================================================================
# 子图匹配器 (对外接口)
# ============================================================================

class SubImageMatcher:
    """
    子图匹配器 (DINOv3 方案)

    成功匹配阈值: 0.6
    """

    def __init__(self, device: str = "cuda:0",
                 min_matches: int = 50,
                 confidence_threshold: float = 0.6,
                 default_method: str = "dinov3"):
        """
        Args:
            device: 推理设备
            min_matches: 最小特征匹配数
            confidence_threshold: 置信度阈值 (默认 0.6)
            default_method: 默认匹配方案 (DINOv3)
        """
        self._device = device if torch.cuda.is_available() else "cpu"
        self.min_matches = min_matches
        self.confidence_threshold = confidence_threshold
        self.default_method = default_method

        # 策略实例缓存 (DINOv3)
        self._strategies: Dict[str, BaseSubImageMatchStrategy] = {}

    def _get_strategy(self, method: Optional[str] = None) -> BaseSubImageMatchStrategy:
        """获取或创建策略实例"""
        method = method or self.default_method
        if method not in STRATEGY_REGISTRY:
            raise ValueError(f"未知匹配方案: {method}, 可选: {list(STRATEGY_REGISTRY.keys())}")

        if method not in self._strategies:
            cls = STRATEGY_REGISTRY[method]
            self._strategies[method] = cls(device=self._device)

        return self._strategies[method]

    def preload(self, method: Optional[str] = None):
        """预加载 DINOv3 模型"""
        strategy = self._get_strategy(method)
        strategy.preload()

    def preload_all(self):
        """预加载所有方案的模型 (DINOv3)"""
        for method_name in STRATEGY_REGISTRY:
            try:
                logger.info(f"[SubImageMatcher] 预加载方案: {method_name}")
                strategy = self._get_strategy(method_name)
                strategy.preload()
                logger.info(f"[SubImageMatcher] 方案 {method_name} 加载完成")
            except Exception as e:
                logger.warning(f"[SubImageMatcher] 方案 {method_name} 预加载失败: {e}")

    def match(self, camera_image: np.ndarray,
              crop_image: np.ndarray,
              method: Optional[str] = None) -> SubImageMatchResult:
        """
        在相机图中定位子图。

        Args:
            camera_image: 当前相机图像 (BGR, H×W×3)
            crop_image: 记忆中的 crop 子图 (BGR, H×W×3)
            method: 匹配方案 (使用 DINOv3)

        Returns:
            SubImageMatchResult
        """
        try:
            strategy = self._get_strategy(method)
            return strategy.match(
                camera_image, crop_image,
                min_matches=self.min_matches,
                confidence_threshold=self.confidence_threshold,
            )
        except Exception as e:
            method_name = method or self.default_method
            logger.error(f"[SubImageMatcher] {method_name} 匹配异常: {e}")
            return SubImageMatchResult(
                found=False, confidence=0.0,
                method=f"{STRATEGY_DISPLAY_NAMES.get(method_name, method_name)} (异常: {e})"
            )

    def match_from_path(self, camera_image: np.ndarray,
                        crop_image_path: str,
                        method: Optional[str] = None) -> SubImageMatchResult:
        """从文件路径加载 crop 图后匹配"""
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

        return self.match(camera_image, crop_image, method=method)

    @staticmethod
    def available_methods() -> List[str]:
        """返回所有可用方案名称"""
        return list_strategies()

    @staticmethod
    def method_display_names() -> Dict[str, str]:
        """返回方案名 → 显示名映射"""
        return dict(STRATEGY_DISPLAY_NAMES)
