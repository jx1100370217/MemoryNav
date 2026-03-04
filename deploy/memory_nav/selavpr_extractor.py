#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelaVPR++ 特征提取器

基于 SelaVPR++ (T-PAMI 2025) 的视觉位置识别特征提取器。
通过 MultiConv Adapter 适配 DINOv2 基础模型，支持标准 VPR 和哈希重排两种模式。

参考:
- SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition (T-PAMI 2025)
- https://github.com/Lu-Feng/SelaVPRplusplus
"""

import os
import sys
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision import transforms as tvf
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("[SelaVPR++] PyTorch 不可用")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# SelaVPR++ 本地仓库路径
_SELAVPR_REPO_PATH = "/home/ubuntu/Disk/codes/jianxiong/SelaVPRplusplus"


class SelaVPRExtractor:
    """
    SelaVPR++ VPR 特征提取器

    通过 torch.hub (本地或远程) 加载预训练模型。

    支持两种模式:
    - 标准 VPR: 输出高维浮点全局描述子 (base: 2048D, large: 4096D)
    - 哈希+重排: 输出二进制特征(512D) + 浮点特征

    Args:
        backbone: 'dinov2-base' 或 'dinov2-large'
        aggregation: 聚合方法 'gem', 'boq', 'salad'
        use_hashing: 是否使用哈希模式
        use_rerank: 是否使用重排 (需 use_hashing=True)
        repo_path: 本地仓库路径
        max_img_size: 最大图像边长
        device: 计算设备
    """

    FEATURE_DIMS = {
        'dinov2-base': 2048,
        'dinov2-large': 4096,
    }

    def __init__(self,
                 backbone: str = "dinov2-large",
                 aggregation: str = "gem",
                 use_hashing: bool = False,
                 use_rerank: bool = False,
                 repo_path: str = None,
                 max_img_size: int = 518,
                 device: str = "cuda:0"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用，无法使用 SelaVPR++")

        self.backbone_name = backbone
        self.aggregation = aggregation
        self.use_hashing = use_hashing
        self.use_rerank = use_rerank
        self.repo_path = repo_path or _SELAVPR_REPO_PATH
        self.max_img_size = max_img_size
        self.device = device

        # 标准 VPR 模式输出维度
        self.feature_dim = self.FEATURE_DIMS.get(backbone, 2048)

        # 图像预处理
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])

        self.model = None
        self._load_model()

        logger.info(f"[SelaVPR++] 初始化完成: backbone={backbone}, agg={aggregation}, "
                    f"hashing={use_hashing}, rerank={use_rerank}, "
                    f"dim={self.feature_dim}, device={device}")

    def _load_model(self):
        """加载 SelaVPR++ 预训练模型"""
        try:
            # 优先使用本地仓库
            if os.path.isdir(self.repo_path) and os.path.exists(
                    os.path.join(self.repo_path, 'hubconf.py')):
                logger.info(f"[SelaVPR++] 使用本地仓库: {self.repo_path}")
                self.model = torch.hub.load(
                    self.repo_path, 'SelaVPRplusplus',
                    source='local', trust_repo=True,
                    backbone=self.backbone_name,
                    aggregation=self.aggregation,
                    hashing=self.use_hashing,
                    rerank=self.use_rerank
                )
            else:
                self.model = torch.hub.load(
                    'Lu-Feng/SelaVPRplusplus', 'SelaVPRplusplus',
                    trust_repo=True,
                    backbone=self.backbone_name,
                    aggregation=self.aggregation,
                    hashing=self.use_hashing,
                    rerank=self.use_rerank
                )

            self.model = self.model.eval().to(self.device)
            logger.info("[SelaVPR++] 模型加载成功")
        except Exception as e:
            logger.error(f"[SelaVPR++] 模型加载失败: {e}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> "torch.Tensor":
        """预处理图像 - SelaVPR++ 要求固定 518x518 正方形输入"""
        if CV2_AVAILABLE and len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        pil_img = PILImage.fromarray(rgb)
        img_pt = self.base_tf(pil_img).to(self.device)
        # SelaVPR++ adapter 要求 patch grid 为正方形
        # 强制 resize 到 518x518 (37*14)
        img_pt = tvf.functional.resize(img_pt, (518, 518),
                                        interpolation=tvf.InterpolationMode.BICUBIC)
        return img_pt[None, ...]

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像的全局描述子

        Args:
            image: BGR 图像

        Returns:
            全局描述子 (feature_dim,)
        """
        if self.model is None:
            return np.random.randn(self.feature_dim).astype(np.float32)

        try:
            img_pt = self._preprocess_image(image)
            with torch.no_grad():
                output = self.model(img_pt)

                if self.use_hashing and self.use_rerank:
                    # 返回 (z, z1, x_g)，使用浮点全局特征 x_g 做 VPR
                    descriptor = output[2]
                elif self.use_hashing and not self.use_rerank:
                    # 返回 (z, z1)，使用连续特征 z
                    descriptor = output[0]
                else:
                    # 标准 VPR: 直接返回全局描述子
                    descriptor = output

            return descriptor.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"[SelaVPR++] 特征提取失败: {e}")
            return np.random.randn(self.feature_dim).astype(np.float32)

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        features = [self.extract(img) for img in images]
        return np.array(features)

    @property
    def output_dim(self) -> int:
        return self.feature_dim
