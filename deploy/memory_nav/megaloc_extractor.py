#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MegaLoc VPR 特征提取器

基于 MegaLoc (CVPR 2025 Workshop) 的视觉位置识别特征提取器。
使用 DINOv2 + Optimal Transport 聚合生成全局描述子。

参考:
- MegaLoc: One Retrieval to Place Them All (CVPR 2025)
- https://github.com/gmberton/MegaLoc
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
    logger.warning("[MegaLoc] PyTorch 不可用")

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

# MegaLoc 本地仓库路径
_MEGALOC_REPO_PATH = "/home/ubuntu/Disk/codes/jianxiong/MegaLoc"


class MegaLocExtractor:
    """
    MegaLoc VPR 特征提取器

    通过 torch.hub (本地或远程) 加载预训练的 MegaLoc 模型，提取全局描述子。
    输出特征维度: 8448

    Args:
        repo_path: MegaLoc 仓库本地路径 (None=自动检测)
        max_img_size: 最大图像边长 (需为14的倍数，默认518)
        device: 计算设备
    """

    FEATURE_DIM = 8448  # MegaLoc 默认输出维度

    def __init__(self,
                 repo_path: str = None,
                 max_img_size: int = 518,
                 device: str = "cuda:0"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用，无法使用 MegaLoc")

        self.repo_path = repo_path or _MEGALOC_REPO_PATH
        self.max_img_size = max_img_size
        self.device = device
        self.feature_dim = self.FEATURE_DIM
        self.model = None

        # 图像预处理 (与 DINOv2 一致)
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])

        self._load_model()

        logger.info(f"[MegaLoc] 初始化完成: dim={self.feature_dim}, "
                    f"max_img_size={max_img_size}, device={device}")

    def _load_model(self):
        """加载 MegaLoc 预训练模型"""
        try:
            # 优先使用本地仓库
            if os.path.isdir(self.repo_path) and os.path.exists(
                    os.path.join(self.repo_path, 'hubconf.py')):
                logger.info(f"[MegaLoc] 使用本地仓库: {self.repo_path}")
                self.model = torch.hub.load(
                    self.repo_path, "get_trained_model",
                    source='local', trust_repo=True)
            else:
                # 尝试远程
                self.model = torch.hub.load(
                    "gmberton/MegaLoc", "get_trained_model",
                    trust_repo=True)

            self.model = self.model.eval().to(self.device)
            logger.info("[MegaLoc] 模型加载成功")
        except Exception as e:
            logger.error(f"[MegaLoc] 模型加载失败: {e}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> 'torch.Tensor':
        """预处理图像为模型输入"""
        if CV2_AVAILABLE and len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        pil_img = PILImage.fromarray(rgb)

        img_pt = self.base_tf(pil_img).to(self.device)

        c, h, w = img_pt.shape
        if max(h, w) > self.max_img_size:
            if h >= w:
                w_new = int(w * self.max_img_size / h)
                h_new = self.max_img_size
            else:
                h_new = int(h * self.max_img_size / w)
                w_new = self.max_img_size
            img_pt = tvf.functional.resize(img_pt, (h_new, w_new),
                                            interpolation=tvf.InterpolationMode.BICUBIC)
            c, h, w = img_pt.shape

        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
        return img_pt

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像的全局描述子

        Args:
            image: BGR 图像 (numpy array)

        Returns:
            全局描述子 (8448,)
        """
        if self.model is None:
            return np.random.randn(self.feature_dim).astype(np.float32)

        try:
            img_pt = self._preprocess_image(image)
            with torch.no_grad():
                descriptor = self.model(img_pt)  # [1, 8448]
            return descriptor.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"[MegaLoc] 特征提取失败: {e}")
            return np.random.randn(self.feature_dim).astype(np.float32)

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        features = [self.extract(img) for img in images]
        return np.array(features)

    @property
    def output_dim(self) -> int:
        return self.feature_dim
