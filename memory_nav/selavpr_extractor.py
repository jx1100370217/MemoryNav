#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelaVPR++ 特征提取器

基于 SelaVPR++ (T-PAMI 2025) 的视觉位置识别特征提取器。
通过 MultiConv Adapter 适配 DINOv2 基础模型，支持标准 VPR 和哈希重排两种模式。

代码源自 SelaVPR++ 项目，已内嵌到 memory-nav 中，
不再依赖外部代码库。

参考:
- SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition (T-PAMI 2025)
- https://github.com/Lu-Feng/SelaVPRplusplus
"""

import os
import logging
from typing import List
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


def _build_selavpr_model(backbone: str, aggregation: str, hashing: bool, rerank: bool):
    """
    构建 SelaVPR++ 模型并加载预训练权重。
    逻辑源自原 hubconf.py，已内嵌到 memory-nav。
    """
    from .selavpr_model.network import GeoLocalizationNet

    class SimpleArgs:
        def __init__(self, **kwargs):
            self.backbone = kwargs.get('backbone', 'dinov2-large')
            self.aggregation = kwargs.get('aggregation', 'gem')
            self.hashing = kwargs.get('hashing', True)
            self.rerank = kwargs.get('rerank', True)
            self.resume = True
            self.foundation_model_path = None

    args = SimpleArgs(
        backbone=backbone,
        aggregation=aggregation,
        hashing=hashing,
        rerank=rerank,
    )
    vpr_model = GeoLocalizationNet(args)
    vpr_model = torch.nn.DataParallel(vpr_model)

    # 加载预训练权重
    if backbone == "dinov2-base":
        if not hashing:
            url = 'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_base.pth'
        elif hashing and rerank:
            url = 'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_base_rerank.pth'
        else:
            url = None
    elif backbone == "dinov2-large":
        if not hashing:
            url = 'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_large.pth'
        elif hashing and rerank:
            url = 'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_large_rerank.pth'
        else:
            url = None
    else:
        url = None

    if url is not None:
        state = torch.hub.load_state_dict_from_url(
            url, map_location=torch.device('cpu'), weights_only=False
        )
        vpr_model.load_state_dict(state["model_state_dict"])

    return vpr_model


class SelaVPRExtractor:
    """
    SelaVPR++ VPR 特征提取器

    直接使用内嵌的模型代码加载预训练模型，不再依赖外部仓库。

    支持两种模式:
    - 标准 VPR: 输出高维浮点全局描述子 (base: 2048D, large: 4096D)
    - 哈希+重排: 输出二进制特征(512D) + 浮点特征

    Args:
        backbone: 'dinov2-base' 或 'dinov2-large'
        aggregation: 聚合方法 'gem', 'boq', 'salad'
        use_hashing: 是否使用哈希模式
        use_rerank: 是否使用重排 (需 use_hashing=True)
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
        self.max_img_size = max_img_size
        self.device = device

        # repo_path 参数保留但不再使用（兼容旧配置）
        if repo_path:
            logger.info(f"[SelaVPR++] repo_path 参数已忽略，使用内嵌模型代码")

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
        """加载 SelaVPR++ 预训练模型（使用内嵌模型代码）"""
        try:
            logger.info("[SelaVPR++] 使用内嵌模型代码构建模型...")
            self.model = _build_selavpr_model(
                backbone=self.backbone_name,
                aggregation=self.aggregation,
                hashing=self.use_hashing,
                rerank=self.use_rerank,
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
        """批量提取特征 — 真正的 batch forward, 单次 GPU 推理处理所有图像

        Args:
            images: BGR 图像列表

        Returns:
            特征矩阵 (N, feature_dim)
        """
        if self.model is None or not images:
            return np.array([np.random.randn(self.feature_dim).astype(np.float32)
                             for _ in images])
        try:
            # 批量预处理: 所有图像 resize 到 518x518 并 stack
            tensors = [self._preprocess_image(img) for img in images]
            batch = torch.cat(tensors, dim=0)  # [N, 3, 518, 518]

            # 单次 forward pass
            with torch.no_grad():
                output = self.model(batch)
                if self.use_hashing and self.use_rerank:
                    descriptors = output[2]
                elif self.use_hashing and not self.use_rerank:
                    descriptors = output[0]
                else:
                    descriptors = output

            return descriptors.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"[SelaVPR++] 批量特征提取失败: {e}, 退回串行模式")
            return np.array([self.extract(img) for img in images])

    @property
    def output_dim(self) -> int:
        return self.feature_dim
