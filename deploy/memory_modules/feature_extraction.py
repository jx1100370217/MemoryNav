#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - LongCLIP视觉特征提取模块

基于 LongCLIP 的视觉特征提取器，用于视觉位置识别 (VPR) 和回环检测。
"""

import logging
from typing import List
import numpy as np
import cv2
from PIL import Image

import torch
from torchvision.transforms import ToPILImage

# 尝试导入 LongCLIP
try:
    from pathlib import Path
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from internnav.model.basemodel.LongCLIP.model import longclip
    LONGCLIP_AVAILABLE = True
except ImportError:
    LONGCLIP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LongCLIPFeatureExtractor:
    """
    基于 LongCLIP 的视觉特征提取器

    使用 InternNav 已集成的 LongCLIP 模型提取图像特征，
    用于视觉位置识别 (VPR) 和回环检测。
    """

    def __init__(self, model_path: str, device: str = "cuda:0", feature_dim: int = 512):
        """
        初始化 LongCLIP 特征提取器

        Args:
            model_path: LongCLIP 模型权重路径
            device: 推理设备
            feature_dim: 输出特征维度 (LongCLIP-B=512)
        """
        self.device = torch.device(device)
        self.to_pil = ToPILImage()
        self.feature_dim = feature_dim  # 保存特征维度用于回退方案

        logger.info(f"加载 LongCLIP 模型: {model_path}")

        if not LONGCLIP_AVAILABLE:
            logger.error("LongCLIP 模块不可用")
            self.is_available = False
            self.model = None
            self.preprocess = None
            return

        try:
            # 加载 LongCLIP 模型
            self.model, self.preprocess = longclip.load(model_path, device=device)

            # 仅保留视觉编码器，删除文本部分以节省内存
            if hasattr(self.model, 'token_embedding'):
                del self.model.token_embedding
            if hasattr(self.model, 'transformer'):
                del self.model.transformer
            if hasattr(self.model, 'positional_embedding'):
                del self.model.positional_embedding
            if hasattr(self.model, 'ln_final'):
                del self.model.ln_final

            # 设置为评估模式
            if hasattr(self.model, 'set_mode_for_inference'):
                self.model.set_mode_for_inference()
            else:
                self.model.eval()

            # 冻结参数
            for param in self.model.parameters():
                param.requires_grad_(False)

            self.is_available = True
            logger.info("LongCLIP 特征提取器初始化成功")

        except Exception as e:
            logger.error(f"LongCLIP 加载失败: {e}, 将使用简化特征提取")
            self.is_available = False
            self.model = None
            self.preprocess = None

    def extract_feature(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        提取单张图像的视觉特征

        Args:
            rgb_image: RGB图像 [H, W, 3], uint8 或 float

        Returns:
            feature: 归一化特征向量 [768]
        """
        if not self.is_available:
            return self._fallback_extract(rgb_image)

        try:
            # 转换为 PIL Image
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)

            # 确保是 RGB 格式
            if len(rgb_image.shape) == 2:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
            elif rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[:, :, :3]

            pil_image = Image.fromarray(rgb_image)

            # 预处理
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                feature = self._encode_image(image_tensor)

            # 转换为 numpy 并归一化
            feature = feature.cpu().numpy().flatten().astype('float32')
            feature = feature / (np.linalg.norm(feature) + 1e-8)

            return feature

        except Exception as e:
            logger.warning(f"LongCLIP 特征提取失败: {e}, 使用回退方案")
            return self._fallback_extract(rgb_image)

    def _encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用 LongCLIP 视觉编码器提取特征
        """
        visual = self.model.visual

        # 获取数据类型
        try:
            data_type = visual.conv1.weight.dtype
        except Exception:
            data_type = torch.float32

        x = image_tensor.type(data_type)

        # 卷积 patch embedding
        x = visual.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
        x = x.permute(0, 2, 1)  # [B, grid**2, width]

        # 添加 CLS token
        x = torch.cat([
            visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        # 添加位置编码
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        # Transformer 编码
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 提取 CLS token 特征
        x = visual.ln_post(x[:, 0, :])

        # 投影到输出空间
        if visual.proj is not None:
            x = x @ visual.proj

        return x

    def _fallback_extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        回退方案：简化的特征提取（当 LongCLIP 不可用时）
        """
        # 缩放图像
        image_small = cv2.resize(rgb_image, (64, 64))
        image_flat = image_small.flatten().astype('float32')

        # 截断或填充到配置的特征维度
        feature = np.zeros(self.feature_dim, dtype='float32')
        length = min(len(image_flat), self.feature_dim)
        feature[:length] = image_flat[:length]

        # 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        批量提取图像特征

        Args:
            images: RGB图像列表

        Returns:
            features: 特征矩阵 [N, feature_dim]
        """
        features = []
        for img in images:
            feat = self.extract_feature(img)
            features.append(feat)
        return np.array(features, dtype='float32')
