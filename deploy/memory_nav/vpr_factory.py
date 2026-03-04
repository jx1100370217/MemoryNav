#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPR 提取器工厂

统一创建不同 VPR 方法的特征提取器:
- anyloc: DINOv2 + VLAD/GeM (AnyLoc, RA-L 2023)
- megaloc: DINOv2 + Optimal Transport (MegaLoc, CVPR 2025)
- effovpr: DINOv2 多层 GeM 池化 (EffoVPR, 2024)
- selavpr: DINOv2 + MultiConv Adapter (SelaVPR++, T-PAMI 2025)
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def create_vpr_extractor(vpr_method: str,
                         device: str = "cuda:0",
                         config: Dict = None) -> Tuple:
    """
    创建 VPR 特征提取器

    Args:
        vpr_method: VPR 方法名称
            - 'anyloc': AnyLoc (DINOv2 + VLAD/GeM)
            - 'megaloc': MegaLoc (DINOv2 + OT聚合, 8448D)
            - 'effovpr': EffoVPR (DINOv2 多层GeM, 768D)
            - 'selavpr': SelaVPR++ (DINOv2 + MultiConv Adapter)
        device: 计算设备
        config: 方法特定配置参数

    Returns:
        (extractor, feature_dim, order_invariant)
        - extractor: 特征提取器实例
        - feature_dim: 输出特征维度
        - order_invariant: 是否使用无序匹配模式
    """
    cfg = config or {}

    if vpr_method == 'anyloc':
        from .anyloc_extractor import AnyLocExtractor
        extractor = AnyLocExtractor(
            dino_model=cfg.get('dino_model', 'dinov2_vitb14'),
            desc_facet=cfg.get('desc_facet', 'value'),
            agg_mode=cfg.get('agg_mode', 'vlad'),
            num_clusters=cfg.get('num_clusters', 32),
            domain=cfg.get('domain', 'indoor'),
            max_img_size=cfg.get('max_img_size', 630),
            device=device
        )
        return extractor, extractor.feature_dim, True

    elif vpr_method == 'megaloc':
        from .megaloc_extractor import MegaLocExtractor
        extractor = MegaLocExtractor(
            max_img_size=cfg.get('max_img_size', 518),
            device=device
        )
        return extractor, extractor.feature_dim, True

    elif vpr_method == 'effovpr':
        from .effovpr_extractor import EffoVPRExtractor
        extractor = EffoVPRExtractor(
            dino_model=cfg.get('dino_model', 'dinov2_vitb14'),
            output_dim=cfg.get('output_dim', 768),
            layers=cfg.get('layers', None),
            gem_p=cfg.get('gem_p', 3.0),
            max_img_size=cfg.get('max_img_size', 518),
            device=device
        )
        return extractor, extractor.feature_dim, True

    elif vpr_method == 'selavpr':
        from .selavpr_extractor import SelaVPRExtractor
        extractor = SelaVPRExtractor(
            backbone=cfg.get('backbone', 'dinov2-large'),
            aggregation=cfg.get('aggregation', 'gem'),
            use_hashing=cfg.get('use_hashing', False),
            use_rerank=cfg.get('use_rerank', False),
            max_img_size=cfg.get('max_img_size', 518),
            device=device
        )
        return extractor, extractor.feature_dim, True


    else:
        raise ValueError(f"不支持的 VPR 方法: {vpr_method}. "
                         f"支持: anyloc, megaloc, effovpr, selavpr")
