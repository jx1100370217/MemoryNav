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
from .vpr_config_loader import load_vpr_config, get_order_invariant
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
    _oi = get_order_invariant()

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
        return extractor, extractor.feature_dim, _oi

    elif vpr_method == 'megaloc':
        from .megaloc_extractor import MegaLocExtractor
        extractor = MegaLocExtractor(
            max_img_size=cfg.get('max_img_size', 518),
            device=device
        )
        return extractor, extractor.feature_dim, _oi

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
        return extractor, extractor.feature_dim, _oi

    elif vpr_method == 'selavpr':
        from .selavpr_extractor import SelaVPRExtractor
        from .vpr_config_loader import load_vpr_config, get_selavpr_config
        # 从统一配置读取 selavpr 参数，config 参数可覆盖
        _selavpr_cfg = get_selavpr_config()
        _selavpr_cfg.update(cfg)
        extractor = SelaVPRExtractor(
            backbone=_selavpr_cfg.get('backbone', 'dinov2-large'),
            aggregation=_selavpr_cfg.get('aggregation', 'gem'),
            use_hashing=_selavpr_cfg.get('use_hashing', True),
            use_rerank=_selavpr_cfg.get('use_rerank', True),
            max_img_size=_selavpr_cfg.get('max_img_size', 518),
            device=device
        )
        return extractor, extractor.feature_dim, _oi


    else:
        raise ValueError(f"不支持的 VPR 方法: {vpr_method}. "
                         f"支持: anyloc, megaloc, effovpr, selavpr")
