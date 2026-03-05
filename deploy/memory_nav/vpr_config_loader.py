#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPR 统一配置加载器

所有 VPR 相关模块通过此加载器读取 deploy/vpr_config.yaml，
确保 vpr_method、device、similarity_threshold 等参数全局一致。

用法:
    from deploy.memory_nav.vpr_config_loader import load_vpr_config
    cfg = load_vpr_config()
    # cfg['vpr_method']            -> 'selavpr'
    # cfg['device']                -> 'cuda:0'
    # cfg['similarity_threshold']  -> 0.70  (当前方法对应的阈值)
    # cfg['anyloc']                -> {...}  (AnyLoc 专用配置)
"""

import os
import logging

logger = logging.getLogger(__name__)

# 缓存，避免重复读取
_cached_config = None

def load_vpr_config(config_path: str = None, force_reload: bool = False) -> dict:
    """
    加载 VPR 统一配置
    
    Args:
        config_path: 配置文件路径，None 则自动查找 deploy/vpr_config.yaml
        force_reload: 强制重新读取（忽略缓存）
    
    Returns:
        dict: 配置字典，包含 vpr_method, device, similarity_threshold 等
    """
    global _cached_config
    if _cached_config is not None and not force_reload and config_path is None:
        return _cached_config.copy()
    
    # 默认值
    defaults = {
        'vpr_method': 'selavpr',
        'device': 'cuda:0',
        'similarity_threshold': {
            'selavpr': 0.70,
            'megaloc': 0.70,
            'effovpr': 0.70,
            'anyloc': 0.70,
        },
        'anyloc': {
            'dino_model': 'dinov2_vitb14',
            'agg_mode': 'vlad',
            'num_clusters': 8,
            'domain': 'indoor',
            'max_img_size': 630,
        },
    }
    
    # 查找配置文件
    if config_path is None:
        # 从当前文件位置推导: memory_nav/ -> deploy/ -> vpr_config.yaml
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(this_dir), 'vpr_config.yaml')
    
    cfg = dict(defaults)
    
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_cfg = yaml.safe_load(f) or {}
            
            if 'vpr_method' in file_cfg:
                cfg['vpr_method'] = file_cfg['vpr_method']
            if 'device' in file_cfg:
                cfg['device'] = file_cfg['device']
            if 'similarity_threshold' in file_cfg:
                if isinstance(file_cfg['similarity_threshold'], dict):
                    cfg['similarity_threshold'].update(file_cfg['similarity_threshold'])
                else:
                    # 兼容: 如果写成单个数值，应用到所有方法
                    for k in cfg['similarity_threshold']:
                        cfg['similarity_threshold'][k] = float(file_cfg['similarity_threshold'])
            if 'anyloc' in file_cfg and isinstance(file_cfg['anyloc'], dict):
                cfg['anyloc'].update(file_cfg['anyloc'])
            
            logger.info(f"[VPR Config] 已加载: {config_path} -> method={cfg['vpr_method']}, device={cfg['device']}")
        except Exception as e:
            logger.warning(f"[VPR Config] 加载失败 ({config_path}): {e}，使用默认值")
    else:
        logger.warning(f"[VPR Config] 配置文件不存在: {config_path}，使用默认值")
    
    # 环境变量覆盖 (向后兼容)
    env_method = os.environ.get('VPR_METHOD')
    if env_method:
        cfg['vpr_method'] = env_method
        logger.info(f"[VPR Config] 环境变量覆盖 VPR_METHOD={env_method}")
    
    _cached_config = cfg
    return cfg.copy()


def get_threshold(cfg: dict = None) -> float:
    """获取当前 VPR 方法对应的相似度阈值"""
    if cfg is None:
        cfg = load_vpr_config()
    method = cfg['vpr_method']
    thresholds = cfg.get('similarity_threshold', {})
    return float(thresholds.get(method, 0.70))


def get_anyloc_config(cfg: dict = None) -> dict:
    """获取 AnyLoc 专用配置（仅 vpr_method=anyloc 时有意义）"""
    if cfg is None:
        cfg = load_vpr_config()
    return cfg.get('anyloc', {})
