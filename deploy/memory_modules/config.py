#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 配置管理模块

包含MemoryNavigationConfig配置类，定义系统的所有可配置参数。
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent.parent


@dataclass
class MemoryNavigationConfig:
    """记忆导航配置"""
    # ★ 记忆功能开关 - 设置为False时行为与ws_proxy.py一致
    memory_enabled: bool = True

    # GPU设置 - 所有GPU卡号由用户统一配置
    # 注意：排除GPU 0，默认使用GPU 1~3
    gpu_id: Optional[str] = None  # None表示多GPU模式，"1"、"2"或"3"表示单GPU模式

    # 各模型GPU设备配置（仅在多GPU模式下生效，单GPU模式下统一使用cuda:0）
    main_model_device: str = "1"  # 主模型GPU编号
    feature_extractor_device: str = "1"  # 特征提取器GPU编号
    vlm_device: str = "2"  # VLM模型GPU编号

    # LongCLIP 特征提取器
    longclip_model_path: str = str(project_root / "checkpoints/longclip-B.pt")
    feature_dim: int = 512  # LongCLIP-B 实际输出维度

    # 视觉位置识别 (VPR) - 优化后的参数
    similarity_threshold: float = 0.78  # 主阈值，从0.80降低到0.78进一步提高召回率
    min_time_gap: float = 0.5  # 最小时间间隔，从1.0降低到0.5秒以提高快速帧检测

    # 拓扑地图
    max_nodes: int = 1000
    node_merge_threshold: float = 0.90

    # 路线记忆
    keyframe_interval: int = 8
    max_memory_routes: int = 100

    # 环视融合 (仅使用 camera_1~4)
    use_surround_cameras: bool = True
    surround_weight: float = 0.25  # 每个环视相机权重 (4×0.25=1.0)

    # 持久化
    memory_save_path: str = str(project_root / "deploy/logs/memory_data/")
    auto_save_interval: int = 300

    # 可视化结果保存
    save_visualization: bool = True
    visualization_save_path: str = str(project_root / "deploy/logs/visualization/")

    # VLM 场景描述生成器 (Qwen2.5-VL-7B-Instruct)
    vlm_enabled: bool = True
    vlm_model_path: str = "/home/ubuntu/Disk/models/vlm/Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_max_new_tokens: int = 256
    vlm_batch_size: int = 4
    vlm_version: str = "v3"  # 场景描述生成器版本: "v2" (旧版) 或 "v3" (新版，更好区分度)

    # GraphRAG 语义地图
    graphrag_enabled: bool = True
    semantic_index_path: str = str(project_root / "deploy/logs/semantic_index/")
