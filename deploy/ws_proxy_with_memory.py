#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternNav InternVLA-N1 WebSocket代理服务 - 增强版 (带视觉记忆功能)

新增功能:
1. 路线记忆: 自动记录导航轨迹
2. 位置识别: 识别已访问位置 (回环检测)
3. 返回导航: 支持"返回起点"任务指令
4. 环视融合: 利用多相机增强位置识别
5. 持久化存储: 记忆跨会话保存

基于InternVLAN1AsyncAgent提供实时导航推理服务
"""

import asyncio
import websockets
import json
import logging
import logging.handlers
import base64
import io
import os
import sys
import time
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

import torch
from torchvision.transforms import ToPILImage

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
from internnav.model.basemodel.LongCLIP.model import longclip

# ============================================================================
# 导入模块化组件
# ============================================================================
from memory_modules import (
    MemoryNavigationConfig,
    TopologicalNode,
    RouteMemory,
    LongCLIPFeatureExtractor,
    SceneDescriptionGenerator,
    VisualPlaceRecognition,
    SemanticGraphManager,
    TopologicalMapManager,
    RouteMemoryManager,
    SurroundCameraFusion,
    ReturnNavigator,
    decode_base64_image,
    decode_base64_depth,
    encode_numpy_to_base64,
    convert_output_action_to_robot_action,
    convert_trajectory_to_robot_action,
)

# 尝试导入FAISS (用于高效相似度搜索)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using numpy-based similarity search (slower).")

# 导入数据库模块
try:
    from memory_modules.database import TopologyDatabase, get_database
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    logging.warning(f"Database module not available: {e}")

# 尝试导入networkx (用于拓扑图)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Topological graph features disabled.")


# ============================================================================
# 日志配置
# ============================================================================

# 使用绝对路径确保日志目录正确
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_SCRIPT_DIR, 'logs')
LOG_FILE = "ws_proxy_memory.log"

# 用于存储日志文件路径，供外部查询
_LOG_FILE_PATH = None


def setup_logging():
    """配置日志记录，同时输出到控制台和文件"""
    global _LOG_FILE_PATH

    # 创建日志目录
    try:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR, exist_ok=True)
    except Exception as e:
        print(f"[WARNING] 创建日志目录失败: {LOG_DIR}, 错误: {e}")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清理已有的handlers，避免重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)  # 明确使用stdout
    console_handler.setLevel(logging.INFO)

    # 文件handler - 使用绝对路径
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    _LOG_FILE_PATH = log_path  # 保存路径供查询

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10,  # 保留更多备份
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
    except Exception as e:
        print(f"[ERROR] 创建日志文件handler失败: {log_path}, 错误: {e}")
        file_handler = None

    # 格式化器 - 包含更多信息
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if file_handler is not None:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"[INFO] 日志文件保存路径: {log_path}")
    else:
        print(f"[WARNING] 日志仅输出到控制台，文件保存失败")

    return logging.getLogger(__name__)


def get_log_file_path() -> str:
    """获取当前日志文件路径"""
    return _LOG_FILE_PATH


logger = setup_logging()


# ============================================================================
# 注意: 以下核心组件已重构到 memory_modules 包中:
# - MemoryNavigationConfig (配置类)
# - TopologicalNode, RouteMemory (数据模型)
# - LongCLIPFeatureExtractor (视觉特征提取)
# - SceneDescriptionGenerator (VLM场景描述)
# - VisualPlaceRecognition (VPR回环检测)
# - SemanticGraphManager (GraphRAG语义图)
# - TopologicalMapManager (拓扑图管理)
# - RouteMemoryManager (路线记忆管理)
# - SurroundCameraFusion (环视融合)
# - ReturnNavigator (返回导航)
# - decode_base64_image, decode_base64_depth, encode_numpy_to_base64 (工具函数)
# - convert_output_action_to_robot_action, convert_trajectory_to_robot_action (动作转换)
# ============================================================================


# ============================================================================
# 记忆导航代理 (保留在此文件，因为包含服务特定逻辑)
# ============================================================================

class MemoryNavigationAgent:
    """集成记忆功能的导航代理 - 增强版支持VLM和GraphRAG"""

    def __init__(self, config: MemoryNavigationConfig):
        self.config = config

        # LongCLIP 视觉特征提取器
        self.feature_extractor = LongCLIPFeatureExtractor(
            model_path=config.longclip_model_path,
            device=config.feature_extractor_device,
            feature_dim=config.feature_dim
        )

        # VLM 场景描述生成器 (Qwen2.5-VL)
        # 方位感知 + 智能命名 + 抗幻觉
        self.scene_generator = SceneDescriptionGenerator(config)
        logger.info("使用场景描述生成器 (方位感知+智能命名+抗幻觉)")

        # 核心模块
        self.topo_map = TopologicalMapManager(config)
        self.route_memory = RouteMemoryManager(config)
        self.return_navigator = ReturnNavigator(self.topo_map, self.route_memory)
        self.surround_fusion = SurroundCameraFusion(config)

        # 状态
        self.last_action: List[int] = []
        self.is_active = True

        # 记忆复用状态
        self.replay_route: Optional[RouteMemory] = None  # 当前正在复用的路线
        self.replay_step: int = 0  # 当前复用步骤
        self.replay_mode: bool = False  # 是否处于记忆复用模式

        # v1.1: 周期性关键帧计数器 - 每N帧强制创建一个关键帧用于语义分析
        self.frame_count_since_last_keyframe = 0
        self.periodic_keyframe_interval = config.keyframe_interval  # 使用配置中的间隔

        # 尝试加载已保存的记忆数据
        self._load_saved_memory_data()

        logger.info("MemoryNavigationAgent初始化完成 (使用LongCLIP+VLM+GraphRAG)")

    def extract_visual_feature(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        使用 LongCLIP 提取视觉特征

        Args:
            rgb_image: RGB图像 [H, W, 3]

        Returns:
            feature: 归一化特征向量 [768]
        """
        return self.feature_extractor.extract_feature(rgb_image)

    def process_observation(self,
                           rgb_image: np.ndarray,
                           surround_images: Dict[str, np.ndarray] = None,
                           action: List[int] = None,
                           instruction: str = None,
                           pixel_target: List[float] = None,
                           source_timestamp: str = None) -> Dict:
        """
        处理新观测 - 增强版支持VLM场景描述

        VPR 仅使用 camera_1~4 四个环视相机的融合特征，
        不包含 front_1 前置相机。

        Args:
            rgb_image: front_1 前置相机图像 (用于存储，不用于VPR)
            surround_images: {camera_1~4: image} 环视相机图像
            action: 当前执行的动作
            instruction: 当前任务指令
            pixel_target: 像素目标坐标，不为None时表示当前帧是关键帧
            source_timestamp: v2.1 来源图片时间戳，用于追踪节点与原始图片的对应关系

        Returns:
            memory_info: 记忆相关信息
        """
        # 提取环视相机特征 (camera_1~4)
        # 注意: 前置相机图像(rgb_image)仅用于存储，不参与VPR特征提取
        surround_features = {}
        if surround_images:
            for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                if cam_id in surround_images and surround_images[cam_id] is not None:
                    surround_features[cam_id] = self.extract_visual_feature(surround_images[cam_id])

        # 融合环视相机特征 (仅使用camera_1~4，不使用front_1)
        fused_feature = self.surround_fusion.fuse_features(surround_features)

        # 如果没有环视特征，跳过VPR处理（不使用前置相机作为回退）
        if fused_feature is None:
            logger.warning("无环视相机特征，跳过本帧的VPR处理 (前置相机不参与记忆模块)")
            # 返回空的记忆信息，不进行VPR检测
            return {
                "node_id": None,
                "is_new_node": False,
                "is_revisited": False,
                "revisit_similarity": None,
                "revisit_node_id": None,
                "topo_stats": self.topo_map.get_stats(),
                "route_progress": self.route_memory.get_current_progress(),
                "return_available": self.route_memory.get_start_node() is not None,
                "is_keyframe": False,
                "scene_description": None,
                "semantic_labels": [],
                "semantic_graph_stats": {
                    "total_semantic_nodes": len(self.topo_map.semantic_graph.node_metadata),
                    "total_labels": len(self.topo_map.semantic_graph.label_index)
                },
                "skipped_no_surround": True
            }

        # v1.1: 增强关键帧判断逻辑 - 支持周期性关键帧
        self.frame_count_since_last_keyframe += 1
        is_pixel_target_keyframe = pixel_target is not None
        is_periodic_keyframe = (self.frame_count_since_last_keyframe >= self.periodic_keyframe_interval)
        is_keyframe = is_pixel_target_keyframe or is_periodic_keyframe

        if is_keyframe:
            self.frame_count_since_last_keyframe = 0  # 重置计数器
            if is_periodic_keyframe and not is_pixel_target_keyframe:
                logger.info(f"[VLM] 周期性关键帧检测到 (每{self.periodic_keyframe_interval}帧)")
            else:
                logger.info(f"[VLM] pixel_target关键帧检测到")

        # 对于关键帧，使用VLM生成场景描述、语义标签和节点名称
        scene_description = None
        semantic_labels = []
        node_name = None
        if is_keyframe and self.config.vlm_enabled and surround_images:
            logger.info(f"[VLM] 关键帧检测到，开始生成完整场景信息...")
            # v2.0: 使用新的完整场景信息生成方法
            scene_description, semantic_labels, node_name = self.scene_generator.generate_complete_scene_info(surround_images)
            logger.info(f"[VLM] 场景描述: {scene_description[:50] if scene_description else 'None'}...")
            logger.info(f"[VLM] 语义标签: {semantic_labels}")
            logger.info(f"[VLM] 节点名称: {node_name}")

        # v2.0: 提取前视图特征（用于记忆检索）
        # 注意: front_1 是 rgb_image 参数（前置相机），不是 camera_1（环视相机）
        front_view_feature = None
        if rgb_image is not None:
            front_view_feature = self.extract_visual_feature(rgb_image)  # 从前置相机提取特征

        # 添加到拓扑图 (包含语义信息) - v2.1增强（含节点来源追踪）
        node_id, is_new_node, revisit_info = self.topo_map.add_observation(
            visual_feature=fused_feature,
            rgb_image=rgb_image,
            surround_images=surround_images,
            action_from_prev=self.last_action if self.last_action else action,
            instruction=instruction,
            is_keyframe=is_keyframe,
            pixel_target=pixel_target,
            scene_description=scene_description,
            semantic_labels=semantic_labels,
            # v2.0 新增参数
            node_name=node_name,
            navigation_instruction=instruction,
            front_view_feature=front_view_feature,
            # v2.1 节点来源追踪
            source_timestamp=source_timestamp
        )

        # v2.0: 如果强制创建了新节点但没有语义信息，补充生成
        if is_new_node and not is_keyframe and not scene_description and self.config.vlm_enabled and surround_images:
            logger.info(f"[VLM] 新节点 {node_id} 缺少语义信息，开始补充生成...")
            # v2.0: 使用完整场景信息生成方法
            scene_description, semantic_labels, node_name = self.scene_generator.generate_complete_scene_info(surround_images)
            logger.info(f"[VLM] 补充场景描述: {scene_description[:50] if scene_description else 'None'}...")
            logger.info(f"[VLM] 补充语义标签: {semantic_labels}")
            logger.info(f"[VLM] 补充节点名称: {node_name}")
            # 更新语义图
            if scene_description or semantic_labels:
                self.topo_map.semantic_graph.add_semantic_node(
                    node_id=node_id,
                    scene_description=scene_description or "",
                    semantic_labels=semantic_labels or [],
                    visual_feature=fused_feature,
                    # v2.0 新增参数
                    node_name=node_name,
                    navigation_instruction=instruction,
                    front_view_feature=front_view_feature,
                    pixel_target=pixel_target
                )
                # 同时更新节点信息
                if node_id in self.topo_map.nodes:
                    self.topo_map.nodes[node_id].scene_description = scene_description
                    self.topo_map.nodes[node_id].semantic_labels = semantic_labels or []
                    self.topo_map.nodes[node_id].node_name = node_name
                    self.topo_map.nodes[node_id].navigation_instruction = instruction
                    self.topo_map.nodes[node_id].front_view_feature = front_view_feature
                logger.info(f"[GraphRAG] 节点 {node_id} 语义信息已补充到语义图")

        # 记录到路线 - 基于pixel_target判断是否为关键帧
        if self.route_memory.is_recording():
            self.route_memory.record_step(
                node_id=node_id,
                visual_feature=fused_feature,
                action=action or [],
                rgb_image=rgb_image,
                is_keyframe=is_keyframe
            )

        # 更新动作历史
        self.last_action = action or []

        # 构建返回信息 - 包含语义信息 v2.0增强
        memory_info = {
            "node_id": node_id,
            "is_new_node": is_new_node,
            "is_revisited": revisit_info is not None,
            "revisit_similarity": revisit_info[1] if revisit_info else None,
            "revisit_node_id": revisit_info[0] if revisit_info else None,
            "topo_stats": self.topo_map.get_stats(),
            "route_progress": self.route_memory.get_current_progress(),
            "return_available": self.route_memory.get_start_node() is not None,
            "is_keyframe": is_keyframe,
            # 语义信息 v1.0
            "scene_description": scene_description,
            "semantic_labels": semantic_labels,
            "semantic_graph_stats": {
                "total_semantic_nodes": len(self.topo_map.semantic_graph.node_metadata),
                "total_labels": len(self.topo_map.semantic_graph.label_index)
            },
            # v2.0 新增字段
            "node_name": node_name,
            "navigation_instruction": instruction,
            "has_front_view_feature": front_view_feature is not None,
            "pixel_target": pixel_target
        }

        return memory_info

    def start_memory_recording(self, instruction: str) -> str:
        """开始记忆记录"""
        return self.route_memory.start_recording(instruction)

    def stop_memory_recording(self) -> Optional[RouteMemory]:
        """停止记忆记录，同时保存语义图数据到磁盘和数据库"""
        route = self.route_memory.stop_recording()
        # 保存语义图数据到磁盘
        if route is not None:
            save_path = os.path.join(project_root, self.config.memory_save_path)
            self.topo_map.semantic_graph.save_to_disk(save_path)

            # v3.0: 同时保存到数据库
            if DATABASE_AVAILABLE:
                try:
                    self._sync_to_database()
                    logger.info("[Memory] 拓扑图已同步到数据库")
                except Exception as e:
                    logger.warning(f"[Memory] 数据库同步失败: {e}")
        return route

    def _sync_to_database(self):
        """同步拓扑图数据到PostgreSQL数据库"""
        if not DATABASE_AVAILABLE:
            return

        db = get_database()

        # 清空数据库中的旧数据
        db.clear_all()

        # 同步节点
        for node_id, node in self.topo_map.nodes.items():
            node_data = {
                'node_id': node_id,
                'node_name': getattr(node, 'node_name', None),
                'scene_description': getattr(node, 'scene_description', None),
                'semantic_labels': getattr(node, 'semantic_labels', []),
                'navigation_instruction': getattr(node, 'navigation_instruction', None),
                'pixel_target': getattr(node, 'pixel_target', None),
                'pixel_target_history': getattr(node, 'pixel_target_history', []),
                'visual_feature': getattr(node, 'visual_feature', None),
                'front_view_feature': getattr(node, 'front_view_feature', None),
                'timestamp': getattr(node, 'timestamp', None),
                'created_at': getattr(node, 'created_at', None),
                'visit_count': getattr(node, 'visit_count', 1),
                'is_keyframe': bool(getattr(node, 'scene_description', None)),
                'source_timestamps': getattr(node, 'source_timestamps', [])
            }
            db.add_node(node_data)

        # 同步边
        if hasattr(self.topo_map, 'semantic_graph') and self.topo_map.semantic_graph.semantic_graph:
            for source, target, data in self.topo_map.semantic_graph.semantic_graph.edges(data=True):
                db.add_edge(
                    source_id=source,
                    target_id=target,
                    action=data.get('action', []),
                    weight=data.get('weight', 1.0),
                    description=data.get('description', '')
                )

        logger.info(f"[Memory] 数据库同步完成: {db.get_node_count()} 节点, {db.get_edge_count()} 边")

    def start_return_navigation(self) -> bool:
        """开始返回导航"""
        return self.return_navigator.start_return()

    def get_return_action(self) -> Tuple[Optional[List[int]], bool]:
        """获取返回导航动作"""
        return self.return_navigator.get_next_return_action()

    def is_returning(self) -> bool:
        """是否正在返回导航"""
        return self.return_navigator.is_returning

    def reset(self):
        """重置状态"""
        self.topo_map.reset()
        self.route_memory.current_route = None
        self.return_navigator.stop_return()
        self.last_action = []
        # 重置记忆复用状态
        self.replay_route = None
        self.replay_step = 0
        self.replay_mode = False
        # v1.1: 重置周期性关键帧计数器
        self.frame_count_since_last_keyframe = 0
        logger.info("MemoryNavigationAgent已重置")

    def check_memory_replay(self, instruction: str) -> Tuple[bool, Optional[RouteMemory]]:
        """
        检查是否可以使用记忆复用

        Args:
            instruction: 导航指令

        Returns:
            (can_replay, matched_route): 是否可以复用，匹配的路线
        """
        if not self.config.memory_enabled:
            return False, None

        matched_route = self.route_memory.find_matching_route(instruction)
        if matched_route is not None and matched_route.is_complete:
            logger.info(f"[记忆复用] 找到匹配路线: {matched_route.route_id}, "
                       f"指令: '{instruction}', "
                       f"步骤数: {len(matched_route.action_history)}")
            return True, matched_route
        return False, None

    def start_memory_replay(self, route: RouteMemory) -> bool:
        """
        开始记忆复用导航

        Args:
            route: 要复用的路线

        Returns:
            是否成功开始复用
        """
        if route is None or not route.is_complete:
            return False

        self.replay_route = route
        self.replay_step = 0
        self.replay_mode = True
        logger.info(f"[记忆复用] 开始复用路线: {route.route_id}, 总步骤: {len(route.action_history)}")
        return True

    def get_replay_action(self) -> Tuple[Optional[List[int]], bool, int, int]:
        """
        获取记忆复用的下一个动作

        Returns:
            (action, is_complete, current_step, total_steps):
            - action: 动作序列
            - is_complete: 是否完成复用
            - current_step: 当前步骤
            - total_steps: 总步骤数
        """
        if not self.replay_mode or self.replay_route is None:
            return None, True, 0, 0

        total_steps = len(self.replay_route.action_history)

        if self.replay_step >= total_steps:
            # 复用完成
            self.stop_memory_replay()
            logger.info(f"[记忆复用] 路线复用完成")
            return [0], True, total_steps, total_steps  # 返回STOP动作

        action = self.replay_route.action_history[self.replay_step]
        current_step = self.replay_step
        self.replay_step += 1

        is_complete = self.replay_step >= total_steps
        if is_complete:
            self.stop_memory_replay()
            logger.info(f"[记忆复用] 路线复用完成")

        logger.info(f"[记忆复用] 步骤 {current_step + 1}/{total_steps}, 动作: {action}")
        return action, is_complete, current_step + 1, total_steps

    def stop_memory_replay(self):
        """停止记忆复用"""
        if self.replay_mode:
            logger.info(f"[记忆复用] 停止复用，已执行步骤: {self.replay_step}")
        self.replay_route = None
        self.replay_step = 0
        self.replay_mode = False

    def is_replaying(self) -> bool:
        """是否正在记忆复用"""
        return self.replay_mode

    def _load_saved_memory_data(self):
        """加载已保存的记忆数据（语义图和VPR索引）

        安全说明: 此方法中使用pickle加载的文件仅来自系统自身生成的内部数据。
        - 保存路径由config.memory_save_path指定(默认: deploy/logs/memory_data/)
        - 仅加载系统在导航过程中自动生成的.pkl文件
        - 不接受任何外部来源的pickle文件输入
        """
        save_path = self.config.memory_save_path
        if not os.path.exists(save_path):
            logger.info(f"记忆数据目录不存在: {save_path}")
            return

        try:
            # pickle用于加载系统内部生成的路线数据（见route_memory.py中的save_route方法）
            import pickle  # nosec B403 - 仅加载系统内部生成的pkl文件
            import networkx as nx

            # 1. 加载语义图
            semantic_graph_path = os.path.join(save_path, 'semantic_graph.json')
            semantic_metadata_path = os.path.join(save_path, 'semantic_metadata.json')

            if os.path.exists(semantic_graph_path):
                with open(semantic_graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)

                # 加载到语义图管理器
                self.topo_map.semantic_graph.semantic_graph = nx.node_link_graph(graph_data)
                logger.info(f"[记忆加载] 语义图已加载: {len(graph_data.get('nodes', []))} 个节点")

                # 从图数据中提取节点信息到拓扑图
                for node_data in graph_data.get('nodes', []):
                    node_id = node_data.get('id')
                    if node_id is not None:
                        # 创建节点占位符
                        from memory_modules.models import TopologicalNode
                        if node_id not in self.topo_map.nodes:
                            # 创建简单的节点对象
                            node = TopologicalNode(
                                node_id=node_id,
                                visual_feature=np.zeros(self.config.feature_dim),
                                timestamp=time.time()
                            )
                            node.scene_description = node_data.get('description', '')
                            node.semantic_labels = node_data.get('labels', [])
                            self.topo_map.nodes[node_id] = node

                            # 添加到networkx图
                            if self.topo_map.graph is not None:
                                self.topo_map.graph.add_node(node_id)

                # 加载边
                for edge_data in graph_data.get('links', []):
                    source = edge_data.get('source')
                    target = edge_data.get('target')
                    if source is not None and target is not None and self.topo_map.graph is not None:
                        self.topo_map.graph.add_edge(source, target, weight=1.0)

                self.topo_map.next_node_id = max(self.topo_map.nodes.keys(), default=-1) + 1
                logger.info(f"[记忆加载] 拓扑图节点已加载: {len(self.topo_map.nodes)} 个节点")

            # 2. 加载语义元数据
            if os.path.exists(semantic_metadata_path):
                with open(semantic_metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                self.topo_map.semantic_graph.node_metadata = {
                    int(k): v for k, v in metadata.get('node_metadata', {}).items()
                }
                self.topo_map.semantic_graph.label_index = metadata.get('label_index', {})
                self.topo_map.semantic_graph.description_index = {
                    int(k): v for k, v in metadata.get('description_index', {}).items()
                }
                logger.info(f"[记忆加载] 语义元数据已加载: {len(self.topo_map.semantic_graph.label_index)} 个标签索引")

            # 3. 加载VPR特征索引
            # 查找特征文件 (仅加载系统生成的.npy文件)
            feature_files = [f for f in os.listdir(save_path) if f.endswith('_features.npy')]
            for feature_file in feature_files:
                feature_path = os.path.join(save_path, feature_file)
                try:
                    features = np.load(feature_path)
                    logger.info(f"[记忆加载] 加载特征文件: {feature_file}, 形状: {features.shape}")

                    # 对应的pkl文件 (系统内部生成的路线数据)
                    route_id = feature_file.replace('_features.npy', '')
                    pkl_path = os.path.join(save_path, f"{route_id}.pkl")

                    node_sequence = None
                    if os.path.exists(pkl_path):
                        # 加载系统自身生成的pickle文件 - 见route_memory.py的save_route()
                        with open(pkl_path, 'rb') as f:
                            route_data = pickle.load(f)  # nosec B301 - 内部生成的数据
                        node_sequence = route_data.get('node_sequence', [])
                        logger.info(f"[记忆加载] 加载路线数据: {route_id}, 节点序列长度: {len(node_sequence)}")

                    # 将特征添加到VPR索引
                    if node_sequence and len(node_sequence) == features.shape[0]:
                        for i, node_id in enumerate(node_sequence):
                            feature = features[i]
                            self.topo_map.vpr.add_feature(
                                feature=feature,
                                node_id=node_id,
                                timestamp=time.time()
                            )
                            if node_id in self.topo_map.nodes:
                                self.topo_map.nodes[node_id].visual_feature = feature
                    else:
                        # 没有节点序列，按顺序添加
                        for i, feature in enumerate(features):
                            node_id = i % len(self.topo_map.nodes) if self.topo_map.nodes else i
                            self.topo_map.vpr.add_feature(
                                feature=feature,
                                node_id=node_id,
                                timestamp=time.time()
                            )

                    logger.info(f"[记忆加载] VPR索引已重建: {self.topo_map.vpr.index.ntotal} 个特征")

                except Exception as e:
                    logger.warning(f"[记忆加载] 加载特征文件失败 {feature_file}: {e}")

            logger.info("[记忆加载] 记忆数据加载完成")

        except Exception as e:
            logger.error(f"[记忆加载] 加载记忆数据失败: {e}", exc_info=True)


# ============================================================================
# 原有ws_proxy功能移植
# ============================================================================

class Args:
    """InternVLAN1AsyncAgent初始化参数"""
    def __init__(self, device="cuda:0"):
        self.device = device  # 可配置的GPU设备
        self.model_path = str(project_root / "checkpoints/InternRobotics/InternVLA-N1-DualVLN")
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 8
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.plan_step_gap = 8


def annotate_image(idx, image, instruction, output_action, trajectory, pixel_goal, output_dir):
    """
    在图像上标注推理结果，包括指令、动作、轨迹和像素目标

    Args:
        idx: 帧ID或时间戳
        image: 输入图像 (H, W, 3) numpy array
        instruction: 导航指令
        output_action: 动作序列列表
        trajectory: 轨迹数组
        pixel_goal: 像素目标 [y, x]
        output_dir: 输出目录

    Returns:
        标注后的图像 numpy array
    """
    try:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        # 使用默认字体（避免字体文件不存在的问题）
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except:
            font = ImageFont.load_default()

        # 构建文本内容
        text_content = []
        text_content.append(f"Frame/PTS: {idx}")
        if output_action:
            action_map = {0: 'STOP', 1: '↑', 2: '←', 3: '→', 5: '↓'}
            action_str = ''.join([action_map.get(a, str(a)) for a in output_action[:10]])
            text_content.append(f"Actions: {action_str}")

        # 计算文本框大小
        max_width = 0
        total_height = 0
        for line in text_content:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(line) * 8  # 估算宽度
            text_height = 20
            max_width = max(max_width, text_width)
            total_height += text_height

        # 绘制文本框背景
        padding = 10
        box_x, box_y = 10, 10
        box_width = max_width + 2 * padding
        box_height = total_height + 2 * padding

        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

        # 绘制文本
        text_color = 'white'
        y_position = box_y + padding

        for line in text_content:
            draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
            y_position += 20

        image = np.array(image)

        # 绘制轨迹可视化（右上角）
        if trajectory is not None and len(trajectory) > 0:
            img_height, img_width = image.shape[:2]

            # 窗口参数
            window_size = 200
            window_margin = 0
            window_x = img_width - window_size - window_margin
            window_y = window_margin

            # 提取轨迹点
            traj_points = []
            for point in trajectory:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    traj_points.append([float(point[0]), float(point[1])])

            if len(traj_points) > 0:
                traj_array = np.array(traj_points)
                x_coords = traj_array[:, 0]
                y_coords = traj_array[:, 1]

                # 创建matplotlib图形
                fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
                fig.patch.set_alpha(0.6)
                fig.patch.set_facecolor('gray')
                ax.set_facecolor('lightgray')

                # 绘制轨迹
                ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')

                # 标记起点（绿色）和终点（红色）
                ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
                ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')

                # 标记原点
                ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')

                # 设置坐标轴
                ax.set_xlabel('Y (left +)', fontsize=8)
                ax.set_ylabel('X (up +)', fontsize=8)
                ax.invert_xaxis()
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(fontsize=6, loc='upper right')

                plt.tight_layout(pad=0.3)

                # 转换为numpy数组
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                # 调整大小并叠加到图像上
                plot_img = cv2.resize(plot_img, (window_size, window_size))
                image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img

        # 绘制像素目标（蓝色圆圈）
        if pixel_goal is not None and len(pixel_goal) >= 2:
            # pixel_goal是[y, x]格式，cv2.circle需要(x, y)格式
            cv2.circle(image, (int(pixel_goal[1]), int(pixel_goal[0])), 5, (255, 0, 0), -1)

        # 保存标注后的图像
        image_pil = Image.fromarray(image).convert('RGB')
        output_path = os.path.join(output_dir, f'annotated_{idx}.jpg')
        image_pil.save(output_path)
        logger.info(f"已保存标注图像: {output_path}")

        return image

    except Exception as e:
        logger.error(f"图像标注失败: {e}", exc_info=True)
        return image if isinstance(image, np.ndarray) else np.array(image)


def save_memory_visualization(
    image: np.ndarray,
    instruction: str,
    step_idx: int,
    output_action: List[int],
    pixel_target: Optional[List[float]],
    memory_mode: str,  # "recording", "replay", "inference", "disabled"
    memory_info: Dict,
    config: MemoryNavigationConfig,
    session_id: str = None,
    surround_images: Dict[str, np.ndarray] = None  # 新增：环视相机图像
):
    """
    保存记忆导航可视化结果 - 增强版支持环视图和语义信息

    Args:
        image: RGB图像 (front_1)
        instruction: 导航指令
        step_idx: 步骤索引
        output_action: 输出动作序列
        pixel_target: 像素目标
        memory_mode: 记忆模式 ("recording", "replay", "inference", "disabled")
        memory_info: 记忆相关信息
        config: 配置对象
        session_id: 会话ID（用于区分不同导航会话）
        surround_images: 环视相机图像字典 {camera_1~4: image}
    """
    if not config.save_visualization:
        return

    try:
        # 创建可视化保存目录
        viz_dir = config.visualization_save_path
        if session_id:
            viz_dir = os.path.join(viz_dir, session_id)
        os.makedirs(viz_dir, exist_ok=True)

        # 创建带有记忆状态信息的标注图像
        image_pil = Image.fromarray(image).convert('RGB')
        draw = ImageDraw.Draw(image_pil)

        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
            font_small = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font

        # 准备标注信息
        mode_colors = {
            "recording": (0, 255, 0),    # 绿色 - 记录模式
            "replay": (0, 128, 255),     # 蓝色 - 复用模式
            "inference": (255, 255, 0),  # 黄色 - 推理模式
            "disabled": (128, 128, 128)  # 灰色 - 记忆关闭
        }
        mode_labels = {
            "recording": "REC",
            "replay": "REPLAY",
            "inference": "INFER",
            "disabled": "MEM_OFF"
        }

        mode_color = mode_colors.get(memory_mode, (255, 255, 255))
        mode_label = mode_labels.get(memory_mode, memory_mode)

        # 绘制模式标签（左上角）
        draw.rectangle([5, 5, 100, 28], fill=(0, 0, 0, 180))
        draw.text((10, 8), mode_label, fill=mode_color, font=font)

        # 绘制步骤信息
        step_text = f"Step: {step_idx}"
        draw.text((10, 32), step_text, fill='white', font=font_small)

        # 绘制动作信息
        if output_action:
            action_map = {0: 'STOP', 1: 'F', 2: 'L', 3: 'R', 5: 'D'}
            action_str = ''.join([action_map.get(a, str(a)) for a in output_action[:8]])
            draw.text((10, 47), f"Act: {action_str}", fill='white', font=font_small)

        # 绘制记忆信息
        y_offset = 62
        if memory_mode == "replay":
            replay_step = memory_info.get('replay_step', 0)
            replay_total = memory_info.get('replay_total', 0)
            draw.text((10, y_offset), f"Replay: {replay_step}/{replay_total}", fill=mode_color, font=font_small)
            y_offset += 15
            if memory_info.get('skipped_inference'):
                draw.text((10, y_offset), "SKIP_INF", fill=(0, 255, 128), font=font_small)
        elif memory_mode == "recording":
            if memory_info.get('is_keyframe'):
                draw.text((10, y_offset), "KEYFRAME", fill=(255, 215, 0), font=font_small)
                y_offset += 15
            route_progress = memory_info.get('route_progress', {})
            frames = route_progress.get('frames', 0)
            keyframes = route_progress.get('keyframes', 0)
            draw.text((10, y_offset), f"F:{frames} KF:{keyframes}", fill='white', font=font_small)
            y_offset += 15

            # 显示语义标签（如果有）
            semantic_labels = memory_info.get('semantic_labels', [])
            if semantic_labels:
                labels_text = ','.join(semantic_labels[:3])
                if len(labels_text) > 20:
                    labels_text = labels_text[:17] + "..."
                draw.text((10, y_offset), f"Tags:{labels_text}", fill=(0, 200, 255), font=font_small)

        # 绘制像素目标
        if pixel_target is not None:
            img_width, img_height = image_pil.size
            px = int(pixel_target[0] * img_width)
            py = int(pixel_target[1] * img_height)
            draw.ellipse([px-8, py-8, px+8, py+8], outline=(255, 0, 0), width=3)
            draw.ellipse([px-3, py-3, px+3, py+3], fill=(255, 0, 0))

        # 保存主图像
        filename = f"{step_idx:06d}_{memory_mode}.jpg"
        output_path = os.path.join(viz_dir, filename)
        image_pil.save(output_path, quality=90)
        logger.debug(f"[可视化] 保存主图像: {output_path}")

        # 保存环视图拼接图（仅关键帧）
        if memory_info.get('is_keyframe') and surround_images:
            try:
                surround_dir = os.path.join(viz_dir, 'surround')
                os.makedirs(surround_dir, exist_ok=True)

                # 创建2x2拼接图
                cam_images = []
                for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
                    if cam_id in surround_images and surround_images[cam_id] is not None:
                        img = surround_images[cam_id]
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        # 缩放到统一大小
                        img_resized = cv2.resize(img, (320, 240))
                        cam_images.append(img_resized)
                    else:
                        cam_images.append(np.zeros((240, 320, 3), dtype=np.uint8))

                if len(cam_images) == 4:
                    # 拼接成2x2
                    top_row = np.hstack([cam_images[0], cam_images[1]])
                    bottom_row = np.hstack([cam_images[2], cam_images[3]])
                    surround_combined = np.vstack([top_row, bottom_row])

                    # 添加相机标签
                    surround_pil = Image.fromarray(surround_combined)
                    surround_draw = ImageDraw.Draw(surround_pil)
                    labels_pos = [(10, 10), (330, 10), (10, 250), (330, 250)]
                    cam_labels = ['cam1(FR)', 'cam2(FL)', 'cam3(BL)', 'cam4(BR)']
                    for pos, label in zip(labels_pos, cam_labels):
                        surround_draw.text(pos, label, fill=(255, 255, 0), font=font_small)

                    # 保存环视图
                    surround_path = os.path.join(surround_dir, f"{step_idx:06d}_surround.jpg")
                    surround_pil.save(surround_path, quality=85)
                    logger.debug(f"[可视化] 保存环视图: {surround_path}")

            except Exception as e:
                logger.warning(f"保存环视图失败: {e}")

        # 保存元数据（JSON）- 包含语义信息
        metadata = {
            "step_idx": step_idx,
            "instruction": instruction,
            "memory_mode": memory_mode,
            "output_action": output_action,
            "pixel_target": pixel_target,
            "memory_info": {
                k: v for k, v in memory_info.items()
                if not isinstance(v, np.ndarray) and not callable(v)
            },
            "timestamp": time.time()
        }
        metadata_path = os.path.join(viz_dir, f"{step_idx:06d}_{memory_mode}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.warning(f"保存可视化结果失败: {e}")


# ============================================================================
# WebSocket服务器
# ============================================================================

connected_clients = {}
global_agent = None
global_memory_agent = None
agent_lock = asyncio.Lock()

# 全局会话状态 - 用于在多连接间共享task信息
# 解决问题：当新连接（如可视化前端的第二个连接）发送task="None"时，
# 可以继承之前连接设置的task
global_session_state = {
    'last_task': None,
    'last_instruction': None,
    'total_request_count': 0
}

# 全局配置
memory_config = MemoryNavigationConfig()
main_model_device = "cuda:0"  # 主模型设备，在main()中根据GPU配置设置


def init_agent(model_path=None, device="cuda:0"):
    """初始化InternVLAN1AsyncAgent"""
    args = Args(device=device)
    if model_path:
        args.model_path = model_path

    logger.info(f"正在加载模型: {args.model_path}")
    logger.info(f"使用设备: {args.device}")

    agent = InternVLAN1AsyncAgent(args)

    logger.info("正在预热模型...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4)
    agent.reset()
    agent.step(dummy_rgb, dummy_depth, dummy_pose, "test", intrinsic=args.camera_intrinsic)
    logger.info("模型加载完成！")

    return agent


async def process_inference_with_memory(message_data, session_state, agent, memory_agent, memory_enabled=True):
    """处理推理请求 (带记忆功能)

    Args:
        message_data: 消息数据
        session_state: 会话状态
        agent: InternVLAN1AsyncAgent实例
        memory_agent: MemoryNavigationAgent实例
        memory_enabled: 记忆功能开关，False时行为与ws_proxy.py一致
    """
    try:
        logger.info(f"开始处理推理请求 (memory_enabled={memory_enabled})")

        # 打印请求JSON（不包含base64图像数据）
        request_log = {k: v for k, v in message_data.items() if k != 'images'}
        if 'images' in message_data:
            images_log = {}
            for img_key, img_val in message_data['images'].items():
                images_log[img_key] = f"<base64 data, length={len(img_val) if img_val else 0}>"
            request_log['images'] = images_log
        logger.info(f"📥 请求JSON: {json.dumps(request_log, ensure_ascii=False, indent=2)}")

        robot_id = message_data.get('id', None)
        pts = int(message_data['pts']) if 'pts' in message_data else None

        # 验证字段
        if 'task' not in message_data:
            return {
                "status": "error",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "缺少必要字段: task"
            }

        if 'images' not in message_data or 'front_1' not in message_data.get('images', {}):
            return {
                "status": "error",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "缺少必要字段: images.front_1"
            }

        instruction = message_data['task']

        # 解码前置相机图像
        rgb_base64 = message_data['images']['front_1']
        rgb = decode_base64_image(rgb_base64)
        if rgb is None:
            return {
                "status": "error",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "RGB图像(images.front_1)解码失败"
            }

        # 打印原始图像信息
        logger.info(f"📸 输入RGB图像: 原始尺寸={rgb.shape}, 数据类型={rgb.dtype}, base64长度={len(rgb_base64)} bytes")

        # 调整尺寸
        target_width, target_height = 640, 480
        if rgb.shape[1] != target_width or rgb.shape[0] != target_height:
            logger.info(f"📐 输入图像尺寸 {rgb.shape[1]}x{rgb.shape[0]} != {target_width}x{target_height}，进行调整")
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"✅ 图像已调整为 {target_width}x{target_height}")
        else:
            logger.info(f"✅ 图像尺寸已符合要求: {target_width}x{target_height}")

        # 创建图像保存目录
        images_dir = os.path.join(LOG_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # 保存输入RGB图像（调整后的 640x480）
        timestamp_str = f"{pts}" if pts is not None else f"{int(time.time() * 1000)}"
        input_image_path = os.path.join(images_dir, f"{timestamp_str}_input.jpg")
        try:
            Image.fromarray(rgb).save(input_image_path)
            logger.info(f"💾 保存输入图像: {input_image_path} (尺寸: {rgb.shape[1]}x{rgb.shape[0]})")
        except Exception as e:
            logger.warning(f"保存输入图像失败: {e}")

        # 解码并保存环视相机图像
        surround_images = {}
        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            if cam_id in message_data.get('images', {}):
                cam_data = message_data['images'][cam_id]
                if cam_data:
                    cam_img = decode_base64_image(cam_data)
                    if cam_img is not None:
                        surround_images[cam_id] = cv2.resize(cam_img, (target_width, target_height))
                        # 保存环视相机图像
                        camera_image_path = os.path.join(images_dir, f"{timestamp_str}_{cam_id}.jpg")
                        try:
                            Image.fromarray(cam_img).save(camera_image_path)
                            logger.info(f"💾 保存环视相机图片: {camera_image_path}")
                        except Exception as e:
                            logger.warning(f"保存 {cam_id} 图片失败: {e}")
                    else:
                        logger.warning(f"{cam_id} 图片解码失败，跳过保存")

        # ===== 特殊指令处理 (仅当记忆功能开启时) =====

        if memory_enabled:
            # 开始记忆
            if instruction in ["START_MEMORY", "start_memory"]:
                original_instruction = message_data.get('original_instruction', 'default_task')
                route_id = memory_agent.start_memory_recording(original_instruction)
                return {
                    "status": "success",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": "executing",
                    "action": [[0.0, 0.0, 0.0]],
                    "message": f"开始记录路线: {route_id}",
                    "memory_info": {"route_id": route_id, "recording": True}
                }

            # 停止记忆
            if instruction in ["STOP_MEMORY", "stop_memory"]:
                route = memory_agent.stop_memory_recording()
                return {
                    "status": "success",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": "end",
                    "action": [[0.0, 0.0, 0.0]],
                    "message": f"路线记录完成: {route.route_id if route else 'None'}",
                    "memory_info": {"recording": False}
                }

            # 返回起点
            if instruction in ["RETURN", "return", "返回", "返回起点", "go back", "return to start"]:
                success = memory_agent.start_return_navigation()
                if not success:
                    return {
                        "status": "error",
                        "id": robot_id,
                        "pts": pts,
                        "task_status": "end",
                        "action": [[0.0, 0.0, 0.0]],
                        "message": "无法启动返回导航: 没有记录的起点"
                    }

                action, is_complete = memory_agent.get_return_action()
                robot_action, task_status = convert_output_action_to_robot_action(action) if action else ([[0.0, 0.0, 0.0]], "end")

                return {
                    "status": "success",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": "end" if is_complete else "executing",
                    "action": robot_action,
                    "message": "返回导航中" if not is_complete else "返回导航完成",
                    "memory_info": {"returning": not is_complete}
                }

            # 查询记忆状态
            if instruction in ["MEMORY_STATUS", "memory_status"]:
                stats = memory_agent.topo_map.get_stats()
                progress = memory_agent.route_memory.get_current_progress()
                # 导出语义图数据用于可视化
                semantic_graph_data = memory_agent.topo_map.semantic_graph.export_graph_data()
                return {
                    "status": "success",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": "end",
                    "action": [[0.0, 0.0, 0.0]],
                    "message": "记忆状态查询",
                    "memory_info": {
                        "topo_stats": stats,
                        "route_progress": progress,
                        "return_available": memory_agent.route_memory.get_start_node() is not None,
                        "semantic_graph": semantic_graph_data
                    }
                }

        # ===== 正常导航推理 =====

        # 处理task为空 - 支持多连接共享task
        if instruction is None or instruction in ["None", "none", ""]:
            # 优先使用当前连接的last_task
            if session_state.get('last_task') is not None:
                instruction = session_state['last_task']
                logger.info(f"📋 task为空，使用本连接上一次的task: {instruction}")
            # 其次使用全局共享的last_task（支持多连接场景）
            elif global_session_state.get('last_task') is not None:
                instruction = global_session_state['last_task']
                logger.info(f"📋 task为空，使用全局共享的task: {instruction}")
                # 同步到当前连接的session_state
                session_state['last_task'] = instruction
            else:
                # 返回完整的错误响应，包含所有必要字段
                error_response = {
                    "status": "error",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": "end",
                    "action": [[0.0, 0.0, 0.0]],
                    "pixel_target": None,
                    "message": "首次请求时task不能为空，请提供有效的导航指令"
                }
                logger.warning(f"⚠️ 首次请求task为空且无全局task，返回错误: {error_response}")
                return error_response

        # 检测task变化
        if session_state.get('last_task') and instruction != session_state['last_task']:
            logger.info(f"task变化: {session_state['last_task']} -> {instruction}")
            async with agent_lock:
                agent.reset()
            if memory_enabled:
                memory_agent.reset()

        # STOP指令
        if instruction in ["STOP", "stop"]:
            logger.info(f"🛑 检测到STOP指令，直接返回停止动作")
            if memory_enabled:
                memory_agent.stop_memory_recording()

            # 更新session_state
            session_state['request_count'] += 1
            session_state['last_instruction'] = instruction
            session_state['last_task'] = instruction

            response = {
                "status": "success",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "收到STOP指令，任务结束"
            }
            logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")
            return response

        # ===== 记忆复用检查 (仅当记忆功能开启时) =====
        if memory_enabled:
            # 如果已经在复用模式，继续使用记忆动作
            if memory_agent.is_replaying():
                action, is_complete, current_step, total_steps = memory_agent.get_replay_action()
                robot_action, task_status = convert_output_action_to_robot_action(action) if action else ([[0.0, 0.0, 0.0]], "end")

                if is_complete:
                    task_status = "end"

                response = {
                    "status": "success",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": task_status,
                    "action": robot_action,
                    "pixel_target": None,
                    "message": f"[记忆复用] 步骤 {current_step}/{total_steps}",
                    "memory_info": {
                        "replay_mode": True,
                        "replay_step": current_step,
                        "replay_total": total_steps,
                        "replay_complete": is_complete,
                        "skipped_inference": True
                    }
                }

                # 保存可视化结果（记忆复用模式）
                if memory_config.save_visualization:
                    save_memory_visualization(
                        image=rgb,
                        instruction=instruction,
                        step_idx=session_state['request_count'],
                        output_action=action,
                        pixel_target=None,
                        memory_mode="replay",
                        memory_info=response["memory_info"],
                        config=memory_config,
                        session_id=f"replay_{instruction.replace(' ', '_')}"
                    )

                # 更新会话状态
                session_state['request_count'] += 1
                session_state['last_instruction'] = instruction
                session_state['last_task'] = instruction

                logger.info(f"📤 响应JSON (记忆复用): {json.dumps(response, ensure_ascii=False, indent=2)}")
                return response

            # 如果不在复用模式，检查是否可以开始记忆复用
            # 只在任务开始时（request_count为0或task刚刚变化时）检查
            if session_state.get('request_count', 0) == 0 or session_state.get('last_task') != instruction:
                can_replay, matched_route = memory_agent.check_memory_replay(instruction)
                if can_replay and matched_route is not None:
                    # 开始记忆复用
                    memory_agent.start_memory_replay(matched_route)

                    # 获取第一个动作
                    action, is_complete, current_step, total_steps = memory_agent.get_replay_action()
                    robot_action, task_status = convert_output_action_to_robot_action(action) if action else ([[0.0, 0.0, 0.0]], "end")

                    if is_complete:
                        task_status = "end"

                    response = {
                        "status": "success",
                        "id": robot_id,
                        "pts": pts,
                        "task_status": task_status,
                        "action": robot_action,
                        "pixel_target": None,
                        "message": f"[记忆复用] 找到匹配路线 {matched_route.route_id}，跳过模型推理，步骤 {current_step}/{total_steps}",
                        "memory_info": {
                            "replay_mode": True,
                            "replay_route_id": matched_route.route_id,
                            "replay_step": current_step,
                            "replay_total": total_steps,
                            "replay_complete": is_complete,
                            "skipped_inference": True
                        }
                    }

                    # 保存可视化结果（记忆复用开始）
                    if memory_config.save_visualization:
                        save_memory_visualization(
                            image=rgb,
                            instruction=instruction,
                            step_idx=session_state['request_count'],
                            output_action=action,
                            pixel_target=None,
                            memory_mode="replay",
                            memory_info=response["memory_info"],
                            config=memory_config,
                            session_id=f"replay_{instruction.replace(' ', '_')[:20]}"
                        )

                    # 更新会话状态
                    session_state['request_count'] += 1
                    session_state['last_instruction'] = instruction
                    session_state['last_task'] = instruction

                    logger.info(f"📤 响应JSON (记忆复用开始): {json.dumps(response, ensure_ascii=False, indent=2)}")
                    return response

        # ===== 直接控制指令处理 =====
        if instruction in ["turn left", "turn right", "go straight"]:
            import math

            # 定义直接控制指令的映射
            direct_commands = {
                "turn left": [0.0, 0.0, math.pi / 12],      # 左转15度
                "turn right": [0.0, 0.0, -math.pi / 12],    # 右转15度
                "go straight": [1.0, 0.0, 0.0]              # 前进1米
            }

            action = direct_commands[instruction]
            logger.info(f"⚡ 检测到直接控制指令: '{instruction}'")
            logger.info(f"   控制命令: x={action[0]:.3f}, y={action[1]:.3f}, yaw={action[2]:.4f} rad ({action[2] * 180 / math.pi:.1f}°)")

            # 更新session_state（不更新last_task，保持导航任务不变）
            session_state['request_count'] += 1
            session_state['last_instruction'] = instruction

            response = {
                "status": "success",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [action],
                "pixel_target": None,
                "message": f"执行直接控制指令: {instruction}"
            }

            # 打印响应JSON
            logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")

            return response

        # 解码深度图（如果提供）
        if 'depth' in message_data and message_data['depth']:
            depth = decode_base64_depth(message_data['depth'])
            if depth is None:
                logger.warning("深度图解码失败，使用全零深度图")
                depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            else:
                logger.info(f"📏 输入深度图: 尺寸={depth.shape}, 数据类型={depth.dtype}, 深度范围=[{depth.min():.2f}, {depth.max():.2f}]")
        else:
            # 如果没有提供深度图，使用全零深度图
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            logger.info("未提供深度图，使用全零深度图")

        # 解析pose（如果提供）
        if 'pose' in message_data and message_data['pose']:
            pose = np.array(message_data['pose'], dtype=np.float32)
        else:
            pose = np.eye(4, dtype=np.float32)

        # 解析intrinsic（如果提供）
        if 'intrinsic' in message_data and message_data['intrinsic']:
            intrinsic = np.array(message_data['intrinsic'], dtype=np.float32)
        else:
            intrinsic = np.array([
                [386.5, 0.0, 328.9, 0.0],
                [0.0, 386.5, 244.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

        # 解析look_down标志
        look_down = message_data.get('look_down', False)

        # 获取agent的历史帧配置和当前状态
        max_history_frames = agent.num_history if hasattr(agent, 'num_history') else 8
        current_history_count = len(agent.rgb_list) if hasattr(agent, 'rgb_list') else 0
        current_episode_idx = agent.episode_idx if hasattr(agent, 'episode_idx') else 0
        resize_h = agent.resize_h if hasattr(agent, 'resize_h') else 384
        resize_w = agent.resize_w if hasattr(agent, 'resize_w') else 384

        # 计算本次推理将要采样的历史帧序号（模拟agent内部的采样逻辑）
        if current_episode_idx == 0 or not look_down:
            if current_episode_idx == 0:
                sampled_history_ids = []
            else:
                sampled_history_ids = np.unique(np.linspace(0, current_episode_idx - 1, max_history_frames, dtype=np.int32)).tolist()
        else:
            sampled_history_ids = "使用上次采样"

        logger.info(f"🎯 推理参数详情:")
        logger.info(f"  ├─ 导航指令: '{instruction}'")
        logger.info(f"  ├─ 输入尺寸: RGB={rgb.shape}, Depth={depth.shape}")
        logger.info(f"  ├─ 模型配置: 目标尺寸={resize_h}x{resize_w}, 最大历史帧数={max_history_frames}")
        logger.info(f"  ├─ 历史帧状态: 已累积={current_history_count}帧, 本次采样使用={sampled_history_ids}")
        logger.info(f"  └─ 其他参数: look_down={look_down}, episode_idx={current_episode_idx}")

        # 执行推理
        start_time = time.time()
        async with agent_lock:
            dual_sys_output = await asyncio.to_thread(
                agent.step, rgb, depth, pose, instruction, intrinsic, False
            )

        # 【新增】检测动作5并处理"向下看"
        if (dual_sys_output.output_action is not None and
            len(dual_sys_output.output_action) > 0 and
            dual_sys_output.output_action[0] == 5):

            logger.info(f"🔍 检测到动作5（向下看），准备执行look_down推理...")
            logger.info(f"   原始输出动作: {dual_sys_output.output_action}")

            # 使用相同的图像，设置look_down=True重新推理
            async with agent_lock:
                dual_sys_output = await asyncio.to_thread(
                    agent.step, rgb, depth, pose, instruction, intrinsic, look_down=True
                )

            logger.info(f"✅ look_down推理完成")
            logger.info(f"   新的输出动作: {dual_sys_output.output_action}")
            logger.info(f"   新的输出像素: {dual_sys_output.output_pixel}")
            logger.info(f"   新的输出轨迹: {dual_sys_output.output_trajectory is not None}")

        inference_time = time.time() - start_time

        # 推理完成后再次获取历史帧数量和episode索引
        history_count_after = len(agent.rgb_list) if hasattr(agent, 'rgb_list') else 0
        episode_idx_after = agent.episode_idx if hasattr(agent, 'episode_idx') else 0
        logger.info(f"✅ 推理完成: 耗时={inference_time:.2f}秒, 累积历史帧={history_count_after}帧 (episode_idx={episode_idx_after})")

        # 构建响应 - 新格式，适配机器人控制接口
        response = {
            "status": "success",
            "id": robot_id,
            "pts": pts,
            "task_status": "executing",  # 默认值，后续根据输出调整
            "action": [[0.0, 0.0, 0.0]],  # 默认值
            "pixel_target": None,  # 归一化像素目标，默认为None
            "message": ""
        }

        # 添加输出字段并转换为机器人控制格式
        logger.info(f"📊 推理结果详情:")

        output_action = None
        if dual_sys_output.output_action is not None:
            # 情况1/2/4：离散动作序列，转换为合并的[x, y, yaw]格式
            output_action = dual_sys_output.output_action
            action_map = {0: 'STOP', 1: '↑前进', 2: '←左转', 3: '→右转', 5: '↓向下看'}
            action_str = ', '.join([f"{action_map.get(a, str(a))}" for a in output_action[:5]])
            if len(output_action) > 5:
                action_str += f", ... (共{len(output_action)}个动作)"
            logger.info(f"  ├─ 输出动作序列: {action_str}")
            logger.info(f"  │  └─ 原始序列: {output_action}")

            # 【新增】如果包含动作5，添加说明
            if 5 in output_action:
                logger.info(f"  │  ⚠️  注意: 输出包含动作5（向下看），已在推理阶段处理")

            # 转换为机器人控制格式
            robot_action, task_status = convert_output_action_to_robot_action(output_action)
            response["action"] = robot_action
            response["task_status"] = task_status
            logger.info(f"  ├─ 转换后机器人动作: {robot_action}")
            logger.info(f"  ├─ 任务状态: {task_status}")

        elif dual_sys_output.output_trajectory is not None:
            # 情况3：轨迹点列表，转换为累积坐标
            traj_shape = dual_sys_output.output_trajectory.shape
            logger.info(f"  ├─ 输出轨迹: shape={traj_shape}")

            # 转换为机器人控制格式（累积坐标）
            robot_action = convert_trajectory_to_robot_action(dual_sys_output.output_trajectory.tolist())
            response["action"] = robot_action
            response["task_status"] = "executing"

            # 计算累积坐标用于日志和可视化
            if len(robot_action) > 0:
                # robot_action 已经是累积坐标格式 [[x, y, yaw], ...]
                cumsum_trajectory = np.array([[pt[0], pt[1]] for pt in robot_action])
                start_point = cumsum_trajectory[0]
                end_point = cumsum_trajectory[-1]
                logger.info(f"  │  ├─ 起点(累积): [{start_point[0]:.3f}, {start_point[1]:.3f}]")
                logger.info(f"  │  └─ 终点(累积): [{end_point[0]:.3f}, {end_point[1]:.3f}]")
                # 保存累积轨迹供可视化使用
                dual_sys_output.output_trajectory = cumsum_trajectory

            logger.info(f"  ├─ 转换后轨迹点数: {len(robot_action)}")

        if dual_sys_output.output_pixel is not None:
            # 图像尺寸为 640x480
            pixel_y_normalized = dual_sys_output.output_pixel[0] / 480.0
            pixel_x_normalized = dual_sys_output.output_pixel[1] / 640.0
            response["pixel_target"] = [pixel_x_normalized, pixel_y_normalized]
            logger.info(f"  └─ 输出像素目标: [y={dual_sys_output.output_pixel[0]}, x={dual_sys_output.output_pixel[1]}]")
            logger.info(f"     归一化像素目标: [y={pixel_y_normalized:.4f}, x={pixel_x_normalized:.4f}]")

        # ===== 小动作检测（33个点的自动停止）=====
        action_list = response["action"]
        if len(action_list) == 33:
            # 检查每个三元组的所有值是否都小于0.5（绝对值）
            all_small_movements = True
            for action_triplet in action_list:
                # action_triplet 格式: [x, y, yaw]
                if len(action_triplet) >= 3:
                    x, y, yaw = action_triplet[0], action_triplet[1], action_triplet[2]
                    if abs(x) >= 0.5 or abs(y) >= 0.5 or abs(yaw) >= 0.5:
                        all_small_movements = False
                        break

            if all_small_movements:
                logger.info(f"🎯 检测到33个小动作（所有值绝对值<0.5），自动转换为停止")
                logger.info(f"   原始action前3个: {action_list[:3]}")
                response["action"] = [[0.0, 0.0, 0.0]]
                response["task_status"] = "end"
                logger.info(f"   修改后: action={response['action']}, task_status={response['task_status']}")

        # 可视化推理结果并保存
        try:
            annotated_image = annotate_image(
                idx=timestamp_str,
                image=rgb,
                instruction=instruction,
                output_action=dual_sys_output.output_action,
                trajectory=dual_sys_output.output_trajectory,
                pixel_goal=dual_sys_output.output_pixel,
                output_dir=images_dir
            )
            logger.info(f"🎨 生成可视化结果: {os.path.join(images_dir, f'annotated_{timestamp_str}.jpg')}")
        except Exception as e:
            logger.warning(f"生成可视化结果失败: {e}", exc_info=True)

        # ===== 记忆处理 (仅当记忆功能开启时) =====
        if memory_enabled:
            # 获取pixel_target用于关键帧判断 - 使用归一化值 [x, y] (与响应中的pixel_target保持一致)
            pixel_target_for_memory = None
            if dual_sys_output.output_pixel is not None:
                # dual_sys_output.output_pixel 是 [y, x] 格式的原始像素值
                # 转换为归一化的 [x, y] 格式
                pixel_y_normalized = dual_sys_output.output_pixel[0] / 480.0
                pixel_x_normalized = dual_sys_output.output_pixel[1] / 640.0
                pixel_target_for_memory = [pixel_x_normalized, pixel_y_normalized]
            memory_info = memory_agent.process_observation(
                rgb_image=rgb,
                surround_images=surround_images,
                action=output_action if output_action else [1],  # 默认前进
                instruction=instruction,
                pixel_target=pixel_target_for_memory,
                # v2.1 节点来源追踪：传递图片时间戳
                source_timestamp=timestamp_str
            )
            response["memory_info"] = memory_info

            # 只在关键帧时保存可视化结果 (pixel_target不为None时)
            is_keyframe = pixel_target_for_memory is not None
            if memory_config.save_visualization and is_keyframe:
                # 确定记忆模式
                if memory_agent.route_memory.is_recording():
                    viz_memory_mode = "recording"
                else:
                    viz_memory_mode = "inference"

                # 生成会话ID（基于指令和时间）
                session_id = f"{instruction.replace(' ', '_')[:20]}_{int(memory_agent.route_memory.start_node_id or 0)}"

                save_memory_visualization(
                    image=rgb,
                    instruction=instruction,
                    step_idx=session_state['request_count'],
                    output_action=output_action,
                    pixel_target=response.get("pixel_target"),
                    memory_mode=viz_memory_mode,
                    memory_info=memory_info,
                    config=memory_config,
                    session_id=session_id,
                    surround_images=surround_images  # 传递环视图用于VLM描述
                )
                logger.info(f"[关键帧可视化] 步骤 {session_state['request_count']} 保存为关键帧")
            elif not is_keyframe:
                logger.debug(f"[非关键帧] 步骤 {session_state['request_count']} 跳过可视化保存")
        else:
            # 记忆关闭时，只在关键帧时保存可视化（用于对比）
            is_keyframe = dual_sys_output.output_pixel is not None
            if memory_config.save_visualization and is_keyframe:
                save_memory_visualization(
                    image=rgb,
                    instruction=instruction,
                    step_idx=session_state['request_count'],
                    output_action=output_action,
                    pixel_target=response.get("pixel_target"),
                    memory_mode="disabled",
                    memory_info={},
                    config=memory_config,
                    session_id=f"no_memory_{instruction.replace(' ', '_')[:20]}",
                    surround_images=surround_images
                )

        # 打印响应JSON
        logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")

        # 更新会话状态（本连接）
        session_state['request_count'] += 1
        session_state['last_instruction'] = instruction
        session_state['last_task'] = instruction

        # 同步到全局会话状态（支持多连接共享task）
        if instruction and instruction not in ["None", "none", "", "STOP", "stop"]:
            global_session_state['last_task'] = instruction
            global_session_state['last_instruction'] = instruction
            global_session_state['total_request_count'] += 1

        return response

    except Exception as e:
        logger.error(f"推理异常: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"推理异常: {e}",
            "task_status": "end",
            "action": [[0.0, 0.0, 0.0]]
        }


async def handle_client(websocket):
    """处理客户端连接"""
    client_id = id(websocket)
    session_state = {
        'last_instruction': None,
        'request_count': 0,
        'last_task': None
    }

    global global_agent, global_memory_agent, main_model_device

    try:
        connected_clients[client_id] = {
            'websocket': websocket,
            'session_state': session_state
        }
        logger.info(f"新客户端连接 [{client_id}]。当前连接数: {len(connected_clients)}")

        # 模型已在服务启动时预加载，这里只做检查
        if global_agent is None:
            logger.error("错误: 主模型未加载，服务可能未正确初始化")
            return

        async for message in websocket:
            try:
                data = json.loads(message)

                # 日志 (简化)
                log_data = {k: v for k, v in data.items() if k != 'images'}
                if 'images' in data:
                    log_data['images'] = {k: f"<base64 len={len(v) if v else 0}>" for k, v in data['images'].items()}
                logger.info(f"收到消息 [{client_id}]: {json.dumps(log_data, ensure_ascii=False)[:500]}")

                logger.info("="*150)

                # 处理命令
                if data.get('command') == 'reset':
                    # v2.5.2: 支持keep_memory参数，保留拓扑图记忆
                    keep_memory = data.get('keep_memory', False)
                    async with agent_lock:
                        global_agent.reset()
                    if memory_config.memory_enabled and global_memory_agent is not None:
                        if keep_memory:
                            # 只重置会话状态，保留拓扑图
                            global_memory_agent.route_memory.clear_current_route()
                            logger.info(f"Agent已重置，记忆拓扑图已保留 [{client_id}]")
                        else:
                            global_memory_agent.reset()
                            logger.info(f"Agent和记忆已重置 [{client_id}]")
                    # 重置本连接的会话状态
                    session_state['last_instruction'] = None
                    session_state['request_count'] = 0
                    session_state['last_task'] = None
                    # 同时重置全局会话状态（所有连接共享的task）
                    global_session_state['last_task'] = None
                    global_session_state['last_instruction'] = None
                    global_session_state['total_request_count'] = 0
                    msg = "Agent已重置" if not memory_config.memory_enabled else ("Agent已重置，记忆已保留" if keep_memory else "Agent和记忆已重置")
                    response = {"status": "success", "message": msg, "keep_memory": keep_memory}

                elif data.get('command') == 'session_status':
                    response = {
                        "status": "success",
                        "message": "会话状态信息",
                        "session_info": {
                            "request_count": session_state['request_count'],
                            "last_instruction": session_state.get('last_instruction', None),
                            "last_task": session_state.get('last_task', None)
                        }
                    }

                elif data.get('command') == 'memory_status':
                    if memory_config.memory_enabled:
                        stats = global_memory_agent.topo_map.get_stats()
                        progress = global_memory_agent.route_memory.get_current_progress()
                        response = {
                            "status": "success",
                            "memory_info": {
                                "topo_stats": stats,
                                "route_progress": progress
                            }
                        }
                    else:
                        response = {
                            "status": "error",
                            "message": "记忆功能已关闭 (memory_enabled=False)"
                        }

                elif data.get('command') == 'start_memory':
                    if memory_config.memory_enabled:
                        instruction = data.get('instruction', 'default')
                        route_id = global_memory_agent.start_memory_recording(instruction)
                        response = {"status": "success", "route_id": route_id}
                    else:
                        response = {
                            "status": "error",
                            "message": "记忆功能已关闭 (memory_enabled=False)"
                        }

                elif data.get('command') == 'stop_memory':
                    if memory_config.memory_enabled:
                        route = global_memory_agent.stop_memory_recording()
                        response = {
                            "status": "success",
                            "route_id": route.route_id if route else None,
                            "frames": len(route.node_sequence) if route else 0
                        }
                    else:
                        response = {
                            "status": "error",
                            "message": "记忆功能已关闭 (memory_enabled=False)"
                        }

                elif data.get('command') == 'return_to_start':
                    if memory_config.memory_enabled:
                        success = global_memory_agent.start_return_navigation()
                        response = {
                            "status": "success" if success else "error",
                            "message": "返回导航已启动" if success else "无法启动返回导航"
                        }
                    else:
                        response = {
                            "status": "error",
                            "message": "记忆功能已关闭 (memory_enabled=False)"
                        }

                elif data.get('command') == 'get_graph':
                    # 获取拓扑图数据用于可视化
                    if memory_config.memory_enabled and global_memory_agent is not None:
                        graph_data = global_memory_agent.topo_map.get_graph_for_visualization()
                        response = {
                            "status": "success",
                            "data": graph_data
                        }
                    else:
                        response = {
                            "status": "error",
                            "message": "记忆功能已关闭",
                            "data": {"nodes": [], "edges": [], "current_node": None}
                        }

                elif data.get('command') == 'vpr_identify':
                    # VPR位置识别 - 根据上传的环视图片识别当前位置
                    if memory_config.memory_enabled and global_memory_agent is not None:
                        try:
                            images_b64 = data.get('images', {})
                            if not images_b64:
                                response = {"status": "error", "message": "请提供环视图片"}
                            else:
                                # 解码图片并提取特征
                                query_features = {}
                                for cam_id, img_b64 in images_b64.items():
                                    img_data = base64.b64decode(img_b64)
                                    img = Image.open(io.BytesIO(img_data))
                                    img_arr = np.array(img)
                                    feat = global_memory_agent.feature_extractor.extract_feature(img_arr)
                                    query_features[cam_id] = feat

                                # 使用VPR进行搜索
                                if query_features and global_memory_agent.topo_map.vpr.get_size() > 0:
                                    # 首先尝试多视角搜索
                                    results = global_memory_agent.topo_map.vpr.search_multi_view(query_features, k=5)
                                    if results:
                                        best_match = results[0]
                                        top_matches = [
                                            {'node_id': r.node_id, 'similarity': r.weighted_similarity, 'voting_score': r.voting_score}
                                            for r in results[:5]
                                        ]
                                        response = {
                                            "status": "success",
                                            "data": {
                                                "matched_node": best_match.node_id,
                                                "similarity": best_match.weighted_similarity,
                                                "top_matches": top_matches,
                                                "source": "multi_view"
                                            }
                                        }
                                        logger.info(f"VPR识别成功(多视角): node={best_match.node_id}, similarity={best_match.weighted_similarity:.3f}")
                                    else:
                                        # 回退: 多视角索引为空，使用主索引搜索
                                        logger.info("多视角索引为空，使用主索引搜索...")
                                        feat_list = list(query_features.values())
                                        if feat_list:
                                            fused_feature = np.mean(feat_list, axis=0)
                                            search_results = global_memory_agent.topo_map.vpr.search(fused_feature, k=5)
                                            if search_results:
                                                best_node_id, best_sim = search_results[0]
                                                top_matches = [
                                                    {'node_id': node_id, 'similarity': sim}
                                                    for node_id, sim in search_results[:5]
                                                ]
                                                response = {
                                                    "status": "success",
                                                    "data": {
                                                        "matched_node": best_node_id,
                                                        "similarity": best_sim,
                                                        "top_matches": top_matches,
                                                        "source": "fused"
                                                    }
                                                }
                                                logger.info(f"VPR识别成功(融合): node={best_node_id}, similarity={best_sim:.3f}")
                                            else:
                                                response = {"status": "error", "message": "VPR搜索未找到匹配节点"}
                                        else:
                                            response = {"status": "error", "message": "VPR搜索未找到匹配节点"}
                                else:
                                    response = {"status": "error", "message": "VPR索引为空，请先进行导航构建记忆"}
                        except Exception as e:
                            logger.error(f"VPR识别错误: {e}", exc_info=True)
                            response = {"status": "error", "message": f"VPR识别失败: {str(e)}"}
                    else:
                        response = {"status": "error", "message": "记忆功能已关闭"}

                elif data.get('command') == 'semantic_search':
                    # 语义检索 - 根据文字描述搜索匹配的节点
                    if memory_config.memory_enabled and global_memory_agent is not None:
                        try:
                            query = data.get('query', '').strip()
                            if not query:
                                response = {"status": "error", "message": "请提供搜索描述"}
                            else:
                                # 使用语义图进行搜索
                                matched_nodes = global_memory_agent.semantic_graph.search_by_description(query, top_k=5)
                                if matched_nodes:
                                    response = {
                                        "status": "success",
                                        "data": {
                                            "matched_nodes": matched_nodes,
                                            "best_match": matched_nodes[0] if matched_nodes else None
                                        }
                                    }
                                    logger.info(f"语义检索成功: query='{query}', 找到 {len(matched_nodes)} 个匹配")
                                else:
                                    response = {"status": "error", "message": "未找到匹配节点"}
                        except Exception as e:
                            logger.error(f"语义检索错误: {e}", exc_info=True)
                            response = {"status": "error", "message": f"语义检索失败: {str(e)}"}
                    else:
                        response = {"status": "error", "message": "记忆功能已关闭"}

                elif data.get('command') == 'clear_memory':
                    # 清空所有记忆数据
                    if memory_config.memory_enabled and global_memory_agent is not None:
                        try:
                            # 清空拓扑图
                            global_memory_agent.topo_map.clear()
                            # 清空语义图
                            global_memory_agent.semantic_graph.clear()
                            # 清空路线记忆
                            global_memory_agent.route_memory.clear_all()
                            logger.info("记忆已完全清空")
                            response = {
                                "status": "success",
                                "message": "记忆已清空"
                            }
                        except Exception as e:
                            logger.error(f"清空记忆失败: {e}", exc_info=True)
                            response = {"status": "error", "message": f"清空记忆失败: {str(e)}"}
                    else:
                        response = {"status": "error", "message": "记忆功能已关闭或未初始化"}

                else:
                    # 正常推理 - 根据配置决定是否启用记忆功能
                    response = await process_inference_with_memory(
                        data, session_state, global_agent, global_memory_agent,
                        memory_enabled=memory_config.memory_enabled
                    )

                await websocket.send(json.dumps(response, ensure_ascii=False))
                logger.info(f"已发送响应 [{client_id}]")

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"status": "error", "message": "无效JSON"}))
            except Exception as e:
                logger.error(f"处理消息错误: {e}", exc_info=True)
                await websocket.send(json.dumps({"status": "error", "message": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"连接关闭 [{client_id}]")
    finally:
        if client_id in connected_clients:
            del connected_clients[client_id]
        logger.info(f"断开连接 [{client_id}]。当前连接数: {len(connected_clients)}")


async def main(port: int = 9528):
    """启动WebSocket服务器"""
    global main_model_device
    os.chdir(project_root)

    # 设置GPU（根据用户配置）
    if memory_config.gpu_id is not None:
        main_model_device = f"cuda:{memory_config.gpu_id}"
        memory_config.main_model_device = f"cuda:{memory_config.gpu_id}"
        memory_config.feature_extractor_device = f"cuda:{memory_config.gpu_id}"
        memory_config.vlm_device = f"cuda:{memory_config.gpu_id}"
        logger.info(f"🎮 所有模型统一加载到 cuda:{memory_config.gpu_id}")
    else:
        # 多GPU模式：使用用户配置的GPU编号
        logger.info(f"🎮 多GPU模式：根据配置分配GPU")
        main_model_device = f"cuda:{memory_config.main_model_device}"
        # 将配置中的GPU编号转换为cuda格式
        memory_config.main_model_device = f"cuda:{memory_config.main_model_device}"
        memory_config.feature_extractor_device = f"cuda:{memory_config.feature_extractor_device}"
        memory_config.vlm_device = f"cuda:{memory_config.vlm_device}"
        logger.info(f"🎮 主模型(InternVLA): {main_model_device}")
        logger.info(f"🎮 特征提取器(LongCLIP): {memory_config.feature_extractor_device}")
        logger.info(f"🎮 VLM模型(Qwen3-VL): {memory_config.vlm_device}")

    # 创建记忆数据目录
    memory_path = os.path.join(project_root, memory_config.memory_save_path)
    os.makedirs(memory_path, exist_ok=True)

    logger.info("🚀 启动InternNav WebSocket服务器...")
    logger.info("=" * 60)
    logger.info("InternNav WebSocket服务器 - 增强版 (带视觉记忆功能)")
    logger.info("=" * 60)
    logger.info(f"📂 工作目录: {os.getcwd()}")
    logger.info(f"📝 日志文件路径: {get_log_file_path()}")
    logger.info(f"💾 记忆存储路径: {memory_path}")
    logger.info(f"🔧 记忆功能开关: {'开启' if memory_config.memory_enabled else '关闭'}")
    logger.info(f"FAISS可用: {FAISS_AVAILABLE}")
    logger.info(f"NetworkX可用: {NETWORKX_AVAILABLE}")

    # ====================================================================
    # 服务启动时预加载模型（不再懒加载）
    # ====================================================================
    global global_agent, global_memory_agent

    logger.info("=" * 60)
    logger.info("预加载模型...")
    logger.info("=" * 60)

    # 预加载主模型 (InternVLA)
    logger.info(f"正在加载主模型 InternVLA 到 {main_model_device}...")
    global_agent = init_agent(device=main_model_device)
    logger.info("✓ 主模型加载完成")

    # 预加载记忆Agent (仅当记忆功能开启时)
    if memory_config.memory_enabled:
        logger.info("正在加载记忆模块...")
        global_memory_agent = MemoryNavigationAgent(memory_config)
        logger.info("✓ 记忆模块加载完成")
    else:
        logger.info("记忆功能已关闭，跳过记忆模块加载")

    logger.info("=" * 60)
    logger.info("所有模型预加载完成！")
    logger.info("=" * 60)

    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        port,
        ping_interval=30,
        ping_timeout=10,
        max_size=50*1024*1024
    )

    logger.info(f"✅ InternNav WebSocket服务器已启动，监听端口 {port}")
    logger.info("📚 支持的消息格式:")
    logger.info("  输入格式:")
    logger.info("    - id: 机器人ID (必需)")
    logger.info("    - pts: 时间戳 (毫秒，必需)")
    logger.info("    - task: 导航指令 (必需，如 '穿过马路后左转')")
    logger.info("    - images: 图像字典 (必需)")
    logger.info("        - front_1: base64编码的前置摄像头图像 (必需)")
    logger.info("        - camera_1~4: 环视摄像头图像 (可选，用于记忆)")
    logger.info("  输出格式:")
    logger.info("    - status: 'success' 或 'error'")
    logger.info("    - id: 机器人ID")
    logger.info("    - pts: 时间戳")
    logger.info("    - task_status: 'executing' 或 'end'")
    logger.info("    - action: [[x, y, yaw], ...] 机器人控制命令")
    logger.info("    - pixel_target: [x, y] 归一化像素目标 (可选)")
    logger.info("    - memory_info: 记忆相关信息 (可选)")
    logger.info("    - message: 错误描述信息")
    logger.info("🔧 会话管理命令:")
    logger.info("  - command: 'reset' (重置Agent和记忆)")
    logger.info("  - command: 'session_status' (查看会话状态)")
    logger.info("  - command: 'memory_status' (查询记忆状态)")
    logger.info("  - command: 'start_memory' (开始记录路线)")
    logger.info("  - command: 'stop_memory' (停止记录)")
    logger.info("  - command: 'return_to_start' (返回起点)")
    logger.info("📌 特殊task指令:")
    logger.info("  - task: 'STOP' (停止任务)")
    logger.info("  - task: 'turn left' / 'turn right' / 'go straight' (直接控制)")
    logger.info("  - task: 'RETURN' / '返回起点' (返回起点导航)")
    logger.info("  - task: 'START_MEMORY' (开始记录)")
    logger.info("  - task: 'STOP_MEMORY' (停止记录)")
    logger.info("  - task: 'MEMORY_STATUS' (查询状态)")

    await server.wait_closed()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='InternNav WebSocket服务器 - 带视觉记忆功能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                使用示例:
                # 使用所有可用GPU，所有模型默认使用GPU 0（默认）
                python ws_proxy_with_memory.py
                
                # 仅使用第1号GPU（单GPU模式）
                python ws_proxy_with_memory.py --gpu 1
                
                # 使用第0号GPU并禁用记忆功能
                python ws_proxy_with_memory.py --gpu 0 --no-memory
                
                # 多GPU模式：主模型用GPU 0，VLM用GPU 1
                python ws_proxy_with_memory.py --main-gpu 0 --vlm-gpu 1
                
                # 多GPU模式：分别指定三个模型的GPU
                python ws_proxy_with_memory.py --main-gpu 0 --feature-gpu 0 --vlm-gpu 1
                
                注意:
                - --gpu参数：单GPU模式，通过CUDA_VISIBLE_DEVICES限制只使用指定的GPU，所有模型加载到cuda:0
                - --main-gpu/--feature-gpu/--vlm-gpu：多GPU模式，分别指定各模型使用的物理GPU卡号
                - GPU编号对应物理GPU卡号，例如 1 对应 /dev/nvidia1
        """
    )
    parser.add_argument('--gpu', type=str, default=1,
                        help='单GPU模式：指定使用的GPU编号')
    parser.add_argument('--main-gpu', type=str, default=None,
                        help='多GPU模式：主模型(InternVLA)使用的GPU编号')
    parser.add_argument('--feature-gpu', type=str, default=None,
                        help='多GPU模式：特征提取器(LongCLIP)使用的GPU编号')
    parser.add_argument('--vlm-gpu', type=str, default=None,
                        help='多GPU模式：VLM模型(Qwen3-VL)使用的GPU编号')
    parser.add_argument('--no-memory', action='store_true',
                        help='禁用记忆功能（行为与原始ws_proxy.py一致）')
    parser.add_argument('--port', type=int, default=9528,
                        help='WebSocket服务端口（默认: 9528）')
    args = parser.parse_args()

    # GPU 0排除检查函数
    def validate_gpu_id(gpu_id: str, param_name: str) -> str:
        """验证GPU ID不是0，如果是0则警告并使用1"""
        if gpu_id == "0":
            logger.warning(f"⚠️ {param_name} 指定了GPU 0，但GPU 0已被排除。自动切换到GPU 1")
            return "1"
        return gpu_id

    # 应用命令行参数到配置
    if args.gpu is not None:
        memory_config.gpu_id = args.gpu
        logger.info(f"命令行参数: 单GPU模式，使用GPU {args.gpu}")
    else:
        # 多GPU模式：分别配置各模型的GPU
        if args.main_gpu is not None:
            validated_gpu = validate_gpu_id(args.main_gpu, "--main-gpu")
            memory_config.main_model_device = validated_gpu
            logger.info(f"命令行参数: 主模型使用GPU {validated_gpu}")
        if args.feature_gpu is not None:
            validated_gpu = validate_gpu_id(args.feature_gpu, "--feature-gpu")
            memory_config.feature_extractor_device = validated_gpu
            logger.info(f"命令行参数: 特征提取器使用GPU {validated_gpu}")
        if args.vlm_gpu is not None:
            validated_gpu = validate_gpu_id(args.vlm_gpu, "--vlm-gpu")
            memory_config.vlm_device = validated_gpu
            logger.info(f"命令行参数: VLM模型使用GPU {validated_gpu}")
    
    if args.no_memory:
        memory_config.memory_enabled = False
        logger.info("命令行参数: 禁用记忆功能")

    # 设置服务端口
    logger.info(f"命令行参数: 服务端口 {args.port}")

    try:
        asyncio.run(main(port=args.port))
    except KeyboardInterrupt:
        logger.info("⛔ 服务器正在关闭...")
    except Exception as e:
        logger.error(f"❌ 服务器发生错误: {e}", exc_info=True)
