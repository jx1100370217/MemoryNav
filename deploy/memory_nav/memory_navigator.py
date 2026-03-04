#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.0 - 记忆导航器

整合 VPR 定位、语义匹配、路径规划的主导航接口：
1. VPR定位：根据当前环视图定位起点
2. 语义匹配：根据目的地名称匹配终点
3. 路径规划：规划最短路径
4. 导航执行：获取每一步的 angle、pixel_position、stitch_image
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import os

from .memory_models import (
    MemoryNode, MemoryEdge, NavigationStep, 
    NavigationPlan, VPRResult
)
from .memory_graph import MemoryGraph
from .memory_vpr import MemoryVPR
from .memory_builder import MemoryBuilder, LongCLIPExtractor
from .anyloc_extractor import AnyLocExtractor
from .vpr_factory import create_vpr_extractor

logger = logging.getLogger(__name__)


class MemoryNavigator:
    """
    记忆导航器
    
    提供完整的记忆导航功能：
    - 定位当前位置（VPR）
    - 查找目的地（语义匹配）
    - 规划导航路径
    - 获取导航指令
    """
    
    def __init__(self, 
                 graph: MemoryGraph = None,
                 vpr: MemoryVPR = None,
                 feature_extractor = None,
                 feature_dim: int = 768,
                 device: str = "cuda:0",
                 vpr_method: str = "anyloc",
                 anyloc_config: dict = None):
        """
        初始化导航器
        
        Args:
            graph: 记忆图（可选，可后续设置）
            vpr: VPR模块（可选，可后续设置）
            feature_extractor: 特征提取器
            feature_dim: 特征维度
            device: 计算设备
            vpr_method: VPR方法 'anyloc', 'megaloc', 'effovpr', 'selavpr', 'longclip'
            anyloc_config: AnyLoc配置 (详见 MemoryBuilder)
        """
        self.graph = graph
        self.vpr = vpr
        self.device = device
        self.vpr_method = vpr_method
        
        # 特征提取器
        if feature_extractor is not None:
            self.extractor = feature_extractor
            self.feature_dim = feature_dim
        else:
            self.extractor, self.feature_dim, _ = create_vpr_extractor(
                vpr_method=vpr_method, device=device, config=anyloc_config)
        
        # 当前导航状态
        self.current_node_id: Optional[str] = None
        self.current_plan: Optional[NavigationPlan] = None
        self.current_step_index: int = 0
        
        logger.info(f"[MemoryNavigator] 初始化完成")
    
    def set_memory(self, graph: MemoryGraph, vpr: MemoryVPR):
        """设置记忆图和VPR"""
        self.graph = graph
        self.vpr = vpr
        logger.info(f"[MemoryNavigator] 记忆已设置: {len(graph.nodes)} 节点")
    
    def load_memory(self, path: str, data_dir: str = None):
        """
        加载或构建记忆
        
        Args:
            path: 记忆文件路径
            data_dir: 数据目录（如果需要构建）
        """
        builder = MemoryBuilder(
            feature_extractor=self.extractor,
            feature_dim=self.feature_dim,
            device=self.device,
            vpr_method=self.vpr_method
        )
        
        graph_path = path if path.endswith('.pkl') else path + "_graph.pkl"
        
        if os.path.exists(graph_path):
            # 加载已有记忆
            self.graph, self.vpr = builder.load(path)
        elif data_dir and os.path.exists(data_dir):
            # 从数据构建
            self.graph, self.vpr = builder.build_from_directory(
                data_dir, extract_features=True, save_path=path
            )
        else:
            raise ValueError(f"记忆文件和数据目录都不存在: {path}, {data_dir}")
        
        logger.info(f"[MemoryNavigator] 记忆已加载: {len(self.graph.nodes)} 节点")
    
    # ========================================================================
    # 定位功能
    # ========================================================================
    
    def locate_by_images(self, camera_images: Dict[str, np.ndarray],
                        return_features: bool = False):
        """
        通过环视图定位当前位置
        
        Args:
            camera_images: 环视相机图像 {'camera_1': image, ...}
            return_features: 是否同时返回提取的特征（用于趋势检测）
            
        Returns:
            如果 return_features=False: VPRResult 或 None
            如果 return_features=True: (VPRResult 或 None, query_features dict)
        """
        if self.vpr is None:
            logger.error("[MemoryNavigator] VPR未初始化")
            return (None, {}) if return_features else None
        
        # 提取特征
        query_features = {}
        for cam_id, image in camera_images.items():
            if image is not None:
                feat = self.extractor.extract(image)
                query_features[cam_id] = feat
        
        # VPR定位
        result = self.vpr.locate(query_features)
        
        if result:
            self.current_node_id = result.matched_node_id
            logger.info(f"[MemoryNavigator] 定位成功: {result.matched_node_id} "
                       f"({result.matched_node_name})")
        
        if return_features:
            return result, query_features
        return result
    
    def locate_by_features(self, camera_features: Dict[str, np.ndarray]) -> Optional[VPRResult]:
        """
        通过已提取的特征定位
        
        Args:
            camera_features: 环视相机特征 {'camera_1': feature, ...}
            
        Returns:
            VPRResult 定位结果
        """
        if self.vpr is None:
            logger.error("[MemoryNavigator] VPR未初始化")
            return None
        
        result = self.vpr.locate(camera_features)
        
        if result:
            self.current_node_id = result.matched_node_id
        
        return result
    
    def set_current_node(self, node_id: str) -> bool:
        """
        手动设置当前节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否设置成功
        """
        if self.graph and node_id in self.graph.nodes:
            self.current_node_id = node_id
            logger.info(f"[MemoryNavigator] 当前节点设置为: {node_id}")
            return True
        return False
    
    # ========================================================================
    # 目的地查找
    # ========================================================================
    
    def find_destination(self, query: str) -> Optional[MemoryNode]:
        """
        通过名称查找目的地节点
        
        支持多种查询格式:
        - 直接节点名: "C8前台区"
        - 自然语言任务: "前往C8前台", "go to A8 front desk"
        - 节点ID: "8"
        
        Args:
            query: 目的地名称或自然语言任务
            
        Returns:
            MemoryNode 或 None
        """
        if self.graph is None:
            logger.error("[MemoryNavigator] Graph未初始化")
            return None
        
        # 1. 直接匹配（ID 或完整名称）
        node = self.graph.get_node(str(query)) or self.graph.get_node_by_name(query)
        if node:
            logger.info(f"[MemoryNavigator] 找到目的地(直接匹配): {node.node_id} ({node.node_name})")
            return node
        
        # 2. 从自然语言任务中提取地名
        # 去掉常见前缀: "前往", "去", "到", "go to", "navigate to" 等
        cleaned = re.sub(
            r'^(前往|去|到|走到|导航到|带我去|带我到|go\s+to|navigate\s+to|take\s+me\s+to)\s*',
            '', query, flags=re.IGNORECASE
        ).strip()
        
        if cleaned and cleaned != query:
            node = self.graph.get_node_by_name(cleaned)
            if node:
                logger.info(f"[MemoryNavigator] 找到目的地(任务提取): {node.node_id} ({node.node_name})")
                return node
        
        # 3. 对所有节点做子串匹配 — 用query中的关键词搜索
        # 将 query 拆分为关键词，逐个搜索
        candidates = self.graph.search_nodes_by_name(query, limit=1)
        if not candidates and cleaned:
            candidates = self.graph.search_nodes_by_name(cleaned, limit=1)
        
        if candidates:
            node = candidates[0]
            logger.info(f"[MemoryNavigator] 找到目的地(模糊搜索): {node.node_id} ({node.node_name})")
            return node
        
        logger.warning(f"[MemoryNavigator] 未找到目的地: {query}")
        return None
    
    def search_destinations(self, query: str, limit: int = 5) -> List[MemoryNode]:
        """
        搜索可能的目的地
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的节点列表
        """
        if self.graph is None:
            return []
        
        return self.graph.search_nodes_by_name(query, limit)
    
    def get_all_destinations(self) -> List[Tuple[str, str]]:
        """获取所有可用目的地"""
        if self.graph is None:
            return []
        return self.graph.get_all_node_names()
    
    # ========================================================================
    # 路径规划
    # ========================================================================
    
    def plan_navigation(self, 
                        goal: Union[str, MemoryNode],
                        start: Union[str, MemoryNode] = None) -> NavigationPlan:
        """
        规划导航路径
        
        Args:
            goal: 目标（节点ID、节点名称或MemoryNode）
            start: 起点（默认使用当前位置）
            
        Returns:
            NavigationPlan 导航计划
        """
        if self.graph is None:
            return NavigationPlan(
                start_node_id="", start_node_name="",
                goal_node_id="", goal_node_name="",
                path=[], steps=[], total_steps=0,
                success=False, message="Graph未初始化"
            )
        
        # 解析起点
        start_id = self._resolve_node_id(start) if start else self.current_node_id
        if not start_id:
            return NavigationPlan(
                start_node_id="", start_node_name="",
                goal_node_id="", goal_node_name="",
                path=[], steps=[], total_steps=0,
                success=False, message="起点未知，请先定位或指定起点"
            )
        
        # 解析终点
        goal_id = self._resolve_node_id(goal)
        if not goal_id:
            return NavigationPlan(
                start_node_id=start_id, start_node_name="",
                goal_node_id="", goal_node_name="",
                path=[], steps=[], total_steps=0,
                success=False, message=f"未找到目的地: {goal}"
            )
        
        # 规划路径
        plan = self.graph.plan_navigation(start_id, goal_id)
        
        if plan.success:
            self.current_plan = plan
            self.current_step_index = 0
            logger.info(f"[MemoryNavigator] 路径规划成功: {' -> '.join(plan.path)}")
        
        return plan
    
    def _resolve_node_id(self, node: Union[str, MemoryNode, None]) -> Optional[str]:
        """解析节点ID"""
        if node is None:
            return None
        
        if isinstance(node, MemoryNode):
            return node.node_id
        
        if isinstance(node, str):
            # 先检查是否为节点ID
            if node in self.graph.nodes:
                return node
            # 再检查是否为节点名称
            found = self.graph.get_node_by_name(node)
            if found:
                return found.node_id
        
        return None
    
    # ========================================================================
    # 导航执行
    # ========================================================================
    
    def get_current_step(self) -> Optional[NavigationStep]:
        """获取当前导航步骤"""
        if self.current_plan is None or not self.current_plan.success:
            return None
        
        if self.current_step_index >= len(self.current_plan.steps):
            return None
        
        return self.current_plan.steps[self.current_step_index]
    
    def get_navigation_info(self) -> Optional[Dict]:
        """
        获取当前导航信息
        
        返回当前步骤所需的导航数据：
        - angle: 需要转到的绝对地理角度
        - pixel_position: 归一化像素目标
        - stitch_image_path: 前视图路径
        - from_node/to_node: 起止节点信息
        """
        step = self.get_current_step()
        if step is None:
            return None
        
        return {
            'step_index': step.step_index,
            'total_steps': self.current_plan.total_steps,
            'from_node': {
                'id': step.from_node_id,
                'name': step.from_node_name
            },
            'to_node': {
                'id': step.to_node_id,
                'name': step.to_node_name
            },
            'angle': step.angle,
            'pixel_position': step.pixel_position,
            'stitch_image_path': step.stitch_image_path,
            'is_last_step': step.step_index == self.current_plan.total_steps - 1
        }
    
    def advance_step(self) -> bool:
        """
        前进到下一步
        
        Returns:
            是否还有下一步
        """
        if self.current_plan is None:
            return False
        
        self.current_step_index += 1
        
        if self.current_step_index < len(self.current_plan.steps):
            step = self.current_plan.steps[self.current_step_index]
            self.current_node_id = step.from_node_id
            return True
        
        # 导航完成
        self.current_node_id = self.current_plan.goal_node_id
        return False
    
    def is_navigation_complete(self) -> bool:
        """检查导航是否完成"""
        if self.current_plan is None:
            return True
        return self.current_step_index >= len(self.current_plan.steps)
    
    def cancel_navigation(self):
        """取消当前导航"""
        self.current_plan = None
        self.current_step_index = 0
        logger.info("[MemoryNavigator] 导航已取消")
    
    # ========================================================================
    # 完整导航流程
    # ========================================================================
    
    def navigate_to(self, 
                    destination: str,
                    camera_images: Dict[str, np.ndarray] = None,
                    start_node_id: str = None) -> Dict:
        """
        完整的导航流程
        
        1. 定位当前位置（如果提供图像）
        2. 查找目的地
        3. 规划路径
        4. 返回第一步导航信息
        
        Args:
            destination: 目的地名称
            camera_images: 当前环视图（用于VPR定位）
            start_node_id: 起点ID（如果不提供图像）
            
        Returns:
            {
                'success': bool,
                'message': str,
                'location': VPRResult,  # 定位结果
                'plan': NavigationPlan,  # 规划结果
                'current_step': dict,    # 当前步骤信息
            }
        """
        result = {
            'success': False,
            'message': '',
            'location': None,
            'plan': None,
            'current_step': None
        }
        
        # 1. 定位
        if camera_images:
            location = self.locate_by_images(camera_images)
            result['location'] = location
            if not location:
                result['message'] = "VPR定位失败"
                return result
        elif start_node_id:
            if not self.set_current_node(start_node_id):
                result['message'] = f"起点节点不存在: {start_node_id}"
                return result
        elif self.current_node_id is None:
            result['message'] = "当前位置未知，请提供环视图或指定起点"
            return result
        
        # 2. 查找目的地
        dest_node = self.find_destination(destination)
        if not dest_node:
            result['message'] = f"未找到目的地: {destination}"
            return result
        
        # 3. 规划路径
        plan = self.plan_navigation(dest_node)
        result['plan'] = plan
        
        if not plan.success:
            result['message'] = plan.message
            return result
        
        # 4. 获取第一步信息
        result['current_step'] = self.get_navigation_info()
        result['success'] = True
        result['message'] = f"导航规划成功: {plan.start_node_name} -> {plan.goal_node_name}, 共{plan.total_steps}步"
        
        return result
    
    # ========================================================================
    # 状态查询
    # ========================================================================
    
    def get_status(self) -> Dict:
        """获取导航器状态"""
        return {
            'current_node_id': self.current_node_id,
            'current_node_name': self.graph.nodes[self.current_node_id].node_name 
                                if self.graph and self.current_node_id in self.graph.nodes else None,
            'has_plan': self.current_plan is not None,
            'plan_progress': f"{self.current_step_index}/{self.current_plan.total_steps}" 
                            if self.current_plan else None,
            'navigation_complete': self.is_navigation_complete(),
            'graph_loaded': self.graph is not None,
            'total_nodes': len(self.graph.nodes) if self.graph else 0
        }
    
    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """获取节点详细信息"""
        if self.graph is None or node_id not in self.graph.nodes:
            return None
        
        node = self.graph.nodes[node_id]
        return {
            'node_id': node.node_id,
            'node_name': node.node_name,
            'camera_images': node.camera_images,
            'neighbors': [
                {'id': e.target_node_id, 'name': e.target_node_name, 'angle': e.angle}
                for e in node.edges
            ],
            'has_features': node.fused_feature is not None
        }
