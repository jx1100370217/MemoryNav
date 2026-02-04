#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 拓扑地图管理模块 v2.0

管理导航拓扑图，支持：
1. 节点创建和VPR回环检测
2. 双向边和最短路径规划
3. 基于相似度的节点合并
4. 语义信息管理

v2.0 新增功能:
- Dijkstra最短路径规划（支持从任意位置出发）
- 双向边自动创建
- 相似度节点合并
- 路径距离估算
"""

import time
import logging
import heapq
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np

# 尝试导入networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .config import MemoryNavigationConfig
from .models import TopologicalNode
from .vpr import VisualPlaceRecognition
from .semantic_graph import SemanticGraphManager
from .surround_fusion import SurroundCameraFusion

logger = logging.getLogger(__name__)


@dataclass
class PathPlanResult:
    """路径规划结果"""
    success: bool
    path: List[int]  # 节点ID序列
    total_distance: float  # 总距离
    total_steps: int  # 总步数
    waypoints: List[Dict]  # 路点详情
    estimated_actions: List[List[int]]  # 估计的动作序列


@dataclass
class NodeMergeResult:
    """节点合并结果"""
    merged: bool
    source_node: int
    target_node: int
    similarity: float
    affected_edges: int


class TopologicalMapManager:
    """
    拓扑地图管理器 v2.0

    支持：
    - 双向边拓扑图
    - Dijkstra最短路径规划
    - 相似度节点合并
    - 多视角VPR检测
    """

    def __init__(self, config: MemoryNavigationConfig):
        self.config = config

        if NETWORKX_AVAILABLE:
            # 使用无向图支持双向导航
            self.graph = nx.Graph()
        else:
            self.graph = None

        self.nodes: Dict[int, TopologicalNode] = {}
        self.next_node_id = 0
        self.last_node_id: Optional[int] = None
        self.current_node_id: Optional[int] = None  # 当前所在节点

        # 视觉位置识别 (v4.0 多视角)
        self.vpr = VisualPlaceRecognition(
            feature_dim=config.feature_dim,
            similarity_threshold=config.similarity_threshold
        )

        # GraphRAG 语义地图
        self.semantic_graph = SemanticGraphManager(config)

        # 环视相机特征融合器 (v2.0 支持独立编码)
        self.surround_fusion = SurroundCameraFusion(config)

        # 节点合并参数
        self.node_merge_threshold = config.node_merge_threshold
        self.auto_merge_enabled = True

        # 边的距离估算（默认为1，可以根据动作序列调整）
        self.default_edge_weight = 1.0

        # v2.1: 连续帧计数器，用于强制创建新节点
        self.consecutive_revisit_count = 0
        self.max_consecutive_revisits = 8  # 连续8帧合并到同一节点后，强制创建新节点
        self.last_revisit_node_id = None

        logger.info(f"[TopoMap v2.0] 初始化完成: 双向图, 最短路径规划, 节点合并阈值={self.node_merge_threshold}")

    def add_observation(self,
                        visual_feature: np.ndarray,
                        rgb_image: np.ndarray = None,
                        surround_images: Dict[str, np.ndarray] = None,
                        surround_features: Dict[str, np.ndarray] = None,
                        action_from_prev: List[int] = None,
                        instruction: str = None,
                        is_keyframe: bool = False,
                        pixel_target: List[float] = None,
                        scene_description: str = None,
                        semantic_labels: List[str] = None,
                        # v2.0 新增参数
                        node_name: str = None,
                        navigation_instruction: str = None,
                        front_view_feature: np.ndarray = None,
                        # v2.1 新增参数：节点来源追踪
                        source_timestamp: str = None) -> Tuple[int, bool, Optional[Tuple[int, float]]]:
        """
        添加新观测 - v2.1 支持多视角独立特征和增强节点信息

        Args:
            visual_feature: 视觉特征向量 (前置相机)
            rgb_image: RGB图像
            surround_images: 环视相机图像字典
            surround_features: 环视相机特征字典 (独立特征，不再融合)
            action_from_prev: 从上一节点到当前的动作
            instruction: 当前任务指令
            is_keyframe: 是否为关键帧
            pixel_target: 像素目标（关键帧时非None）
            scene_description: VLM生成的场景描述
            semantic_labels: 语义标签列表
            node_name: 节点名称（v2.0新增）
            navigation_instruction: 导航指令（v2.0新增）
            front_view_feature: 前视图特征编码（v2.0新增）
            source_timestamp: 来源图片时间戳（v2.1新增，用于追踪节点来源）

        Returns:
            (node_id, is_new_node, revisit_info): 节点ID, 是否新节点, 回访信息
        """
        current_time = time.time()

        # 获取独立的多视角特征
        multi_view_features = None
        fused_feature = None
        if surround_features and self.config.use_surround_cameras:
            multi_view_data = self.surround_fusion.get_independent_features(surround_features)
            multi_view_features = multi_view_data.camera_features
            fused_feature = multi_view_data.fused_feature

        # 执行多视角VPR检测
        revisit_result = self.vpr.is_revisited_multi_view(
            visual_feature, current_time, self.config.min_time_gap,
            query_surround_dict=multi_view_features,
            query_semantic_labels=semantic_labels
        )

        revisit_info = None
        if revisit_result:
            revisit_info = (revisit_result[0], revisit_result[1])

        # 关键帧策略：关键帧总是创建新节点
        if is_keyframe:
            logger.info(f"[Keyframe] 检测到关键帧，强制创建新节点 (VPR结果: {revisit_info})")
            node_id = self._create_node(
                visual_feature=visual_feature,
                rgb_image=rgb_image,
                surround_images=surround_images,
                surround_features=multi_view_features,
                fused_feature=fused_feature,
                action_from_prev=action_from_prev,
                instruction=instruction,
                pixel_target=pixel_target,
                scene_description=scene_description,
                semantic_labels=semantic_labels,
                current_time=current_time,
                is_keyframe=True,
                # v2.0 新增参数
                node_name=node_name,
                navigation_instruction=navigation_instruction,
                front_view_feature=front_view_feature,
                # v2.1 节点来源
                source_timestamp=source_timestamp
            )
            return node_id, True, revisit_info

        # 非关键帧：检查是否为已访问位置
        if revisit_info is not None:
            matched_node_id, similarity = revisit_info
            logger.info(f"[VPR] 回环检测: 当前位置匹配到节点 {matched_node_id} (相似度={similarity:.3f})")

            # v2.1: 检查连续合并计数，避免长时间停留在同一节点
            if matched_node_id == self.last_revisit_node_id:
                self.consecutive_revisit_count += 1
                if self.consecutive_revisit_count >= self.max_consecutive_revisits:
                    logger.info(f"[TopoMap] 连续{self.consecutive_revisit_count}帧合并到节点{matched_node_id}，强制创建新节点")
                    self.consecutive_revisit_count = 0
                    self.last_revisit_node_id = None
                    # 强制创建新节点
                    node_id = self._create_node(
                        visual_feature=visual_feature,
                        rgb_image=rgb_image,
                        surround_images=surround_images,
                        surround_features=multi_view_features,
                        fused_feature=fused_feature,
                        action_from_prev=action_from_prev,
                        instruction=instruction,
                        pixel_target=pixel_target,
                        scene_description=scene_description,
                        semantic_labels=semantic_labels,
                        current_time=current_time,
                        is_keyframe=False,
                        # v2.0 新增参数
                        node_name=node_name,
                        navigation_instruction=navigation_instruction,
                        front_view_feature=front_view_feature,
                        # v2.1 节点来源
                        source_timestamp=source_timestamp
                    )
                    return node_id, True, revisit_info
            else:
                self.consecutive_revisit_count = 1
                self.last_revisit_node_id = matched_node_id

            # 更新已有节点
            if matched_node_id in self.nodes:
                self.nodes[matched_node_id].visit_count += 1
                self._update_node_feature(matched_node_id, visual_feature)
                # v2.1: 记录节点来源
                if source_timestamp:
                    self._add_source_timestamp(matched_node_id, source_timestamp, current_time)

            # 添加双向边 (如果有前序节点且不同)
            if self.last_node_id is not None and self.last_node_id != matched_node_id:
                self._add_bidirectional_edge(self.last_node_id, matched_node_id, action_from_prev or [])
                self.semantic_graph.add_semantic_edge(
                    self.last_node_id, matched_node_id, action_from_prev or []
                )

            self.last_node_id = matched_node_id
            self.current_node_id = matched_node_id
            return matched_node_id, False, revisit_info

        # 检查与最近节点的相似度 (节点合并)
        nearest_results = self.vpr.search(visual_feature, k=1)
        if nearest_results:
            nearest_id, nearest_sim = nearest_results[0]
            logger.debug(f"[TopoMap] 节点检查: 最近节点={nearest_id}, 相似度={nearest_sim:.4f}, "
                        f"合并阈值={self.node_merge_threshold}")

            if nearest_sim > self.node_merge_threshold and self.auto_merge_enabled:
                # v2.1: 检查连续合并计数，避免长时间停留在同一节点
                if nearest_id == self.last_revisit_node_id:
                    self.consecutive_revisit_count += 1
                    if self.consecutive_revisit_count >= self.max_consecutive_revisits:
                        logger.info(f"[TopoMap] 连续{self.consecutive_revisit_count}帧合并到节点{nearest_id}，强制创建新节点")
                        self.consecutive_revisit_count = 0
                        self.last_revisit_node_id = None
                        # 强制创建新节点
                        node_id = self._create_node(
                            visual_feature=visual_feature,
                            rgb_image=rgb_image,
                            surround_images=surround_images,
                            surround_features=multi_view_features,
                            fused_feature=fused_feature,
                            action_from_prev=action_from_prev,
                            instruction=instruction,
                            pixel_target=pixel_target,
                            scene_description=scene_description,
                            semantic_labels=semantic_labels,
                            current_time=current_time,
                            is_keyframe=False,
                            # v2.0 新增参数
                            node_name=node_name,
                            navigation_instruction=navigation_instruction,
                            front_view_feature=front_view_feature,
                            # v2.1 节点来源
                            source_timestamp=source_timestamp
                        )
                        return node_id, True, None
                else:
                    self.consecutive_revisit_count = 1
                    self.last_revisit_node_id = nearest_id

                logger.info(f"[TopoMap] 节点合并: 与节点 {nearest_id} 相似度 {nearest_sim:.3f} > {self.node_merge_threshold}")

                if nearest_id in self.nodes:
                    self.nodes[nearest_id].visit_count += 1
                    self._update_node_feature(nearest_id, visual_feature)
                    # v2.1: 记录节点来源
                    if source_timestamp:
                        self._add_source_timestamp(nearest_id, source_timestamp, current_time)

                if self.last_node_id is not None and self.last_node_id != nearest_id:
                    self._add_bidirectional_edge(self.last_node_id, nearest_id, action_from_prev or [])
                    self.semantic_graph.add_semantic_edge(
                        self.last_node_id, nearest_id, action_from_prev or []
                    )

                self.last_node_id = nearest_id
                self.current_node_id = nearest_id
                return nearest_id, False, None

        # 创建新节点
        node_id = self._create_node(
            visual_feature=visual_feature,
            rgb_image=rgb_image,
            surround_images=surround_images,
            surround_features=multi_view_features,
            fused_feature=fused_feature,
            action_from_prev=action_from_prev,
            instruction=instruction,
            pixel_target=pixel_target,
            scene_description=scene_description,
            semantic_labels=semantic_labels,
            current_time=current_time,
            is_keyframe=False,
            # v2.0 新增参数
            node_name=node_name,
            navigation_instruction=navigation_instruction,
            front_view_feature=front_view_feature,
            # v2.1 节点来源
            source_timestamp=source_timestamp
        )

        return node_id, True, None

    def _create_node(self,
                     visual_feature: np.ndarray,
                     rgb_image: np.ndarray,
                     surround_images: Dict[str, np.ndarray],
                     surround_features: Dict[str, np.ndarray],
                     fused_feature: np.ndarray,
                     action_from_prev: List[int],
                     instruction: str,
                     pixel_target: List[float],
                     scene_description: str,
                     semantic_labels: List[str],
                     current_time: float,
                     is_keyframe: bool,
                     # v2.0 新增参数
                     node_name: str = None,
                     navigation_instruction: str = None,
                     front_view_feature: np.ndarray = None,
                     # v2.1 新增参数：节点来源追踪
                     source_timestamp: str = None) -> int:
        """创建新节点 - v2.1 支持增强节点信息和来源追踪"""
        # v2.0: 语义标签去重
        unique_labels = []
        if semantic_labels:
            seen = set()
            for label in semantic_labels:
                label_clean = label.strip()
                if label_clean and label_clean.lower() not in seen:
                    unique_labels.append(label_clean)
                    seen.add(label_clean.lower())

        # v2.1: 初始化来源时间戳列表
        initial_source_timestamps = []
        if source_timestamp:
            initial_source_timestamps.append({
                'timestamp': source_timestamp,
                'camera': 'front_1',  # 默认使用front_1相机
                'merged_at': current_time,
                'is_initial': True  # 标记为初始创建
            })

        node = TopologicalNode(
            node_id=self.next_node_id,
            visual_feature=visual_feature.copy(),
            rgb_image=rgb_image.copy() if rgb_image is not None else None,
            surround_images={k: v.copy() for k, v in (surround_images or {}).items()},
            timestamp=current_time,
            instruction_context=instruction,
            scene_description=scene_description,
            semantic_labels=unique_labels,
            pixel_target=pixel_target,
            is_keyframe=is_keyframe,
            # v2.0 新增字段
            node_name=node_name,
            navigation_instruction=navigation_instruction or instruction,
            front_view_feature=front_view_feature.copy() if front_view_feature is not None else None,
            created_at=current_time,
            updated_at=current_time,
            pixel_target_history=[{'target': pixel_target, 'timestamp': current_time}] if pixel_target else [],
            # v2.1 来源追踪
            source_timestamps=initial_source_timestamps
        )

        self.nodes[self.next_node_id] = node

        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_node(self.next_node_id, **{
                'timestamp': current_time,
                'is_keyframe': is_keyframe,
                'semantic_labels': unique_labels,
                'node_name': node_name  # v2.0 新增
            })

        # 添加到VPR索引 (多视角独立特征)
        self.vpr.add_feature(
            visual_feature, self.next_node_id, current_time,
            surround_feature=fused_feature,
            surround_features_dict=surround_features,
            semantic_labels=unique_labels,
            scene_description=scene_description
        )

        # 关键帧添加到语义图 - v2.0 增强
        if is_keyframe and (scene_description or unique_labels):
            self.semantic_graph.add_semantic_node(
                node_id=self.next_node_id,
                scene_description=scene_description or "",
                semantic_labels=unique_labels,
                visual_feature=visual_feature,
                # v2.0 新增参数
                node_name=node_name,
                navigation_instruction=navigation_instruction or instruction,
                front_view_feature=front_view_feature,
                pixel_target=pixel_target
            )
            logger.info(f"[GraphRAG] {'关键帧' if is_keyframe else '普通'}节点 {self.next_node_id} 已添加到语义图 (名称: {node_name})")

        # 添加双向边
        if self.last_node_id is not None and action_from_prev is not None:
            self._add_bidirectional_edge(self.last_node_id, self.next_node_id, action_from_prev)
            self.semantic_graph.add_semantic_edge(
                self.last_node_id, self.next_node_id, action_from_prev
            )

        logger.info(f"新增拓扑节点: {self.next_node_id} (总节点数={len(self.nodes)}, 是否关键帧={is_keyframe})")

        self.last_node_id = self.next_node_id
        self.current_node_id = self.next_node_id
        self.next_node_id += 1

        # 内存管理
        if len(self.nodes) > self.config.max_nodes:
            self._cleanup_old_nodes()

        return self.next_node_id - 1

    def _add_bidirectional_edge(self, from_node: int, to_node: int, actions: List[int],
                                 weight: float = None):
        """
        添加双向边

        Args:
            from_node: 起始节点
            to_node: 目标节点
            actions: 正向动作序列
            weight: 边权重（距离），None则使用默认值
        """
        if NETWORKX_AVAILABLE and self.graph is not None:
            edge_weight = weight if weight is not None else self.default_edge_weight

            # 计算反向动作 (简单取反)
            reverse_actions = self._compute_reverse_actions(actions)

            # 添加无向边 (Graph会自动处理为双向)
            self.graph.add_edge(from_node, to_node,
                               weight=edge_weight,
                               forward_actions=actions,
                               backward_actions=reverse_actions)

            logger.debug(f"[TopoMap] 添加双向边: {from_node} <-> {to_node}, weight={edge_weight}")

    def _compute_reverse_actions(self, actions: List[int]) -> List[int]:
        """
        计算反向动作

        简单实现：
        - 0 (停止) -> 0
        - 1 (前进) -> 3 (后退)
        - 2 (左转) -> 4 (右转)
        - 3 (后退) -> 1 (前进)
        - 4 (右转) -> 2 (左转)
        """
        action_reverse_map = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2}
        return [action_reverse_map.get(a, a) for a in reversed(actions)]

    # ========================================================================
    # v2.0 新增: 最短路径规划
    # ========================================================================

    def plan_shortest_path(self, start_node: int, goal_node: int) -> PathPlanResult:
        """
        Dijkstra最短路径规划

        Args:
            start_node: 起始节点ID
            goal_node: 目标节点ID

        Returns:
            PathPlanResult: 路径规划结果
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return PathPlanResult(
                success=False, path=[], total_distance=0,
                total_steps=0, waypoints=[], estimated_actions=[]
            )

        # 检查节点存在性
        if start_node not in self.graph or goal_node not in self.graph:
            logger.warning(f"[TopoMap] 路径规划失败: 节点不存在 (start={start_node}, goal={goal_node})")
            return PathPlanResult(
                success=False, path=[], total_distance=0,
                total_steps=0, waypoints=[], estimated_actions=[]
            )

        try:
            # 使用Dijkstra算法
            path = nx.dijkstra_path(self.graph, start_node, goal_node, weight='weight')
            total_distance = nx.dijkstra_path_length(self.graph, start_node, goal_node, weight='weight')

            # 构建路点详情和动作序列
            waypoints = []
            estimated_actions = []

            for i, node_id in enumerate(path):
                node = self.nodes.get(node_id)
                waypoint = {
                    'node_id': node_id,
                    'step': i,
                    'is_keyframe': node.is_keyframe if node else False,
                    'semantic_labels': node.semantic_labels if node else [],
                    'scene_description': node.scene_description if node else ''
                }
                waypoints.append(waypoint)

                # 获取到下一个节点的动作
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_data = self.graph.get_edge_data(node_id, next_node)
                    if edge_data:
                        # 判断是正向还是反向
                        forward_actions = edge_data.get('forward_actions', [])
                        backward_actions = edge_data.get('backward_actions', [])
                        # 这里简化处理，实际可能需要更复杂的逻辑
                        estimated_actions.append(forward_actions if forward_actions else backward_actions)
                    else:
                        estimated_actions.append([])

            logger.info(f"[TopoMap] 路径规划成功: {start_node} -> {goal_node}, "
                       f"距离={total_distance:.2f}, 步数={len(path)}")

            return PathPlanResult(
                success=True,
                path=path,
                total_distance=total_distance,
                total_steps=len(path),
                waypoints=waypoints,
                estimated_actions=estimated_actions
            )

        except nx.NetworkXNoPath:
            logger.warning(f"[TopoMap] 路径规划失败: 无法到达 (start={start_node}, goal={goal_node})")
            return PathPlanResult(
                success=False, path=[], total_distance=0,
                total_steps=0, waypoints=[], estimated_actions=[]
            )
        except nx.NodeNotFound as e:
            logger.warning(f"[TopoMap] 路径规划失败: 节点不存在 {e}")
            return PathPlanResult(
                success=False, path=[], total_distance=0,
                total_steps=0, waypoints=[], estimated_actions=[]
            )

    def plan_path_from_current(self, goal_node: int) -> PathPlanResult:
        """
        从当前位置规划到目标的路径

        Args:
            goal_node: 目标节点ID

        Returns:
            PathPlanResult: 路径规划结果
        """
        if self.current_node_id is None:
            logger.warning("[TopoMap] 路径规划失败: 当前位置未知")
            return PathPlanResult(
                success=False, path=[], total_distance=0,
                total_steps=0, waypoints=[], estimated_actions=[]
            )

        return self.plan_shortest_path(self.current_node_id, goal_node)

    def find_nearest_node_to_query(self, query_feature: np.ndarray,
                                    surround_features: Dict[str, np.ndarray] = None) -> Optional[int]:
        """
        找到与查询特征最相似的节点（用于定位当前位置）

        Args:
            query_feature: 查询视觉特征
            surround_features: 多视角特征

        Returns:
            最相似节点的ID，如果无匹配则返回None
        """
        if surround_features:
            results = self.vpr.search_multi_view(surround_features, k=1)
            if results:
                return results[0].node_id
        else:
            results = self.vpr.search(query_feature, k=1)
            if results:
                return results[0][0]
        return None

    # ========================================================================
    # v2.0 新增: 节点合并
    # ========================================================================

    def merge_similar_nodes(self, threshold: float = None) -> List[NodeMergeResult]:
        """
        合并相似度超过阈值的节点

        Args:
            threshold: 合并阈值，None则使用默认值

        Returns:
            List[NodeMergeResult]: 合并结果列表
        """
        if threshold is None:
            threshold = self.node_merge_threshold

        merge_results = []
        nodes_to_merge: List[Tuple[int, int, float]] = []

        # 找出所有需要合并的节点对
        node_ids = list(self.nodes.keys())
        for i, node_id in enumerate(node_ids):
            if node_id not in self.nodes:
                continue
            feature = self.nodes[node_id].visual_feature
            results = self.vpr.search(feature, k=5)

            for other_id, similarity in results:
                if other_id != node_id and similarity > threshold:
                    # 保留访问次数更多的节点
                    if self.nodes[node_id].visit_count >= self.nodes.get(other_id, self.nodes[node_id]).visit_count:
                        nodes_to_merge.append((other_id, node_id, similarity))  # 合并other到node
                    else:
                        nodes_to_merge.append((node_id, other_id, similarity))  # 合并node到other

        # 执行合并
        merged_nodes: Set[int] = set()
        for source_node, target_node, similarity in nodes_to_merge:
            if source_node in merged_nodes or target_node in merged_nodes:
                continue

            result = self._merge_node(source_node, target_node)
            if result.merged:
                merged_nodes.add(source_node)
                merge_results.append(result)

        if merge_results:
            logger.info(f"[TopoMap] 节点合并完成: 合并了 {len(merge_results)} 个节点")

        return merge_results

    def _merge_node(self, source_node: int, target_node: int) -> NodeMergeResult:
        """
        将source_node合并到target_node

        Args:
            source_node: 要被合并的节点
            target_node: 合并目标节点

        Returns:
            NodeMergeResult: 合并结果
        """
        if source_node not in self.nodes or target_node not in self.nodes:
            return NodeMergeResult(
                merged=False, source_node=source_node, target_node=target_node,
                similarity=0.0, affected_edges=0
            )

        source = self.nodes[source_node]
        target = self.nodes[target_node]

        # 计算相似度
        similarity = float(np.dot(source.visual_feature, target.visual_feature))

        # 更新target节点
        target.visit_count += source.visit_count
        # 使用加权平均更新特征
        alpha = source.visit_count / (source.visit_count + target.visit_count)
        target.visual_feature = (1 - alpha) * target.visual_feature + alpha * source.visual_feature
        target.visual_feature = target.visual_feature / (np.linalg.norm(target.visual_feature) + 1e-8)

        # 转移边
        affected_edges = 0
        if NETWORKX_AVAILABLE and self.graph is not None:
            if source_node in self.graph:
                for neighbor in list(self.graph.neighbors(source_node)):
                    if neighbor != target_node:
                        edge_data = self.graph.get_edge_data(source_node, neighbor)
                        if edge_data and not self.graph.has_edge(target_node, neighbor):
                            self.graph.add_edge(target_node, neighbor, **edge_data)
                            affected_edges += 1
                self.graph.remove_node(source_node)

        # 删除source节点
        del self.nodes[source_node]

        logger.info(f"[TopoMap] 节点合并: {source_node} -> {target_node}, "
                   f"相似度={similarity:.4f}, 受影响边数={affected_edges}")

        return NodeMergeResult(
            merged=True,
            source_node=source_node,
            target_node=target_node,
            similarity=similarity,
            affected_edges=affected_edges
        )

    # ========================================================================
    # 原有方法（保持兼容）
    # ========================================================================

    def find_path(self, start_node: int, goal_node: int) -> List[int]:
        """在拓扑图上查找路径（兼容旧接口）"""
        result = self.plan_shortest_path(start_node, goal_node)
        return result.path

    def get_edge_actions(self, from_node: int, to_node: int) -> List[int]:
        """获取两节点间的动作序列"""
        if NETWORKX_AVAILABLE and self.graph is not None:
            if self.graph.has_edge(from_node, to_node):
                edge_data = self.graph.get_edge_data(from_node, to_node)
                return edge_data.get('forward_actions', [])
        return []

    def get_node_feature(self, node_id: int) -> Optional[np.ndarray]:
        """获取节点特征"""
        if node_id in self.nodes:
            return self.nodes[node_id].visual_feature
        return None

    def get_node(self, node_id: int) -> Optional[TopologicalNode]:
        """获取节点对象"""
        return self.nodes.get(node_id)

    def _update_node_feature(self, node_id: int, new_feature: np.ndarray, alpha: float = 0.1):
        """使用EMA更新节点特征"""
        if node_id in self.nodes:
            old_feature = self.nodes[node_id].visual_feature
            self.nodes[node_id].visual_feature = (
                alpha * new_feature + (1 - alpha) * old_feature
            )

    def _add_source_timestamp(self, node_id: int, source_timestamp: str, merge_time: float):
        """
        v2.1: 添加来源时间戳记录

        当观测被合并到现有节点时，记录来源图片的时间戳，
        用于追踪节点与原始图片的对应关系。

        Args:
            node_id: 目标节点ID
            source_timestamp: 来源图片时间戳
            merge_time: 合并操作的时间戳
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # 检查是否已存在相同时间戳（避免重复记录）
        existing_timestamps = {s.get('timestamp') for s in node.source_timestamps}
        if source_timestamp in existing_timestamps:
            logger.debug(f"[TopoMap] 来源时间戳 {source_timestamp} 已存在于节点 {node_id}")
            return

        # 添加新的来源记录
        node.source_timestamps.append({
            'timestamp': source_timestamp,
            'camera': 'front_1',  # 默认使用front_1相机
            'merged_at': merge_time,
            'is_initial': False  # 标记为合并添加
        })

        # 更新节点的最后更新时间
        node.updated_at = merge_time

        logger.debug(f"[TopoMap] 节点 {node_id} 添加来源时间戳: {source_timestamp} (总来源数: {len(node.source_timestamps)})")

    def _cleanup_old_nodes(self):
        """清理旧节点"""
        if len(self.nodes) <= self.config.max_nodes:
            return

        # 按访问次数和时间排序，删除最不重要的节点（保护关键帧）
        non_keyframe_nodes = [
            (nid, node) for nid, node in self.nodes.items()
            if not node.is_keyframe
        ]
        sorted_nodes = sorted(
            non_keyframe_nodes,
            key=lambda x: (x[1].visit_count, x[1].timestamp)
        )

        nodes_to_remove = len(self.nodes) - self.config.max_nodes
        for node_id, _ in sorted_nodes[:nodes_to_remove]:
            del self.nodes[node_id]
            if NETWORKX_AVAILABLE and self.graph is not None:
                if node_id in self.graph:
                    self.graph.remove_node(node_id)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        keyframe_count = sum(1 for n in self.nodes.values() if n.is_keyframe)
        return {
            "total_nodes": len(self.nodes),
            "keyframe_nodes": keyframe_count,
            "total_edges": self.graph.number_of_edges() if NETWORKX_AVAILABLE and self.graph else 0,
            "vpr_size": self.vpr.get_size(),
            "current_node": self.current_node_id
        }

    def get_graph_for_visualization(self) -> Dict:
        """
        获取用于可视化的图数据 - v2.1增强版（含前视特征嵌入）

        Returns:
            包含nodes和edges的字典，适合vis-network.js
        """
        vis_nodes = []
        vis_edges = []

        for node_id, node in self.nodes.items():
            # v2.4: 直接返回完整的512维向量（优先front_view_feature，回退visual_feature）
            front_view_embedding_vector = None
            has_front_view = False
            if node.front_view_feature is not None:
                feature_array = np.array(node.front_view_feature)
                # 直接返回完整的512维向量
                front_view_embedding_vector = [float(x) for x in feature_array]
                has_front_view = True
            elif node.visual_feature is not None:
                # v2.4: 回退使用visual_feature（从npy文件加载的特征）
                feature_array = np.array(node.visual_feature)
                # 检查是否为非零向量
                if np.any(feature_array != 0):
                    front_view_embedding_vector = [float(x) for x in feature_array]
                    has_front_view = True

            vis_nodes.append({
                'id': node_id,
                'label': f'N{node_id}',
                'is_keyframe': node.is_keyframe,
                'visit_count': node.visit_count,
                'semantic_labels': node.semantic_labels,
                'scene_description': node.scene_description or '',
                'is_current': node_id == self.current_node_id,
                # v2.0 新增字段
                'node_name': node.node_name,
                'navigation_instruction': node.navigation_instruction,
                'pixel_target': node.pixel_target,
                'created_at': node.created_at,
                'updated_at': node.updated_at,
                'has_front_view_feature': has_front_view,  # v2.4: 使用计算后的标志
                # v2.4 前视特征嵌入（完整512维向量，优先front_view_feature，回退visual_feature）
                'front_view_embedding': front_view_embedding_vector,
                # v2.1 节点来源追踪
                'source_timestamps': node.source_timestamps if hasattr(node, 'source_timestamps') else []
            })

        if NETWORKX_AVAILABLE and self.graph is not None:
            for u, v, data in self.graph.edges(data=True):
                vis_edges.append({
                    'from': u,
                    'to': v,
                    'weight': data.get('weight', 1.0)
                })

        return {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'current_node': self.current_node_id
        }

    def reset(self):
        """重置拓扑图"""
        self.nodes.clear()
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.clear()
        self.vpr.clear()
        self.next_node_id = 0
        self.last_node_id = None
        self.current_node_id = None
        # v2.1: 重置连续帧计数器
        self.consecutive_revisit_count = 0
        self.last_revisit_node_id = None
        logger.info("[TopoMap] 拓扑图已重置")

    def clear(self):
        """
        完全清空拓扑图及相关数据 v2.1

        清空：
        - 所有节点
        - 所有边
        - VPR索引
        - 语义图
        """
        self.reset()
        # 清空语义图
        if hasattr(self, 'semantic_graph') and self.semantic_graph is not None:
            self.semantic_graph.clear()
        logger.info("[TopoMap] 拓扑图已完全清空")

    def set_current_node(self, node_id: int):
        """手动设置当前节点"""
        if node_id in self.nodes:
            self.current_node_id = node_id
            self.last_node_id = node_id
            logger.info(f"[TopoMap] 当前节点设置为: {node_id}")
