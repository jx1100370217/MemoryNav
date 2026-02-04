#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - 增强版拓扑地图管理模块 v2.0

优化改进:
1. 基于图的最短路径规划，而非动作回放
2. 支持从任意位置检索记忆并规划最短路径
3. 相似度阈值节点合并功能
4. 支持VPR多视角独立编码

作者: Memory Navigation Team
日期: 2026-01-26
"""

import time
import logging
import heapq
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .config import MemoryNavigationConfig
from .vpr_enhanced import EnhancedVisualPlaceRecognition, MultiViewMatchResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTopologicalNode:
    """增强版拓扑图节点"""
    node_id: int
    view_features: Dict[str, np.ndarray]  # 各视角独立特征
    rgb_image: Optional[np.ndarray] = None
    surround_images: Dict[str, np.ndarray] = field(default_factory=dict)
    timestamp: float = 0.0
    instruction_context: Optional[str] = None
    scene_description: Optional[str] = None
    semantic_labels: List[str] = field(default_factory=list)
    pixel_target: Optional[List[float]] = None
    is_keyframe: bool = False
    visit_count: int = 1
    merged_node_ids: List[int] = field(default_factory=list)  # 被合并的节点ID列表


@dataclass
class PathPlanResult:
    """路径规划结果"""
    path: List[int]  # 节点ID序列
    total_distance: float  # 总距离
    estimated_actions: List[List[int]]  # 估计的动作序列
    waypoints: List[int]  # 关键路点


class EnhancedTopologicalMapManager:
    """
    增强版拓扑地图管理器

    核心改进:
    1. 基于图的最短路径规划
    2. 节点合并功能（相似度阈值）
    3. 多视角VPR支持
    4. 灵活的路径检索
    """

    def __init__(self, config: MemoryNavigationConfig):
        self.config = config

        # 使用NetworkX管理拓扑图
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            self._simple_edges: Dict[int, List[Tuple[int, float, List[int]]]] = {}  # 简单邻接表

        # 节点存储
        self.nodes: Dict[int, EnhancedTopologicalNode] = {}
        self.next_node_id = 0
        self.last_node_id: Optional[int] = None

        # 增强版VPR (多视角独立编码)
        self.vpr = EnhancedVisualPlaceRecognition(
            feature_dim=config.feature_dim,
            similarity_threshold=config.similarity_threshold
        )

        # 节点合并参数
        self.merge_threshold = getattr(config, 'node_merge_threshold', 0.95)  # 合并阈值
        self.enable_node_merging = True

        # 边权重参数
        self.default_edge_weight = 1.0

        logger.info(f"[TopoMap v2.0] 初始化完成: 合并阈值={self.merge_threshold}")

    def add_observation(self,
                        view_features: Dict[str, np.ndarray],
                        rgb_image: np.ndarray = None,
                        surround_images: Dict[str, np.ndarray] = None,
                        action_from_prev: List[int] = None,
                        instruction: str = None,
                        is_keyframe: bool = False,
                        pixel_target: List[float] = None,
                        scene_description: str = None,
                        semantic_labels: List[str] = None) -> Tuple[int, bool, Optional[MultiViewMatchResult]]:
        """
        添加新观测 - 多视角版本

        Args:
            view_features: {camera_id: feature} 各视角特征
            rgb_image: RGB图像
            surround_images: 环视相机图像字典
            action_from_prev: 从上一节点到当前的动作
            instruction: 当前任务指令
            is_keyframe: 是否为关键帧
            pixel_target: 像素目标
            scene_description: 场景描述
            semantic_labels: 语义标签

        Returns:
            (node_id, is_new_node, vpr_result)
        """
        current_time = time.time()

        # VPR回环检测 (多视角)
        vpr_result = self.vpr.is_revisited(
            view_features, current_time, self.config.min_time_gap,
            query_semantic_labels=semantic_labels
        )

        # 关键帧处理
        if is_keyframe:
            logger.info(f"[TopoMap v2.0] 关键帧检测到，强制创建新节点 (VPR结果: {vpr_result})")
            node_id = self._create_new_node(
                view_features=view_features,
                rgb_image=rgb_image,
                surround_images=surround_images,
                action_from_prev=action_from_prev,
                instruction=instruction,
                is_keyframe=True,
                pixel_target=pixel_target,
                scene_description=scene_description,
                semantic_labels=semantic_labels,
                timestamp=current_time
            )
            return node_id, True, vpr_result

        # 非关键帧: 检查VPR回环
        if vpr_result is not None:
            matched_node_id = vpr_result.node_id
            logger.info(f"[TopoMap v2.0] VPR回环检测: 匹配节点 {matched_node_id} "
                       f"(sim={vpr_result.combined_similarity:.3f}, views={vpr_result.matched_views})")

            # 更新已有节点
            if matched_node_id in self.nodes:
                self.nodes[matched_node_id].visit_count += 1

            # 添加边
            if self.last_node_id is not None and self.last_node_id != matched_node_id:
                self._add_edge(self.last_node_id, matched_node_id, action_from_prev or [], self.default_edge_weight)

            self.last_node_id = matched_node_id
            return matched_node_id, False, vpr_result

        # 检查节点合并
        if self.enable_node_merging:
            merge_candidate = self._find_merge_candidate(view_features)
            if merge_candidate is not None:
                logger.info(f"[TopoMap v2.0] 节点合并: 与节点 {merge_candidate} 合并")

                if merge_candidate in self.nodes:
                    self.nodes[merge_candidate].visit_count += 1

                if self.last_node_id is not None and self.last_node_id != merge_candidate:
                    self._add_edge(self.last_node_id, merge_candidate, action_from_prev or [], self.default_edge_weight)

                self.last_node_id = merge_candidate
                return merge_candidate, False, None

        # 创建新节点
        node_id = self._create_new_node(
            view_features=view_features,
            rgb_image=rgb_image,
            surround_images=surround_images,
            action_from_prev=action_from_prev,
            instruction=instruction,
            is_keyframe=is_keyframe,
            pixel_target=pixel_target,
            scene_description=scene_description,
            semantic_labels=semantic_labels,
            timestamp=current_time
        )
        return node_id, True, None

    def _create_new_node(self,
                        view_features: Dict[str, np.ndarray],
                        rgb_image: np.ndarray,
                        surround_images: Dict[str, np.ndarray],
                        action_from_prev: List[int],
                        instruction: str,
                        is_keyframe: bool,
                        pixel_target: List[float],
                        scene_description: str,
                        semantic_labels: List[str],
                        timestamp: float) -> int:
        """创建新节点"""
        node_id = self.next_node_id

        # 创建节点对象
        node = EnhancedTopologicalNode(
            node_id=node_id,
            view_features={k: v.copy() for k, v in view_features.items()},
            rgb_image=rgb_image.copy() if rgb_image is not None else None,
            surround_images={k: v.copy() for k, v in (surround_images or {}).items()},
            timestamp=timestamp,
            instruction_context=instruction,
            scene_description=scene_description,
            semantic_labels=semantic_labels or [],
            pixel_target=pixel_target,
            is_keyframe=is_keyframe
        )

        self.nodes[node_id] = node

        # 添加到图
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_node(node_id)
        else:
            self._simple_edges[node_id] = []

        # 添加到VPR
        self.vpr.add_node(
            node_id=node_id,
            view_features=view_features,
            timestamp=timestamp,
            semantic_labels=semantic_labels,
            scene_description=scene_description or "",
            is_keyframe=is_keyframe
        )

        # 添加边
        if self.last_node_id is not None and action_from_prev is not None:
            self._add_edge(self.last_node_id, node_id, action_from_prev, self.default_edge_weight)

        logger.info(f"[TopoMap v2.0] 新增节点: {node_id} (总数={len(self.nodes)}, 关键帧={is_keyframe})")

        self.last_node_id = node_id
        self.next_node_id += 1

        # 内存管理
        if len(self.nodes) > self.config.max_nodes:
            self._cleanup_old_nodes()

        return node_id

    def _find_merge_candidate(self, view_features: Dict[str, np.ndarray]) -> Optional[int]:
        """
        查找可以合并的节点

        Args:
            view_features: 当前帧的各视角特征

        Returns:
            可以合并的节点ID，如果没有则返回None
        """
        if len(self.nodes) == 0:
            return None

        # 多视角搜索
        results = self.vpr.search_multi_view(view_features, k=3)

        for result in results:
            if result.combined_similarity >= self.merge_threshold:
                return result.node_id

        return None

    def merge_similar_nodes(self, similarity_threshold: float = None) -> int:
        """
        合并相似度超过阈值的节点

        Args:
            similarity_threshold: 合并阈值 (None则使用默认值)

        Returns:
            合并的节点对数
        """
        threshold = similarity_threshold or self.merge_threshold
        merged_count = 0

        # 获取所有节点ID
        node_ids = list(self.nodes.keys())
        merged_nodes: Set[int] = set()

        for i, node_id in enumerate(node_ids):
            if node_id in merged_nodes:
                continue

            node = self.nodes[node_id]
            if not node.view_features:
                continue

            # 查找相似节点
            results = self.vpr.search_multi_view(node.view_features, k=10)

            for result in results:
                if result.node_id == node_id or result.node_id in merged_nodes:
                    continue

                if result.combined_similarity >= threshold:
                    # 合并节点
                    self._merge_nodes(node_id, result.node_id)
                    merged_nodes.add(result.node_id)
                    merged_count += 1
                    logger.info(f"[TopoMap v2.0] 合并节点: {result.node_id} -> {node_id} "
                               f"(sim={result.combined_similarity:.3f})")

        return merged_count

    def _merge_nodes(self, target_node_id: int, source_node_id: int):
        """
        将source节点合并到target节点

        Args:
            target_node_id: 目标节点ID
            source_node_id: 源节点ID (将被删除)
        """
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            return

        source_node = self.nodes[source_node_id]
        target_node = self.nodes[target_node_id]

        # 记录被合并的节点
        target_node.merged_node_ids.append(source_node_id)
        target_node.visit_count += source_node.visit_count

        # 转移边关系
        if NETWORKX_AVAILABLE and self.graph is not None:
            # 入边
            for pred in list(self.graph.predecessors(source_node_id)):
                if pred != target_node_id:
                    edge_data = self.graph.get_edge_data(pred, source_node_id)
                    if not self.graph.has_edge(pred, target_node_id):
                        self.graph.add_edge(pred, target_node_id, **edge_data)
            # 出边
            for succ in list(self.graph.successors(source_node_id)):
                if succ != target_node_id:
                    edge_data = self.graph.get_edge_data(source_node_id, succ)
                    if not self.graph.has_edge(target_node_id, succ):
                        self.graph.add_edge(target_node_id, succ, **edge_data)
            # 删除source节点
            self.graph.remove_node(source_node_id)

        # 从节点字典中删除
        del self.nodes[source_node_id]

    def _add_edge(self, from_node: int, to_node: int, actions: List[int], weight: float = 1.0):
        """添加边"""
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_edge(from_node, to_node, actions=actions, weight=weight)
            # 添加反向边（双向可达）
            reversed_actions = self._reverse_actions(actions)
            if not self.graph.has_edge(to_node, from_node):
                self.graph.add_edge(to_node, from_node, actions=reversed_actions, weight=weight)
        else:
            if from_node not in self._simple_edges:
                self._simple_edges[from_node] = []
            self._simple_edges[from_node].append((to_node, weight, actions))
            # 反向边
            if to_node not in self._simple_edges:
                self._simple_edges[to_node] = []
            reversed_actions = self._reverse_actions(actions)
            self._simple_edges[to_node].append((from_node, weight, reversed_actions))

    def _reverse_actions(self, actions: List[int]) -> List[int]:
        """反转动作序列"""
        action_map = {0: 0, 1: 1, 2: 3, 3: 2, 5: 5}
        return [action_map.get(a, a) for a in reversed(actions)]

    def find_shortest_path(self, start_node: int, goal_node: int) -> Optional[PathPlanResult]:
        """
        使用Dijkstra算法查找最短路径

        Args:
            start_node: 起始节点ID
            goal_node: 目标节点ID

        Returns:
            PathPlanResult 或 None
        """
        if start_node == goal_node:
            return PathPlanResult(path=[start_node], total_distance=0.0,
                                 estimated_actions=[], waypoints=[start_node])

        if NETWORKX_AVAILABLE and self.graph is not None:
            try:
                path = nx.shortest_path(self.graph, start_node, goal_node, weight='weight')
                total_distance = nx.shortest_path_length(self.graph, start_node, goal_node, weight='weight')

                # 获取动作序列
                actions = []
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    if edge_data and 'actions' in edge_data:
                        actions.append(edge_data['actions'])
                    else:
                        actions.append([1])  # 默认前进

                # 提取关键路点
                waypoints = self._extract_waypoints(path)

                return PathPlanResult(
                    path=path,
                    total_distance=total_distance,
                    estimated_actions=actions,
                    waypoints=waypoints
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
        else:
            return self._dijkstra_simple(start_node, goal_node)

    def _dijkstra_simple(self, start: int, goal: int) -> Optional[PathPlanResult]:
        """简单Dijkstra实现"""
        if start not in self._simple_edges or goal not in self._simple_edges:
            return None

        distances = {start: 0.0}
        previous = {}
        edge_actions = {}
        pq = [(0.0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # 重建路径
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = previous.get(node)
                path.reverse()

                # 获取动作
                actions = []
                for i in range(len(path) - 1):
                    key = (path[i], path[i+1])
                    actions.append(edge_actions.get(key, [1]))

                return PathPlanResult(
                    path=path,
                    total_distance=current_dist,
                    estimated_actions=actions,
                    waypoints=self._extract_waypoints(path)
                )

            for neighbor, weight, acts in self._simple_edges.get(current, []):
                if neighbor in visited:
                    continue
                new_dist = current_dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    edge_actions[(current, neighbor)] = acts
                    heapq.heappush(pq, (new_dist, neighbor))

        return None

    def _extract_waypoints(self, path: List[int]) -> List[int]:
        """提取关键路点（关键帧节点）"""
        waypoints = []
        for node_id in path:
            if node_id in self.nodes:
                if self.nodes[node_id].is_keyframe:
                    waypoints.append(node_id)

        # 始终包含起点和终点
        if path and path[0] not in waypoints:
            waypoints.insert(0, path[0])
        if path and path[-1] not in waypoints:
            waypoints.append(path[-1])

        return waypoints

    def find_path_to_location(self, current_features: Dict[str, np.ndarray],
                             target_node_id: int) -> Optional[PathPlanResult]:
        """
        从当前位置（通过视觉特征定位）规划到目标节点的路径

        Args:
            current_features: 当前帧的各视角特征
            target_node_id: 目标节点ID

        Returns:
            PathPlanResult 或 None
        """
        # 首先定位当前位置
        results = self.vpr.search_multi_view(current_features, k=1)
        if not results:
            logger.warning("[TopoMap v2.0] 无法定位当前位置")
            return None

        current_node_id = results[0].node_id
        logger.info(f"[TopoMap v2.0] 当前位置定位: 节点 {current_node_id} "
                   f"(sim={results[0].combined_similarity:.3f})")

        return self.find_shortest_path(current_node_id, target_node_id)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, float, List[int]]]:
        """获取节点的邻居"""
        if NETWORKX_AVAILABLE and self.graph is not None:
            neighbors = []
            for neighbor in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                weight = edge_data.get('weight', 1.0)
                actions = edge_data.get('actions', [])
                neighbors.append((neighbor, weight, actions))
            return neighbors
        else:
            return self._simple_edges.get(node_id, [])

    def get_all_keyframes(self) -> List[int]:
        """获取所有关键帧节点ID"""
        return [node_id for node_id, node in self.nodes.items() if node.is_keyframe]

    def get_graph_info(self) -> Dict:
        """获取图的基本信息"""
        if NETWORKX_AVAILABLE and self.graph is not None:
            return {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
                'keyframe_count': len(self.get_all_keyframes())
            }
        else:
            num_edges = sum(len(edges) for edges in self._simple_edges.values())
            return {
                'num_nodes': len(self.nodes),
                'num_edges': num_edges // 2,  # 双向边只算一次
                'is_connected': False,
                'keyframe_count': len(self.get_all_keyframes())
            }

    def export_to_dict(self) -> Dict:
        """导出拓扑图为字典格式"""
        nodes_data = {}
        for node_id, node in self.nodes.items():
            nodes_data[node_id] = {
                'node_id': node.node_id,
                'timestamp': node.timestamp,
                'is_keyframe': node.is_keyframe,
                'visit_count': node.visit_count,
                'semantic_labels': node.semantic_labels,
                'scene_description': node.scene_description,
                'merged_nodes': node.merged_node_ids
            }

        edges_data = []
        if NETWORKX_AVAILABLE and self.graph is not None:
            for u, v, data in self.graph.edges(data=True):
                edges_data.append({
                    'from': u,
                    'to': v,
                    'weight': data.get('weight', 1.0),
                    'actions': data.get('actions', [])
                })
        else:
            visited_edges = set()
            for from_node, neighbors in self._simple_edges.items():
                for to_node, weight, actions in neighbors:
                    edge_key = (min(from_node, to_node), max(from_node, to_node))
                    if edge_key not in visited_edges:
                        edges_data.append({
                            'from': from_node,
                            'to': to_node,
                            'weight': weight,
                            'actions': actions
                        })
                        visited_edges.add(edge_key)

        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'info': self.get_graph_info()
        }

    def _cleanup_old_nodes(self):
        """清理旧节点"""
        if len(self.nodes) <= self.config.max_nodes:
            return

        # 按访问次数和时间排序，保留关键帧
        sorted_nodes = sorted(
            [(k, v) for k, v in self.nodes.items() if not v.is_keyframe],
            key=lambda x: (x[1].visit_count, x[1].timestamp)
        )

        nodes_to_remove = len(self.nodes) - self.config.max_nodes
        for node_id, _ in sorted_nodes[:nodes_to_remove]:
            del self.nodes[node_id]
            if NETWORKX_AVAILABLE and self.graph is not None:
                if node_id in self.graph:
                    self.graph.remove_node(node_id)

    def reset(self):
        """重置拓扑图"""
        self.nodes.clear()
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.clear()
        else:
            self._simple_edges.clear()
        self.vpr.clear()
        self.next_node_id = 0
        self.last_node_id = None


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建配置
    class SimpleConfig:
        feature_dim = 512
        similarity_threshold = 0.78
        min_time_gap = 0.5
        max_nodes = 1000
        node_merge_threshold = 0.95

    config = SimpleConfig()
    topo_map = EnhancedTopologicalMapManager(config)

    # 添加测试节点
    for i in range(5):
        view_features = {
            'camera_1': np.random.randn(512).astype('float32'),
            'camera_2': np.random.randn(512).astype('float32'),
            'camera_3': np.random.randn(512).astype('float32'),
            'camera_4': np.random.randn(512).astype('float32'),
        }
        topo_map.add_observation(
            view_features=view_features,
            action_from_prev=[1] if i > 0 else None,
            is_keyframe=(i % 2 == 0),
            semantic_labels=['test', f'node_{i}']
        )

    # 测试路径规划
    path_result = topo_map.find_shortest_path(0, 4)
    if path_result:
        print(f"\n路径规划结果:")
        print(f"  路径: {path_result.path}")
        print(f"  总距离: {path_result.total_distance}")
        print(f"  关键路点: {path_result.waypoints}")

    # 打印图信息
    info = topo_map.get_graph_info()
    print(f"\n拓扑图信息: {info}")

    # 导出
    export_data = topo_map.export_to_dict()
    print(f"\n导出节点数: {len(export_data['nodes'])}")
    print(f"导出边数: {len(export_data['edges'])}")
