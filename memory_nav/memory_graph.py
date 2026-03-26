#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.0 - 记忆拓扑图

管理记忆节点和边的拓扑结构，支持：
- 节点的增删改查
- 最短路径规划（Dijkstra/BFS）
- 语义查询（按名称搜索节点）
- 图的序列化/反序列化
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import deque
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .memory_models import MemoryNode, MemoryEdge, NavigationStep, NavigationPlan

logger = logging.getLogger(__name__)


class MemoryGraph:
    """
    记忆拓扑图管理器
    
    使用 NetworkX 实现图结构，支持：
    - 节点管理
    - 最短路径规划
    - 语义查询
    """
    
    def __init__(self):
        """初始化记忆图"""
        self.nodes: Dict[str, MemoryNode] = {}  # node_id -> MemoryNode
        self.name_to_id: Dict[str, str] = {}    # node_name/node_name_eng -> node_id (中英双语匹配)
        
        # NetworkX 图（用于路径规划）
        if NETWORKX_AVAILABLE:
            self.graph = nx.Graph()  # 无向图
        else:
            self.graph = None
            logger.warning("[MemoryGraph] NetworkX 不可用，路径规划功能受限")
        
        logger.info("[MemoryGraph] 初始化完成")
    
    def add_node(self, node: MemoryNode) -> bool:
        """
        添加节点到图中
        
        Args:
            node: MemoryNode 对象
            
        Returns:
            是否添加成功
        """
        if node.node_id in self.nodes:
            logger.warning(f"[MemoryGraph] 节点 {node.node_id} 已存在，将更新")
        
        self.nodes[node.node_id] = node
        self.name_to_id[node.node_name] = node.node_id
        if node.node_name_eng:
            self.name_to_id[node.node_name_eng] = node.node_id
        
        # 添加到 NetworkX 图
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_node(node.node_id, name=node.node_name)
            
            # 添加边
            for edge in node.edges:
                self.graph.add_edge(
                    node.node_id, 
                    edge.target_node_id,
                    weight=1.0  # 默认权重
                )
        
        logger.debug(f"[MemoryGraph] 添加节点: {node.node_id} ({node.node_name}), "
                    f"邻居数: {len(node.edges)}")
        return True
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_node_by_name(self, name: str) -> Optional[MemoryNode]:
        """按名称获取节点（语义匹配）"""
        node_id = self.name_to_id.get(name)
        if node_id:
            return self.nodes.get(node_id)
        
        # 模糊匹配
        for node_name, nid in self.name_to_id.items():
            if name in node_name or node_name in name:
                return self.nodes.get(nid)
        
        return None
    
    def search_nodes_by_name(self, query: str, limit: int = 5) -> List[MemoryNode]:
        """
        按名称搜索节点（支持模糊匹配）
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的节点列表
        """
        results = []
        query_lower = query.lower()
        
        for node_id, node in self.nodes.items():
            cn_name = node.node_name or ""
            en_name = getattr(node, 'node_name_eng', '') or ""
            
            # 完全匹配优先（中文或英文）
            if cn_name == query or en_name.lower() == query_lower:
                results.insert(0, node)
            # 包含匹配（中文或英文）
            elif (query_lower in cn_name.lower() or 
                  query_lower in en_name.lower()):
                results.append(node)
        
        return results[:limit]
    
    def find_shortest_path(self, start_id: str, goal_id: str) -> Optional[List[str]]:
        """
        查找最短路径（BFS/Dijkstra）
        
        Args:
            start_id: 起点节点ID
            goal_id: 终点节点ID
            
        Returns:
            节点ID序列，如 ['1', '3', '8', '12']，失败返回None
        """
        if start_id not in self.nodes or goal_id not in self.nodes:
            logger.warning(f"[MemoryGraph] 路径规划失败: 节点不存在 "
                          f"(start={start_id}, goal={goal_id})")
            return None
        
        if start_id == goal_id:
            return [start_id]
        
        # 使用 NetworkX
        if NETWORKX_AVAILABLE and self.graph is not None:
            try:
                path = nx.shortest_path(self.graph, start_id, goal_id)
                logger.info(f"[MemoryGraph] 路径规划成功: {' -> '.join(path)}")
                return path
            except nx.NetworkXNoPath:
                logger.warning(f"[MemoryGraph] 无法找到路径: {start_id} -> {goal_id}")
                return None
            except nx.NodeNotFound as e:
                logger.warning(f"[MemoryGraph] 节点不存在: {e}")
                return None
        
        # 回退到 BFS
        return self._bfs_path(start_id, goal_id)
    
    def _bfs_path(self, start_id: str, goal_id: str) -> Optional[List[str]]:
        """BFS 路径搜索（NetworkX 不可用时的后备方案）"""
        visited = set()
        queue = deque([(start_id, [start_id])])
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal_id:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            node = self.nodes.get(current)
            if node:
                for neighbor_id in node.get_neighbor_ids():
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    def plan_navigation(self, start_id: str, goal_id: str) -> NavigationPlan:
        """
        完整的导航规划
        
        Args:
            start_id: 起点节点ID
            goal_id: 终点节点ID
            
        Returns:
            NavigationPlan 对象
        """
        start_node = self.get_node(start_id)
        goal_node = self.get_node(goal_id)
        
        if not start_node:
            return NavigationPlan(
                start_node_id=start_id, start_node_name="",
                goal_node_id=goal_id, goal_node_name="",
                path=[], steps=[], total_steps=0,
                success=False, message=f"起点节点 {start_id} 不存在"
            )
        
        if not goal_node:
            return NavigationPlan(
                start_node_id=start_id, start_node_name=start_node.node_name,
                goal_node_id=goal_id, goal_node_name="",
                path=[], steps=[], total_steps=0,
                success=False, message=f"终点节点 {goal_id} 不存在"
            )
        
        # 查找最短路径
        path = self.find_shortest_path(start_id, goal_id)
        
        if not path:
            return NavigationPlan(
                start_node_id=start_id, start_node_name=start_node.node_name,
                goal_node_id=goal_id, goal_node_name=goal_node.node_name,
                path=[], steps=[], total_steps=0,
                success=False, message=f"无法找到从 {start_id} 到 {goal_id} 的路径"
            )
        
        # 构建导航步骤
        steps = []
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]
            edge = from_node.get_edge_to(to_id)
            
            if edge:
                step = NavigationStep(
                    from_node_id=from_id,
                    from_node_name=from_node.node_name,
                    from_node_name_eng=getattr(from_node, 'node_name_eng', ''),
                    to_node_id=to_id,
                    to_node_name=to_node.node_name,
                    to_node_name_eng=getattr(to_node, 'node_name_eng', ''),
                    camera_name=edge.camera_name,
                    landmark_name=edge.landmark_name,
                    landmark_name_eng=getattr(edge, 'landmark_name_eng', ''),
                    crop_image_paths=edge.crop_image_paths,
                    crop_image_path=edge.crop_image_path,
                    step_index=i,
                )
                steps.append(step)
            else:
                logger.warning(f"[MemoryGraph] 缺少边: {from_id} -> {to_id}")
        
        return NavigationPlan(
            start_node_id=start_id,
            start_node_name=start_node.node_name,
            start_node_name_eng=getattr(start_node, 'node_name_eng', ''),
            goal_node_id=goal_id,
            goal_node_name=goal_node.node_name,
            goal_node_name_eng=getattr(goal_node, 'node_name_eng', ''),
            path=path,
            steps=steps,
            total_steps=len(steps),
            success=True,
            message="路径规划成功"
        )
    
    def get_all_node_names(self) -> List[Tuple[str, str]]:
        """获取所有节点的 (id, name) 列表"""
        return [(node.node_id, node.node_name) for node in self.nodes.values()]
    
    def get_all_node_names_bilingual(self) -> list:
        """获取所有节点名称(含英文) [(node_id, node_name, node_name_eng), ...]"""
        return [(node.node_id, node.node_name, getattr(node, 'node_name_eng', '')) 
                for node in self.nodes.values()]
    
    def get_stats(self) -> Dict:
        """获取图的统计信息"""
        edge_count = sum(len(n.edges) for n in self.nodes.values())
        return {
            'total_nodes': len(self.nodes),
            'total_edges': edge_count,
            'node_names': [n.node_name for n in self.nodes.values()],
            'node_names_eng': [getattr(n, 'node_name_eng', '') for n in self.nodes.values()],
            'networkx_available': NETWORKX_AVAILABLE
        }
    
    def clear(self):
        """清空图"""
        self.nodes.clear()
        self.name_to_id.clear()
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.clear()
        logger.info("[MemoryGraph] 图已清空")
    
    def save(self, path: str):
        """
        保存图到文件
        
        Args:
            path: 保存路径 (.pkl)
        """
        save_data = {
            'nodes': {},
            'name_to_id': self.name_to_id
        }
        
        # 保存节点（不含图像数据）
        for node_id, node in self.nodes.items():
            node_data = {
                'node_id': node.node_id,
                'node_name': node.node_name,
                'node_name_eng': getattr(node, 'node_name_eng', ''),
                'camera_images': node.camera_images,
                'camera_features': {k: v.tolist() for k, v in node.camera_features.items()},
                'fused_feature': node.fused_feature.tolist() if node.fused_feature is not None else None,
                'edges': [e.to_dict() for e in node.edges],
                'base_path': node.base_path,
                'timestamp': node.timestamp
            }
            save_data['nodes'][node_id] = node_data
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"[MemoryGraph] 图已保存到 {path}")
    
    def load(self, path: str) -> bool:
        """
        从文件加载图
        
        Args:
            path: 加载路径 (.pkl)
            
        Returns:
            是否加载成功
        """
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.clear()
            self.name_to_id = save_data.get('name_to_id', {})
            
            for node_id, node_data in save_data.get('nodes', {}).items():
                # 重建边
                edges = []
                for e_data in node_data.get('edges', []):
                    edge = MemoryEdge(
                        target_node_id=e_data['target_node_id'],
                        target_node_name=e_data['target_node_name'],
                        target_node_name_eng=e_data.get('target_node_name_eng', ''),
                        camera_name=e_data.get('camera_name', ''),
                        landmark_name=e_data.get('landmark_name', ''),
                        landmark_name_eng=e_data.get('landmark_name_eng', ''),
                        crop_image_paths=e_data.get('crop_image_paths', {}),
                        crop_image_path=e_data.get('crop_image_path', ''),
                        big_box=tuple(e_data.get('big_box', ())),
                        mid_box=tuple(e_data.get('mid_box', ())),
                        small_box=tuple(e_data.get('small_box', ())),
                    )
                    edges.append(edge)
                
                # 重建特征
                camera_features = {}
                for k, v in node_data.get('camera_features', {}).items():
                    camera_features[k] = np.array(v, dtype=np.float32)
                
                fused_feature = None
                if node_data.get('fused_feature'):
                    fused_feature = np.array(node_data['fused_feature'], dtype=np.float32)
                
                # 创建节点
                node = MemoryNode(
                    node_id=node_data['node_id'],
                    node_name=node_data['node_name'],
                    node_name_eng=node_data.get('node_name_eng', ''),
                    camera_images=node_data.get('camera_images', {}),
                    camera_features=camera_features,
                    fused_feature=fused_feature,
                    edges=edges,
                    base_path=node_data.get('base_path', ''),
                    timestamp=node_data.get('timestamp', '')
                )
                
                self.add_node(node)
            
            logger.info(f"[MemoryGraph] 图已从 {path} 加载，共 {len(self.nodes)} 个节点")
            return True
            
        except Exception as e:
            logger.error(f"[MemoryGraph] 加载失败: {e}")
            return False
    
    def to_visualization_data(self) -> Dict:
        """
        导出用于可视化的数据（兼容 vis-network.js）
        """
        vis_nodes = []
        vis_edges = []
        
        for node_id, node in self.nodes.items():
            vis_nodes.append({
                'id': node_id,
                'label': f"{node_id}\n{node.node_name}" + (f"\n{node.node_name_eng}" if getattr(node, 'node_name_eng', '') else ""),
                'title': f"{node.node_name}" + (f" ({node.node_name_eng})" if getattr(node, 'node_name_eng', '') else ""),
                'has_features': node.fused_feature is not None
            })
        
        seen_edges = set()
        for node_id, node in self.nodes.items():
            for edge in node.edges:
                edge_key = tuple(sorted([node_id, edge.target_node_id]))
                if edge_key not in seen_edges:
                    vis_edges.append({
                        'from': node_id,
                        'to': edge.target_node_id,
                        'arrows': 'to,from'  # 双向箭头
                    })
                    seen_edges.add(edge_key)
        
        return {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'stats': self.get_stats()
        }
