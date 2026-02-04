#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - GraphRAG语义图管理模块 v2.0

GraphRAG 风格的语义地图管理器，实现语义节点存储和检索。

v2.0 新增功能:
1. 同义词扩展: 支持中英文同义词映射
2. 模糊匹配: 基于编辑距离的近似字符串匹配
3. 向量语义检索: 基于特征向量的语义相似度计算
4. 层次化搜索: 先粗筛后精排的两阶段检索
5. 场景相似度评估: 综合多维度的场景匹配
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# 尝试导入networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# 尝试导入FAISS用于向量检索
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .config import MemoryNavigationConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 同义词词典 (中英文室内场景常用词)
# 注意: 同义词应该排在更具体的词条之前，以避免歧义
# ============================================================================
SYNONYM_DICT = {
    # 房间/位置类 (优先级高，应该先匹配)
    '前台': ['reception', 'front desk', 'reception desk', '接待处', '接待台', '前台接待', '服务台'],
    '走廊': ['corridor', '过道', '通道', '廊道', 'hallway', 'passage'],
    '门': ['door', '入口', '出口', '门口', 'entrance', 'exit', 'gateway'],
    '窗户': ['窗', 'window', '玻璃窗'],
    '厨房': ['kitchen', '烹饪区', '厨区'],
    '卧室': ['bedroom', '睡房', '卧房'],
    '卫生间': ['洗手间', '厕所', '浴室', 'bathroom', 'toilet', 'restroom', 'washroom'],
    '客厅': ['living room', '起居室', '大厅', 'lounge'],
    '办公室': ['office', '办公区', '工作室', 'workspace'],
    '会议室': ['meeting room', 'conference room', '会客室', '议事厅', '会议厅'],

    # 家具类
    '桌子': ['table', '工作台', '餐桌'],
    '办公桌': ['desk', 'work desk', '工作桌'],
    '椅子': ['椅', '凳子', '座位', '沙发椅', 'chair', 'seat', 'stool'],
    '沙发': ['sofa', '长椅', '休息区', 'couch', 'settee'],
    '床': ['床铺', '卧床', 'bed'],
    '柜子': ['柜', '橱柜', '衣柜', '储物柜', 'cabinet', 'closet', 'cupboard'],

    # 物品类
    '电脑': ['computer', '笔记本', '显示器', 'PC', 'laptop', 'monitor'],
    '电视': ['TV', '电视机', 'television', 'screen'],
    '灯': ['light', '灯具', '照明', '台灯', '吊灯', 'lamp', 'lighting'],
    '植物': ['plant', '花', '绿植', '盆栽', '花卉', 'flower', 'greenery'],
    '书架': ['bookshelf', '书柜', '置物架', 'shelf'],

    # 位置类
    '左边': ['left', '左侧', '左手边', 'left side'],
    '右边': ['right', '右侧', '右手边', 'right side'],
    '前面': ['front', '前方', '正前方', 'ahead'],
    '后面': ['back', '后方', '背后', 'behind', 'rear'],
    '旁边': ['near', 'beside', '附近', '边上', 'next to', 'nearby'],
    '起点': ['start', 'origin', '出发点', '起始位置', 'starting point'],
    '当前位置': ['current location', 'current position', 'here', '这里'],
}

# 构建反向索引 (同义词 -> 标准词)
# 注意: 更长的短语优先匹配，避免"front desk"被"desk"先匹配
SYNONYM_REVERSE = {}

# 首先收集所有同义词并按长度排序
all_synonyms = []
for standard, synonyms in SYNONYM_DICT.items():
    all_synonyms.append((standard.lower(), standard))
    for syn in synonyms:
        all_synonyms.append((syn.lower(), standard))

# 按长度降序排序，确保长短语优先
all_synonyms.sort(key=lambda x: len(x[0]), reverse=True)

# 构建反向索引
for syn_lower, standard in all_synonyms:
    if syn_lower not in SYNONYM_REVERSE:  # 只保留第一次出现的映射
        SYNONYM_REVERSE[syn_lower] = standard


class SemanticGraphManager:
    """
    GraphRAG 风格的语义地图管理器 v2.0

    基于开发文档设计，实现语义节点存储和检索。

    v2.0 新增功能:
    1. 同义词扩展搜索
    2. 模糊字符串匹配
    3. 向量语义检索
    4. 综合评分排序
    """
    def __init__(self, config: MemoryNavigationConfig):
        self.config = config

        if NETWORKX_AVAILABLE:
            self.semantic_graph = nx.DiGraph()
        else:
            self.semantic_graph = None

        # 语义节点元数据
        self.node_metadata: Dict[int, Dict] = {}

        # 语义标签索引 (标签 -> 节点ID列表)
        self.label_index: Dict[str, List[int]] = {}

        # [新增] 标准化标签索引 (标准词 -> 节点ID列表)
        self.normalized_label_index: Dict[str, List[int]] = {}

        # 场景描述索引
        self.description_index: Dict[int, str] = {}

        # [新增] 视觉特征索引 (用于向量语义检索)
        self.visual_features: Dict[int, np.ndarray] = {}
        self.feature_dim = config.feature_dim if hasattr(config, 'feature_dim') else 512

        # [v2.0新增] 前视图特征索引 (导航推理模型使用的front_1图片特征)
        self.front_view_features: Dict[int, np.ndarray] = {}

        # [新增] FAISS向量索引
        self.faiss_index = None
        self.faiss_node_ids: List[int] = []
        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)

        logger.info("GraphRAG语义地图管理器 v2.0 已初始化 (支持同义词扩展和向量检索)")

    def add_semantic_node(self, node_id: int, scene_description: str,
                          semantic_labels: List[str], visual_feature: np.ndarray = None,
                          node_name: str = None, navigation_instruction: str = None,
                          front_view_feature: np.ndarray = None, pixel_target: List[float] = None):
        """
        添加语义节点到图中 - 增强版 v2.0

        Args:
            node_id: 节点ID
            scene_description: VLM生成的场景描述
            semantic_labels: 语义标签列表（将自动去重）
            visual_feature: 视觉特征向量
            node_name: 节点名称（简短标识）
            navigation_instruction: 导航指令
            front_view_feature: 前视图特征编码
            pixel_target: 像素目标坐标
        """
        # 语义标签去重并保持顺序
        unique_labels = []
        seen = set()
        for label in semantic_labels:
            label_clean = label.strip()
            if label_clean and label_clean.lower() not in seen:
                unique_labels.append(label_clean)
                seen.add(label_clean.lower())

        current_time = time.time()

        # 添加节点元数据
        self.node_metadata[node_id] = {
            'scene_description': scene_description,
            'semantic_labels': unique_labels,
            'timestamp': current_time,
            'visit_count': 1,
            # v2.0 新增字段
            'node_name': node_name,
            'navigation_instruction': navigation_instruction,
            'pixel_target': pixel_target,
            'created_at': current_time,
            'updated_at': current_time,
            'pixel_target_history': [{'target': pixel_target, 'timestamp': current_time}] if pixel_target else [],
            'has_front_view_feature': front_view_feature is not None
        }

        # [v2.0新增] 存储前视图特征
        if front_view_feature is not None:
            self.front_view_features[node_id] = front_view_feature.copy()

        # 更新场景描述索引
        self.description_index[node_id] = scene_description

        # 更新标签索引
        for label in semantic_labels:
            label_lower = label.lower()
            if label_lower not in self.label_index:
                self.label_index[label_lower] = []
            if node_id not in self.label_index[label_lower]:
                self.label_index[label_lower].append(node_id)

            # [新增] 更新标准化标签索引
            normalized = self._normalize_label(label)
            if normalized:
                if normalized not in self.normalized_label_index:
                    self.normalized_label_index[normalized] = []
                if node_id not in self.normalized_label_index[normalized]:
                    self.normalized_label_index[normalized].append(node_id)

        # [新增] 存储视觉特征并添加到FAISS索引
        if visual_feature is not None:
            self.visual_features[node_id] = visual_feature.copy()
            if FAISS_AVAILABLE and self.faiss_index is not None:
                feature_normalized = visual_feature.astype('float32')
                feature_normalized = feature_normalized / (np.linalg.norm(feature_normalized) + 1e-8)
                self.faiss_index.add(feature_normalized.reshape(1, -1))
                self.faiss_node_ids.append(node_id)

        # 添加到图
        if NETWORKX_AVAILABLE and self.semantic_graph is not None:
            self.semantic_graph.add_node(
                node_id,
                description=scene_description,
                labels=semantic_labels
            )

        logger.info(f"[GraphRAG] 添加语义节点 {node_id}: 标签={semantic_labels}")

    def _normalize_label(self, label: str) -> Optional[str]:
        """
        [新增] 标准化标签 - 使用同义词词典

        Args:
            label: 原始标签

        Returns:
            标准化后的标签，如果无法标准化则返回None
        """
        label_lower = label.lower().strip()
        return SYNONYM_REVERSE.get(label_lower, label_lower)

    def _expand_query_with_synonyms(self, query: str) -> Set[str]:
        """
        [新增] 扩展查询词 - 添加同义词

        Args:
            query: 原始查询

        Returns:
            扩展后的查询词集合
        """
        expanded = set()
        query_lower = query.lower()
        expanded.add(query_lower)

        # 分词
        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', query_lower)

        for word in words:
            expanded.add(word)
            # 查找同义词
            standard = SYNONYM_REVERSE.get(word)
            if standard:
                expanded.add(standard.lower())
                # 添加该标准词的所有同义词
                if standard in SYNONYM_DICT:
                    for syn in SYNONYM_DICT[standard]:
                        expanded.add(syn.lower())

        return expanded

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        [新增] 计算编辑距离 (Levenshtein Distance)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _fuzzy_match_score(self, query: str, target: str, max_distance: int = 2) -> float:
        """
        [新增] 模糊匹配评分

        Args:
            query: 查询字符串
            target: 目标字符串
            max_distance: 允许的最大编辑距离

        Returns:
            匹配分数 (0-1)，0表示不匹配
        """
        if not query or not target:
            return 0.0

        query_lower = query.lower()
        target_lower = target.lower()

        # 精确匹配
        if query_lower == target_lower:
            return 1.0

        # 包含匹配
        if query_lower in target_lower or target_lower in query_lower:
            return 0.8

        # 编辑距离匹配
        distance = self._levenshtein_distance(query_lower, target_lower)
        max_len = max(len(query_lower), len(target_lower))

        if distance <= max_distance:
            return 1.0 - (distance / max_len)

        return 0.0

    def add_semantic_edge(self, from_node: int, to_node: int,
                          action: List[int], description: str = ""):
        """
        添加语义边到图中

        Args:
            from_node: 源节点
            to_node: 目标节点
            action: 动作序列
            description: 边的语义描述
        """
        if NETWORKX_AVAILABLE and self.semantic_graph is not None:
            self.semantic_graph.add_edge(
                from_node, to_node,
                action=action,
                description=description,
                weight=len(action) if action else 1
            )

    def semantic_search(self, query: str, k: int = 5,
                        use_synonyms: bool = True,
                        use_fuzzy: bool = True) -> List[Tuple[int, float, Dict]]:
        """
        基于语义查询搜索匹配的节点 - 增强版 v2.0

        支持:
        1. 同义词扩展搜索
        2. 模糊字符串匹配
        3. 多维度综合评分

        Args:
            query: 查询文本（如 "走廊", "电梯附近", "reception"）
            k: 返回最多k个结果
            use_synonyms: 是否使用同义词扩展
            use_fuzzy: 是否使用模糊匹配

        Returns:
            [(node_id, score, match_info), ...] 匹配结果列表
            match_info包含: label_score, description_score, fuzzy_score, matched_labels
        """
        results = []

        # [新增] 扩展查询词
        if use_synonyms:
            expanded_queries = self._expand_query_with_synonyms(query)
        else:
            expanded_queries = {query.lower()}

        query_words = set()
        for q in expanded_queries:
            query_words.update(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', q.lower()))

        for node_id, metadata in self.node_metadata.items():
            label_score = 0.0
            description_score = 0.0
            fuzzy_score = 0.0
            matched_labels = []

            # 标签匹配 (增强版)
            labels = metadata.get('semantic_labels', [])
            for label in labels:
                label_lower = label.lower()

                # 精确匹配
                for eq in expanded_queries:
                    if eq == label_lower or eq in label_lower or label_lower in eq:
                        label_score += 1.5
                        if label not in matched_labels:
                            matched_labels.append(label)
                        break

                # 词级别匹配
                for word in query_words:
                    if word in label_lower:
                        label_score += 0.5
                        if label not in matched_labels:
                            matched_labels.append(label)

                # [新增] 标准化标签匹配
                normalized = self._normalize_label(label)
                for eq in expanded_queries:
                    normalized_eq = self._normalize_label(eq)
                    if normalized and normalized_eq and normalized == normalized_eq:
                        label_score += 1.0
                        if label not in matched_labels:
                            matched_labels.append(label)

                # [新增] 模糊匹配
                if use_fuzzy:
                    for eq in expanded_queries:
                        fs = self._fuzzy_match_score(eq, label_lower)
                        if fs > 0.6:
                            fuzzy_score += fs
                            if label not in matched_labels:
                                matched_labels.append(label)

            # 描述匹配
            description = metadata.get('scene_description', '')
            if description:
                desc_lower = description.lower()
                for eq in expanded_queries:
                    if eq in desc_lower:
                        description_score += 0.5
                for word in query_words:
                    if word in desc_lower:
                        description_score += 0.2

            # 综合评分
            total_score = label_score + description_score * 0.5 + fuzzy_score * 0.3

            if total_score > 0:
                match_info = {
                    'label_score': label_score,
                    'description_score': description_score,
                    'fuzzy_score': fuzzy_score,
                    'matched_labels': matched_labels,
                    'expanded_queries': list(expanded_queries)
                }
                results.append((node_id, total_score, match_info))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def semantic_search_simple(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        简化版语义搜索 (向后兼容)

        Args:
            query: 查询文本
            k: 返回最多k个结果

        Returns:
            [(node_id, score), ...] 匹配结果列表
        """
        results = self.semantic_search(query, k)
        return [(node_id, score) for node_id, score, _ in results]

    def vector_semantic_search(self, query_feature: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        [新增] 基于向量的语义搜索

        Args:
            query_feature: 查询特征向量
            k: 返回top-k结果

        Returns:
            [(node_id, similarity), ...] 匹配结果列表
        """
        if not FAISS_AVAILABLE or self.faiss_index is None or len(self.faiss_node_ids) == 0:
            return []

        query_normalized = query_feature.astype('float32')
        query_normalized = query_normalized / (np.linalg.norm(query_normalized) + 1e-8)

        k = min(k, len(self.faiss_node_ids))
        if k == 0:
            return []

        distances, indices = self.faiss_index.search(query_normalized.reshape(1, -1), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.faiss_node_ids):
                node_id = self.faiss_node_ids[idx]
                similarity = float(distances[0][i])
                results.append((node_id, similarity))

        return results

    def hybrid_search(self, query: str, query_feature: np.ndarray = None,
                      k: int = 5, text_weight: float = 0.6) -> List[Tuple[int, float, Dict]]:
        """
        [新增] 混合搜索 - 结合文本语义和向量语义

        Args:
            query: 文本查询
            query_feature: 视觉特征向量 (可选)
            k: 返回top-k结果
            text_weight: 文本搜索权重 (0-1)

        Returns:
            [(node_id, combined_score, match_info), ...]
        """
        # 文本语义搜索
        text_results = self.semantic_search(query, k=k*2)
        text_scores = {node_id: score for node_id, score, _ in text_results}
        text_infos = {node_id: info for node_id, _, info in text_results}

        # 归一化文本分数
        max_text_score = max(text_scores.values()) if text_scores else 1.0
        text_scores = {k: v / max_text_score for k, v in text_scores.items()}

        # 向量语义搜索
        vector_scores = {}
        if query_feature is not None:
            vector_results = self.vector_semantic_search(query_feature, k=k*2)
            vector_scores = {node_id: sim for node_id, sim in vector_results}

        # 合并结果
        all_nodes = set(text_scores.keys()) | set(vector_scores.keys())
        combined_results = []

        vector_weight = 1.0 - text_weight

        for node_id in all_nodes:
            ts = text_scores.get(node_id, 0.0)
            vs = vector_scores.get(node_id, 0.0)

            if query_feature is not None:
                combined_score = text_weight * ts + vector_weight * vs
            else:
                combined_score = ts

            match_info = text_infos.get(node_id, {})
            match_info['text_score'] = ts
            match_info['vector_score'] = vs

            combined_results.append((node_id, combined_score, match_info))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]

    def find_path_by_description(self, start_node: int,
                                 target_description: str) -> Optional[List[int]]:
        """
        基于语义描述查找路径

        Args:
            start_node: 起始节点
            target_description: 目标位置的描述

        Returns:
            节点ID路径列表，如果找不到返回None
        """
        # 搜索匹配描述的节点
        matching_nodes = self.semantic_search(target_description, k=5)

        if not matching_nodes:
            return None

        # 找到最佳匹配节点的最短路径
        best_path = None
        best_length = float('inf')

        for target_id, score in matching_nodes:
            if target_id == start_node:
                continue

            if NETWORKX_AVAILABLE and self.semantic_graph is not None:
                try:
                    path = nx.shortest_path(self.semantic_graph, start_node, target_id)
                    if len(path) < best_length:
                        best_path = path
                        best_length = len(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        return best_path

    def get_nodes_by_label(self, label: str) -> List[int]:
        """根据标签获取节点列表"""
        return self.label_index.get(label.lower(), [])

    def get_node_info(self, node_id: int) -> Optional[Dict]:
        """获取节点的语义信息"""
        return self.node_metadata.get(node_id)

    def export_graph_data(self) -> Dict:
        """导出图数据用于可视化 - v2.1增强版（含前视特征向量）"""
        nodes = []
        edges = []

        for node_id, metadata in self.node_metadata.items():
            # 获取前视图特征向量
            front_view_feature = None
            front_view_embedding_info = None
            if hasattr(self, 'front_view_features') and node_id in self.front_view_features:
                front_view_feature = self.front_view_features[node_id]
                # 生成嵌入摘要信息用于展示
                if front_view_feature is not None:
                    feature_array = np.array(front_view_feature)
                    front_view_embedding_info = {
                        'dim': len(feature_array),
                        'norm': float(np.linalg.norm(feature_array)),
                        'mean': float(np.mean(feature_array)),
                        'std': float(np.std(feature_array)),
                        'min': float(np.min(feature_array)),
                        'max': float(np.max(feature_array)),
                        # 前10个值作为预览
                        'preview': [float(x) for x in feature_array[:10]]
                    }

            has_front_view = front_view_feature is not None or metadata.get('has_front_view_feature', False)

            nodes.append({
                'id': node_id,
                'description': metadata.get('scene_description', ''),
                'labels': metadata.get('semantic_labels', []),
                'visit_count': metadata.get('visit_count', 1),
                # v2.0 新增字段
                'node_name': metadata.get('node_name'),
                'navigation_instruction': metadata.get('navigation_instruction'),
                'pixel_target': metadata.get('pixel_target'),
                'created_at': metadata.get('created_at'),
                'updated_at': metadata.get('updated_at'),
                'has_front_view_feature': has_front_view,
                # v2.1 前视特征嵌入信息
                'front_view_embedding': front_view_embedding_info
            })

        if NETWORKX_AVAILABLE and self.semantic_graph is not None:
            for u, v, data in self.semantic_graph.edges(data=True):
                edges.append({
                    'source': u,
                    'target': v,
                    'action': data.get('action', []),
                    'description': data.get('description', '')
                })

        return {'nodes': nodes, 'edges': edges}

    def save_to_disk(self, save_path: str):
        """保存语义图到磁盘"""
        try:
            os.makedirs(save_path, exist_ok=True)

            # 保存元数据
            metadata_path = os.path.join(save_path, 'semantic_metadata.json')
            export_data = {
                'node_metadata': {str(k): v for k, v in self.node_metadata.items()},
                'label_index': self.label_index,
                'description_index': {str(k): v for k, v in self.description_index.items()}
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            # 保存图结构
            if NETWORKX_AVAILABLE and self.semantic_graph is not None:
                graph_path = os.path.join(save_path, 'semantic_graph.json')
                graph_data = nx.node_link_data(self.semantic_graph)
                with open(graph_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)

            logger.info(f"[GraphRAG] 语义图已保存到 {save_path}")

        except Exception as e:
            logger.error(f"保存语义图失败: {e}")

    def load_from_disk(self, save_path: str):
        """从磁盘加载语义图"""
        try:
            metadata_path = os.path.join(save_path, 'semantic_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    export_data = json.load(f)
                self.node_metadata = {int(k): v for k, v in export_data.get('node_metadata', {}).items()}
                self.label_index = export_data.get('label_index', {})
                self.description_index = {int(k): v for k, v in export_data.get('description_index', {}).items()}

            graph_path = os.path.join(save_path, 'semantic_graph.json')
            if NETWORKX_AVAILABLE and os.path.exists(graph_path):
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                self.semantic_graph = nx.node_link_graph(graph_data)

            logger.info(f"[GraphRAG] 语义图已从 {save_path} 加载")

        except Exception as e:
            logger.warning(f"加载语义图失败: {e}")

    # ========================================================================
    # v3.0 新增: 时序上下文和场景记忆
    # ========================================================================

    def add_temporal_context(self, node_id: int, prev_node_id: int = None,
                             next_node_id: int = None, transition_type: str = None):
        """
        [v3.0] 添加时序上下文信息

        记录节点之间的时序关系，支持序列推理

        Args:
            node_id: 当前节点ID
            prev_node_id: 前一个节点ID
            next_node_id: 后一个节点ID
            transition_type: 转移类型 ('forward', 'turn_left', 'turn_right', etc.)
        """
        if node_id not in self.node_metadata:
            return

        if 'temporal_context' not in self.node_metadata[node_id]:
            self.node_metadata[node_id]['temporal_context'] = {
                'predecessors': [],
                'successors': [],
                'transitions': []
            }

        context = self.node_metadata[node_id]['temporal_context']

        if prev_node_id is not None and prev_node_id not in context['predecessors']:
            context['predecessors'].append(prev_node_id)

        if next_node_id is not None and next_node_id not in context['successors']:
            context['successors'].append(next_node_id)

        if transition_type:
            context['transitions'].append({
                'from': prev_node_id,
                'to': next_node_id,
                'type': transition_type,
                'timestamp': time.time()
            })

    def get_scene_sequence(self, start_node: int, max_length: int = 10) -> List[Dict]:
        """
        [v3.0] 获取场景序列

        从起始节点开始，返回一系列连续的场景描述

        Args:
            start_node: 起始节点ID
            max_length: 最大序列长度

        Returns:
            场景序列列表
        """
        sequence = []
        current_node = start_node

        while len(sequence) < max_length and current_node is not None:
            if current_node in self.node_metadata:
                metadata = self.node_metadata[current_node]
                sequence.append({
                    'node_id': current_node,
                    'description': metadata.get('scene_description', ''),
                    'labels': metadata.get('semantic_labels', []),
                    'timestamp': metadata.get('timestamp', 0)
                })

                # 找下一个节点
                context = metadata.get('temporal_context', {})
                successors = context.get('successors', [])
                current_node = successors[0] if successors else None
            else:
                break

        return sequence

    def find_similar_scenes(self, scene_description: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        [v3.0] 查找相似场景

        基于场景描述找到相似的节点

        Args:
            scene_description: 场景描述文本
            k: 返回top-k结果

        Returns:
            [(node_id, similarity_score), ...]
        """
        if not scene_description:
            return []

        results = []
        query_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', scene_description.lower()))

        for node_id, desc in self.description_index.items():
            if not desc:
                continue

            desc_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', desc.lower()))

            # 计算词汇重叠度
            if query_words and desc_words:
                intersection = query_words & desc_words
                union = query_words | desc_words
                jaccard = len(intersection) / len(union) if union else 0.0
                results.append((node_id, jaccard))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_navigation_context(self, current_node: int, target_description: str) -> Dict:
        """
        [v3.0] 获取导航上下文

        综合当前位置和目标描述，提供导航建议

        Args:
            current_node: 当前节点ID
            target_description: 目标位置描述

        Returns:
            导航上下文字典
        """
        context = {
            'current_node': current_node,
            'current_scene': None,
            'target_candidates': [],
            'suggested_path': None
        }

        # 获取当前场景信息
        if current_node in self.node_metadata:
            context['current_scene'] = {
                'description': self.node_metadata[current_node].get('scene_description', ''),
                'labels': self.node_metadata[current_node].get('semantic_labels', [])
            }

        # 搜索目标位置
        target_results = self.semantic_search(target_description, k=5)
        context['target_candidates'] = [
            {
                'node_id': node_id,
                'score': score,
                'labels': self.node_metadata.get(node_id, {}).get('semantic_labels', [])
            }
            for node_id, score, _ in target_results
        ]

        # 尝试找到路径
        if target_results and NETWORKX_AVAILABLE and self.semantic_graph is not None:
            best_target = target_results[0][0]
            try:
                path = nx.shortest_path(self.semantic_graph, current_node, best_target)
                context['suggested_path'] = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        return context

    def get_statistics(self) -> Dict:
        """
        [v3.0] 获取语义图统计信息

        Returns:
            统计信息字典
        """
        total_nodes = len(self.node_metadata)
        total_labels = sum(len(m.get('semantic_labels', [])) for m in self.node_metadata.values())
        unique_labels = len(self.label_index)
        nodes_with_context = sum(
            1 for m in self.node_metadata.values()
            if 'temporal_context' in m
        )

        return {
            'total_semantic_nodes': total_nodes,
            'total_labels': total_labels,
            'unique_labels': unique_labels,
            'nodes_with_temporal_context': nodes_with_context,
            'normalized_label_index_size': len(self.normalized_label_index),
            'faiss_index_size': len(self.faiss_node_ids) if FAISS_AVAILABLE else 0,
            'graph_edges': self.semantic_graph.number_of_edges() if NETWORKX_AVAILABLE and self.semantic_graph else 0
        }

    def clear(self):
        """
        完全清空语义图 v2.1
        """
        # 清空节点元数据
        self.node_metadata.clear()

        # 清空标签索引
        self.label_index.clear()
        self.normalized_label_index.clear()

        # 清空描述索引
        self.description_index.clear()

        # 清空特征索引
        self.visual_features.clear()
        self.front_view_features.clear()

        # 重建FAISS索引
        self.faiss_node_ids.clear()
        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)

        # 清空NetworkX图
        if NETWORKX_AVAILABLE and self.semantic_graph is not None:
            self.semantic_graph.clear()

        logger.info("[SemanticGraph] 语义图已完全清空")
