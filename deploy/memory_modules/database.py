#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆导航系统 - PostgreSQL数据库模块

使用PostgreSQL存储拓扑图的节点和边信息。
支持增删改查操作，以及numpy数组的序列化存储。
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,  # PostgreSQL 14 默认端口
    'database': 'topology_db',
    'user': 'memorynav',
    'password': 'memorynav123'
}


class TopologyDatabase:
    """
    拓扑图数据库管理类

    使用PostgreSQL存储拓扑图数据，包括：
    - nodes: 拓扑图节点
    - edges: 拓扑图边
    - metadata: 元数据信息
    """

    def __init__(self, db_config: Dict = None):
        """
        初始化数据库连接

        Args:
            db_config: 数据库配置字典
        """
        self.db_config = db_config or DB_CONFIG
        self._init_database()
        logger.info(f"拓扑图数据库已初始化: PostgreSQL {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")

    def _get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(**self.db_config)

    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建节点表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INTEGER PRIMARY KEY,
                node_name TEXT,
                scene_description TEXT,
                semantic_labels JSONB DEFAULT '[]',
                navigation_instruction TEXT,
                pixel_target JSONB,
                pixel_target_history JSONB DEFAULT '[]',
                visual_feature BYTEA,
                front_view_feature BYTEA,
                front_view_image_path TEXT,
                surround_images JSONB DEFAULT '{}',
                timestamp DOUBLE PRECISION,
                created_at DOUBLE PRECISION,
                updated_at DOUBLE PRECISION,
                visit_count INTEGER DEFAULT 1,
                is_landmark BOOLEAN DEFAULT FALSE,
                is_keyframe BOOLEAN DEFAULT FALSE,
                source_timestamps JSONB DEFAULT '[]',
                entry_actions JSONB DEFAULT '[]',
                exit_actions JSONB DEFAULT '[]'
            )
        ''')

        # 创建边表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id SERIAL PRIMARY KEY,
                source_node_id INTEGER NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                target_node_id INTEGER NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                action JSONB DEFAULT '[]',
                weight DOUBLE PRECISION DEFAULT 1.0,
                description TEXT DEFAULT '',
                created_at DOUBLE PRECISION,
                UNIQUE(source_node_id, target_node_id)
            )
        ''')

        # 创建元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value JSONB,
                updated_at DOUBLE PRECISION
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(node_name)')

        conn.commit()
        conn.close()

    # ================== 节点操作 ==================

    def add_node(self, node_data: Dict[str, Any]) -> int:
        """
        添加节点

        Args:
            node_data: 节点数据字典

        Returns:
            节点ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().timestamp()

        # 序列化numpy数组为bytes
        visual_feature = node_data.get('visual_feature')
        if visual_feature is not None:
            if isinstance(visual_feature, np.ndarray):
                visual_feature = visual_feature.tobytes()

        front_view_feature = node_data.get('front_view_feature')
        if front_view_feature is not None:
            if isinstance(front_view_feature, np.ndarray):
                front_view_feature = front_view_feature.tobytes()

        cursor.execute('''
            INSERT INTO nodes (
                node_id, node_name, scene_description, semantic_labels,
                navigation_instruction, pixel_target, pixel_target_history,
                visual_feature, front_view_feature, front_view_image_path,
                surround_images, timestamp, created_at, updated_at,
                visit_count, is_landmark, is_keyframe, source_timestamps,
                entry_actions, exit_actions
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE SET
                node_name = EXCLUDED.node_name,
                scene_description = EXCLUDED.scene_description,
                semantic_labels = EXCLUDED.semantic_labels,
                navigation_instruction = EXCLUDED.navigation_instruction,
                pixel_target = EXCLUDED.pixel_target,
                pixel_target_history = EXCLUDED.pixel_target_history,
                visual_feature = EXCLUDED.visual_feature,
                front_view_feature = EXCLUDED.front_view_feature,
                front_view_image_path = EXCLUDED.front_view_image_path,
                surround_images = EXCLUDED.surround_images,
                updated_at = EXCLUDED.updated_at,
                visit_count = EXCLUDED.visit_count,
                is_landmark = EXCLUDED.is_landmark,
                is_keyframe = EXCLUDED.is_keyframe,
                source_timestamps = EXCLUDED.source_timestamps,
                entry_actions = EXCLUDED.entry_actions,
                exit_actions = EXCLUDED.exit_actions
        ''', (
            node_data.get('node_id'),
            node_data.get('node_name'),
            node_data.get('scene_description'),
            json.dumps(node_data.get('semantic_labels', []), ensure_ascii=False),
            node_data.get('navigation_instruction'),
            json.dumps(node_data.get('pixel_target')),
            json.dumps(node_data.get('pixel_target_history', []), ensure_ascii=False),
            visual_feature,
            front_view_feature,
            node_data.get('front_view_image_path'),
            json.dumps(node_data.get('surround_images', {}), ensure_ascii=False),
            node_data.get('timestamp', now),
            node_data.get('created_at', now),
            now,
            node_data.get('visit_count', 1),
            node_data.get('is_landmark', False),
            node_data.get('is_keyframe', False),
            json.dumps(node_data.get('source_timestamps', []), ensure_ascii=False),
            json.dumps(node_data.get('entry_actions', []), ensure_ascii=False),
            json.dumps(node_data.get('exit_actions', []), ensure_ascii=False)
        ))

        node_id = node_data.get('node_id')
        conn.commit()
        conn.close()

        logger.debug(f"添加节点: {node_id}")
        return node_id

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        获取单个节点

        Args:
            node_id: 节点ID

        Returns:
            节点数据字典，不存在返回None
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT * FROM nodes WHERE node_id = %s', (node_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_node_dict(dict(row))

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """获取所有节点"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT * FROM nodes ORDER BY node_id')
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_node_dict(dict(row)) for row in rows]

    def update_node(self, node_id: int, updates: Dict[str, Any]) -> bool:
        """
        更新节点

        Args:
            node_id: 节点ID
            updates: 要更新的字段

        Returns:
            是否成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 构建更新语句
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ['visual_feature', 'front_view_feature']:
                if value is not None and isinstance(value, np.ndarray):
                    value = value.tobytes()
                set_clauses.append(f"{key} = %s")
                values.append(value)
            elif key in ['semantic_labels', 'pixel_target', 'pixel_target_history',
                         'surround_images', 'source_timestamps', 'entry_actions', 'exit_actions']:
                set_clauses.append(f"{key} = %s")
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                set_clauses.append(f"{key} = %s")
                values.append(value)

        set_clauses.append("updated_at = %s")
        values.append(datetime.now().timestamp())

        values.append(node_id)

        sql = f"UPDATE nodes SET {', '.join(set_clauses)} WHERE node_id = %s"
        cursor.execute(sql, values)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logger.debug(f"更新节点: {node_id}")
        return success

    def delete_node(self, node_id: int) -> bool:
        """
        删除节点（边会自动级联删除）

        Args:
            node_id: 节点ID

        Returns:
            是否成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM nodes WHERE node_id = %s', (node_id,))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logger.info(f"删除节点: {node_id}")
        return success

    def _row_to_node_dict(self, row: Dict) -> Dict[str, Any]:
        """将数据库行转换为节点字典"""
        node = dict(row)

        # 反序列化numpy数组 (512维float32向量)
        if node.get('visual_feature'):
            node['visual_feature'] = np.frombuffer(bytes(node['visual_feature']), dtype=np.float32)
        if node.get('front_view_feature'):
            node['front_view_feature'] = np.frombuffer(bytes(node['front_view_feature']), dtype=np.float32)

        return node

    # ================== 边操作 ==================

    def add_edge(self, source_id: int, target_id: int, action: List[int] = None,
                 weight: float = 1.0, description: str = "") -> int:
        """
        添加边

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            action: 动作列表
            weight: 边权重
            description: 描述

        Returns:
            边ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().timestamp()

        cursor.execute('''
            INSERT INTO edges (source_node_id, target_node_id, action, weight, description, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_node_id, target_node_id) DO UPDATE SET
                action = EXCLUDED.action,
                weight = EXCLUDED.weight,
                description = EXCLUDED.description
            RETURNING id
        ''', (
            source_id,
            target_id,
            json.dumps(action or []),
            weight,
            description,
            now
        ))

        edge_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()

        logger.debug(f"添加边: {source_id} -> {target_id}")
        return edge_id

    def get_edges(self, source_id: int = None, target_id: int = None) -> List[Dict[str, Any]]:
        """
        获取边

        Args:
            source_id: 源节点ID（可选）
            target_id: 目标节点ID（可选）

        Returns:
            边列表
        """
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if source_id is not None and target_id is not None:
            cursor.execute(
                'SELECT * FROM edges WHERE source_node_id = %s AND target_node_id = %s',
                (source_id, target_id)
            )
        elif source_id is not None:
            cursor.execute('SELECT * FROM edges WHERE source_node_id = %s', (source_id,))
        elif target_id is not None:
            cursor.execute('SELECT * FROM edges WHERE target_node_id = %s', (target_id,))
        else:
            cursor.execute('SELECT * FROM edges')

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_all_edges(self) -> List[Dict[str, Any]]:
        """获取所有边"""
        return self.get_edges()

    def update_edge(self, source_id: int, target_id: int, updates: Dict[str, Any]) -> bool:
        """更新边"""
        conn = self._get_connection()
        cursor = conn.cursor()

        set_clauses = []
        values = []

        for key, value in updates.items():
            if key == 'action':
                set_clauses.append(f"{key} = %s")
                values.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = %s")
                values.append(value)

        values.extend([source_id, target_id])

        sql = f"UPDATE edges SET {', '.join(set_clauses)} WHERE source_node_id = %s AND target_node_id = %s"
        cursor.execute(sql, values)

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def delete_edge(self, source_id: int, target_id: int) -> bool:
        """删除边"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'DELETE FROM edges WHERE source_node_id = %s AND target_node_id = %s',
            (source_id, target_id)
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if success:
            logger.debug(f"删除边: {source_id} -> {target_id}")
        return success

    # ================== 图操作 ==================

    def get_graph_data(self) -> Dict[str, Any]:
        """
        获取完整的图数据（用于可视化）

        Returns:
            包含nodes和edges的字典
        """
        nodes = self.get_all_nodes()
        edges = self.get_all_edges()

        # 转换为前端需要的格式
        formatted_nodes = []
        for node in nodes:
            formatted_nodes.append({
                'id': node['node_id'],
                'label': f"N{node['node_id']}",
                'node_name': node.get('node_name'),
                'scene_description': node.get('scene_description'),
                'semantic_labels': node.get('semantic_labels', []),
                'visit_count': node.get('visit_count', 1),
                'is_keyframe': node.get('is_keyframe', False),
                'navigation_instruction': node.get('navigation_instruction'),
                'pixel_target': node.get('pixel_target'),
                'created_at': node.get('created_at'),
                'updated_at': node.get('updated_at'),
                'has_front_view_feature': node.get('front_view_feature') is not None,
                'source_timestamps': node.get('source_timestamps', []),
                'front_view_embedding': self._serialize_embedding(node.get('front_view_feature'))
            })

        formatted_edges = []
        for edge in edges:
            formatted_edges.append({
                'source': edge['source_node_id'],
                'target': edge['target_node_id'],
                'action': edge.get('action', []),
                'weight': edge.get('weight', 1.0),
                'description': edge.get('description', '')
            })

        return {
            'nodes': formatted_nodes,
            'edges': formatted_edges
        }

    def _serialize_embedding(self, feature: Optional[np.ndarray]) -> Optional[List[float]]:
        """序列化嵌入向量"""
        if feature is None:
            return None
        return [float(x) for x in feature]

    def clear_all(self):
        """清空所有数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM edges')
        cursor.execute('DELETE FROM nodes')
        cursor.execute('DELETE FROM metadata')

        conn.commit()
        conn.close()

        logger.info("数据库已清空")

    def get_node_count(self) -> int:
        """获取节点数量"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM nodes')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_edge_count(self) -> int:
        """获取边数量"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM edges')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # ================== 元数据操作 ==================

    def set_metadata(self, key: str, value: Any):
        """设置元数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO metadata (key, value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = EXCLUDED.updated_at
        ''', (key, json.dumps(value, ensure_ascii=False), datetime.now().timestamp()))

        conn.commit()
        conn.close()

    def get_metadata(self, key: str) -> Optional[Any]:
        """获取元数据"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT value FROM metadata WHERE key = %s', (key,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return row[0]

    # ================== 导入导出 ==================

    def import_from_json(self, graph_json: Dict, metadata_json: Dict = None):
        """
        从JSON数据导入

        Args:
            graph_json: semantic_graph.json的内容
            metadata_json: semantic_metadata.json的内容
        """
        # 清空现有数据
        self.clear_all()

        node_metadata = {}
        if metadata_json:
            node_metadata = metadata_json.get('node_metadata', {})

        # 导入节点
        for node in graph_json.get('nodes', []):
            node_id = node.get('id')
            meta = node_metadata.get(str(node_id), {})

            node_data = {
                'node_id': node_id,
                'node_name': meta.get('node_name'),
                'scene_description': meta.get('scene_description') or node.get('description'),
                'semantic_labels': meta.get('semantic_labels') or node.get('labels', []),
                'navigation_instruction': meta.get('navigation_instruction'),
                'pixel_target': meta.get('pixel_target'),
                'pixel_target_history': meta.get('pixel_target_history', []),
                'timestamp': meta.get('timestamp'),
                'created_at': meta.get('created_at') or meta.get('timestamp'),
                'visit_count': meta.get('visit_count', 1),
                'is_keyframe': bool(meta.get('scene_description') or node.get('description')),
                'source_timestamps': meta.get('source_timestamps', [])
            }

            self.add_node(node_data)

        # 导入边
        for link in graph_json.get('links', []):
            self.add_edge(
                source_id=link.get('source'),
                target_id=link.get('target'),
                action=link.get('action', []),
                weight=link.get('weight', 1.0),
                description=link.get('description', '')
            )

        logger.info(f"导入完成: {self.get_node_count()} 节点, {self.get_edge_count()} 边")

    def export_to_json(self) -> Tuple[Dict, Dict]:
        """
        导出为JSON格式

        Returns:
            (semantic_graph.json内容, semantic_metadata.json内容)
        """
        nodes = self.get_all_nodes()
        edges = self.get_all_edges()

        # 构建semantic_graph.json
        graph_json = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": []
        }

        # 构建semantic_metadata.json
        metadata_json = {
            "node_metadata": {},
            "label_index": {},
            "description_index": {}
        }

        for node in nodes:
            # graph节点
            graph_json["nodes"].append({
                "id": node['node_id'],
                "description": node.get('scene_description', ''),
                "labels": node.get('semantic_labels', [])
            })

            # metadata
            node_id_str = str(node['node_id'])
            metadata_json["node_metadata"][node_id_str] = {
                "scene_description": node.get('scene_description'),
                "semantic_labels": node.get('semantic_labels', []),
                "timestamp": node.get('timestamp'),
                "visit_count": node.get('visit_count', 1),
                "node_name": node.get('node_name'),
                "navigation_instruction": node.get('navigation_instruction'),
                "pixel_target": node.get('pixel_target'),
                "created_at": node.get('created_at'),
                "updated_at": node.get('updated_at'),
                "pixel_target_history": node.get('pixel_target_history', []),
                "has_front_view_feature": node.get('front_view_feature') is not None,
                "source_timestamps": node.get('source_timestamps', [])
            }

            # 标签索引
            for label in node.get('semantic_labels', []):
                if label not in metadata_json["label_index"]:
                    metadata_json["label_index"][label] = []
                if node['node_id'] not in metadata_json["label_index"][label]:
                    metadata_json["label_index"][label].append(node['node_id'])

            # 描述索引
            if node.get('scene_description'):
                metadata_json["description_index"][node_id_str] = node['scene_description']

        # 边
        for edge in edges:
            graph_json["links"].append({
                "source": edge['source_node_id'],
                "target": edge['target_node_id'],
                "action": edge.get('action', []),
                "weight": edge.get('weight', 1.0),
                "description": edge.get('description', '')
            })

        return graph_json, metadata_json


    def close(self):
        """关闭数据库连接（兼容性方法）"""
        # 连接是按需创建的，不需要显式关闭
        pass


# 全局数据库实例
_db_instance: Optional[TopologyDatabase] = None


def get_database() -> TopologyDatabase:
    """获取全局数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TopologyDatabase()
    return _db_instance
