# InternNav 视觉记忆导航系统

## 摘要

本文档提出了一套完整的优化方案，为 InternNav 实现**视觉记忆导航系统**。该系统解决了当前 `ws_proxy.py` 导航服务的两个关键限制：

1. **缺乏路线记忆**：已访问位置在后续访问时被视为完全陌生
2. **返回导航**：无法支持"返回起点"类型的指令

方案采用**视觉位置识别 (VPR)**、**基于 FAISS 的语义建图**和**图检索增强生成 (GraphRAG)** 技术，构建持久化视觉记忆系统，实现智能路径规划和重定位。

---

## 目录

1. [系统架构概述](#1-系统架构概述)
2. [核心组件](#2-核心组件)
3. [技术实现细节](#3-技术实现细节)
4. [数据结构](#4-数据结构)
5. [API 设计](#5-api-设计)
6. [与现有 InternNav 的集成](#6-与现有-internnav-的集成)
7. [性能优化](#7-性能优化)
8. [参考实现](#8-参考实现)
9. [实施路线图](#9-实施路线图)

---

## 1. 系统架构概述

### 1.1 高层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          视觉记忆导航系统                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   输入      │    │   特征      │    │   记忆      │    │   图查询    │   │
│  │  处理层    │───▶│  提取层    │───▶│  存储层    │───▶│    层      │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                  │                  │                  │            │
│        ▼                  ▼                  ▼                  ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ camera_1~4  │    │  LongCLIP   │    │   FAISS     │    │  GraphRAG   │   │
│  │ front_1     │    │  DINOv2     │    │  节点数据库  │    │  路径规划   │   │
│  │ Qwen3-VL    │    │  VLAD       │    │  边数据库    │    │  检索查询   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
输入图像 (camera_1~4)
        │
        ▼
┌───────────────────┐
│ 视觉特征提取       │ ─────────────────────────────────────┐
│ (LongCLIP/DINOv2) │                                       │
└───────────────────┘                                       │
        │                                                   │
        ▼                                                   │
┌───────────────────┐     ┌───────────────────┐            │
│ 场景描述生成       │     │ 语义标签提取       │            │
│ (Qwen3-VL)        │     │ (Qwen3-VL)        │            │
└───────────────────┘     └───────────────────┘            │
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    记忆节点创建                               │
│  {                                                            │
│    node_id: uuid,                                             │
│    timestamp: epoch,                                          │
│    visual_features: [cam1_feat, cam2_feat, cam3_feat, cam4_feat],
│    scene_description: "文本描述",                              │
│    semantic_labels: ["走廊", "门", "窗户"],                    │
│    pixel_target: [y, x],                                      │
│    task_instruction: "当前任务"                                │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│   FAISS 索引      │     │   图数据库         │
│   (节点存储)      │◀───▶│   (边存储)         │
└───────────────────┘     └───────────────────┘
```

---

## 2. 核心组件

### 2.1 视觉特征提取器 (LongCLIPFeatureExtractor)

**目的**：从多相机输入中提取鲁棒的视觉特征，用于位置识别。

**技术栈**：
- **核心模型**：LongCLIP-B（已集成在 InternNav 中）
- **特征维度**：768 维
- **预处理**：LongCLIP 内置 image processor

**实现文件**：`deploy/ws_proxy_with_memory.py`

```python
class LongCLIPFeatureExtractor:
    """
    基于 LongCLIP 的视觉特征提取器

    使用 InternNav 已集成的 LongCLIP 模型提取图像特征，
    用于视觉位置识别 (VPR) 和回环检测。
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)

        # 加载 LongCLIP 模型
        self.model, self.preprocess = longclip.load(model_path, device=device)

        # 仅保留视觉编码器，删除文本部分以节省显存
        del self.model.token_embedding
        del self.model.transformer
        del self.model.positional_embedding
        del self.model.ln_final

        # 设置为评估模式并冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def extract_feature(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        提取单张图像的视觉特征

        参数:
            rgb_image: RGB图像 [H, W, 3]

        返回:
            feature: 归一化特征向量 [768]
        """
        # 转换为 PIL Image 并预处理
        pil_image = Image.fromarray(rgb_image)
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # 使用视觉编码器提取特征
        with torch.no_grad():
            feature = self._encode_image(image_tensor)

        # 归一化
        feature = feature.cpu().numpy().flatten().astype('float32')
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature

    def _encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """使用 LongCLIP 视觉编码器提取特征"""
        visual = self.model.visual

        x = image_tensor.type(visual.conv1.weight.dtype)
        x = visual.conv1(x)                      # Patch embedding
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # 添加 CLS token 和位置编码
        x = torch.cat([visual.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device), x], dim=1)
        x = x + visual.positional_embedding
        x = visual.ln_pre(x)

        # Transformer 编码
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

        # 提取 CLS token 并投影
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj

        return x
```

**配置参数** (`MemoryNavigationConfig`):
```python
longclip_model_path: str = "checkpoints/longclip-B.pt"  # LongCLIP-B 模型路径
feature_extractor_device: str = "cuda:0"                 # 推理设备
feature_dim: int = 768                                   # 特征向量维度
```

### 2.2 场景描述生成器

**目的**：使用 VLM 生成场景的文本描述，用于语义理解。

**技术栈**：
- **模型**：Qwen3-VL（本地部署）
- **输出**：自然语言场景描述 + 语义标签

```python
class SceneDescriptionGenerator:
    """
    使用 Qwen3-VL 生成场景描述和语义标签。
    """
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate_description(self, images: Dict[str, np.ndarray]) -> str:
        """
        从全景视图生成综合场景描述。
        """
        # 将图像组合成网格以获取上下文
        combined_prompt = """基于4个相机视图（前右、前左、后左、后右），
        用2-3句话描述当前位置。重点关注：
        1. 关键地标和物体
        2. 环境类型（走廊、房间、室外等）
        3. 可用于后续识别此位置的显著特征。
        """

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(images.get("camera_1", np.zeros((480,640,3), dtype=np.uint8)))},
                {"type": "image", "image": Image.fromarray(images.get("camera_2", np.zeros((480,640,3), dtype=np.uint8)))},
                {"type": "text", "text": combined_prompt}
            ]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[Image.fromarray(img) for img in images.values()], return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(**inputs.to(self.model.device), max_new_tokens=256)

        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def extract_semantic_labels(self, images: Dict[str, np.ndarray]) -> List[str]:
        """
        从图像中提取语义标签（物体、场景类型）。
        """
        label_prompt = """列出这些图像中可见的关键物体和场景元素。
        仅输出逗号分隔的名词列表。例如：门、窗户、走廊、电梯、标识牌
        """

        # 类似上述处理，但使用标签提取提示
        # 返回: ["走廊", "玻璃门", "电梯", "盆栽"]
        pass
```

### 2.3 记忆节点管理器

**目的**：使用 FAISS 创建、存储和检索记忆节点。

**技术栈**：
- **向量数据库**：FAISS（支持 GPU 加速）
- **元数据**：JSON/SQLite 存储节点属性
- **索引类型**：IVF-PQ 用于大规模高效检索

```python
import faiss
import numpy as np
import json
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import sqlite3

@dataclass
class MemoryNode:
    """
    表示关键帧位置的单个记忆节点。
    """
    node_id: str
    timestamp: float
    visual_features: Dict[str, List[float]]  # 相机名称 -> 特征向量
    global_descriptor: List[float]
    scene_description: str
    semantic_labels: List[str]
    pixel_target: Optional[List[float]]
    task_instruction: str
    position_estimate: Optional[List[float]] = None  # [x, y, yaw] 如果可用

class MemoryNodeManager:
    """
    使用 FAISS 管理记忆节点，实现高效相似度搜索。
    """
    def __init__(self, feature_dim: int = 768, index_path: str = "./memory_index"):
        self.feature_dim = feature_dim
        self.index_path = index_path

        # 初始化 FAISS 索引 (IVF + PQ 用于可扩展性)
        self.nlist = 100  # 聚类数量
        self.m = 8  # 子量化器数量

        # 创建索引
        quantizer = faiss.IndexFlatL2(feature_dim)
        self.index = faiss.IndexIVFPQ(quantizer, feature_dim, self.nlist, self.m, 8)
        self.is_trained = False

        # 节点元数据存储
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_id_to_idx: Dict[str, int] = {}
        self.idx_to_node_id: Dict[int, str] = {}

        # SQLite 用于持久化元数据
        self._init_database()

    def _init_database(self):
        """初始化 SQLite 数据库用于元数据持久化。"""
        self.conn = sqlite3.connect(f"{self.index_path}/nodes.db")
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                timestamp REAL,
                scene_description TEXT,
                semantic_labels TEXT,
                task_instruction TEXT,
                pixel_target TEXT,
                position_estimate TEXT
            )
        ''')
        self.conn.commit()

    def add_node(self, node: MemoryNode) -> str:
        """
        向索引添加新的记忆节点。
        """
        # 将全局描述符转换为 numpy
        descriptor = np.array(node.global_descriptor, dtype=np.float32).reshape(1, -1)

        # 如果未训练则训练索引（需要至少 nlist 个向量）
        if not self.is_trained:
            if len(self.nodes) >= self.nlist:
                all_descriptors = np.array([
                    n.global_descriptor for n in self.nodes.values()
                ], dtype=np.float32)
                self.index.train(all_descriptors)
                self.index.add(all_descriptors)
                for i, nid in enumerate(self.nodes.keys()):
                    self.node_id_to_idx[nid] = i
                    self.idx_to_node_id[i] = nid
                self.is_trained = True

        # 添加到索引
        if self.is_trained:
            idx = self.index.ntotal
            self.index.add(descriptor)
            self.node_id_to_idx[node.node_id] = idx
            self.idx_to_node_id[idx] = node.node_id

        # 存储节点
        self.nodes[node.node_id] = node

        # 持久化到数据库
        self._save_node_to_db(node)

        return node.node_id

    def search_similar(self, query_descriptor: np.ndarray, k: int = 5) -> List[tuple]:
        """
        搜索 k 个最相似的节点。

        返回:
            (node_id, distance) 元组列表
        """
        if not self.is_trained or self.index.ntotal == 0:
            # 小数据库回退到暴力搜索
            return self._brute_force_search(query_descriptor, k)

        query = query_descriptor.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.idx_to_node_id:
                node_id = self.idx_to_node_id[idx]
                results.append((node_id, float(dist)))

        return results

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """通过 ID 检索节点。"""
        return self.nodes.get(node_id)

    def _brute_force_search(self, query: np.ndarray, k: int) -> List[tuple]:
        """小数据集的暴力搜索回退方案。"""
        results = []
        for node_id, node in self.nodes.items():
            dist = np.linalg.norm(query - np.array(node.global_descriptor))
            results.append((node_id, dist))
        return sorted(results, key=lambda x: x[1])[:k]

    def _save_node_to_db(self, node: MemoryNode):
        """将节点元数据持久化到 SQLite。"""
        self.conn.execute('''
            INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.node_id,
            node.timestamp,
            node.scene_description,
            json.dumps(node.semantic_labels),
            node.task_instruction,
            json.dumps(node.pixel_target) if node.pixel_target else None,
            json.dumps(node.position_estimate) if node.position_estimate else None
        ))
        self.conn.commit()

    def save_index(self):
        """将 FAISS 索引保存到磁盘。"""
        if self.is_trained:
            faiss.write_index(self.index, f"{self.index_path}/faiss.index")

        # 保存节点到索引的映射
        with open(f"{self.index_path}/mappings.json", 'w') as f:
            json.dump({
                'node_id_to_idx': self.node_id_to_idx,
                'idx_to_node_id': {str(k): v for k, v in self.idx_to_node_id.items()}
            }, f)

    def load_index(self):
        """从磁盘加载 FAISS 索引。"""
        import os
        index_file = f"{self.index_path}/faiss.index"
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            self.is_trained = True

            with open(f"{self.index_path}/mappings.json", 'r') as f:
                mappings = json.load(f)
                self.node_id_to_idx = mappings['node_id_to_idx']
                self.idx_to_node_id = {int(k): v for k, v in mappings['idx_to_node_id'].items()}
```

### 2.4 图边管理器（GraphRAG 风格）

**目的**：存储和查询节点间的连接关系，用于路径规划。

**技术栈**：
- **图库**：NetworkX（轻量级）或 Neo4j（生产环境）
- **查询**：图遍历 + 语义匹配

```python
import networkx as nx
from typing import List, Tuple, Optional
import json

class GraphEdgeManager:
    """
    管理表示记忆节点间可通行路径的图边。
    实现 GraphRAG 风格的导航知识图谱。
    """
    def __init__(self, index_path: str = "./memory_index"):
        self.graph = nx.DiGraph()
        self.index_path = index_path

    def add_edge(self,
                 from_node_id: str,
                 to_node_id: str,
                 action_taken: List[float],  # [x, y, yaw]
                 traversal_time: float,
                 semantic_description: str = ""):
        """
        在两个节点间添加有向边。

        参数:
            from_node_id: 源节点 ID
            to_node_id: 目标节点 ID
            action_taken: 从源到目标的机器人动作 [x, y, yaw]
            traversal_time: 遍历所需时间
            semantic_description: 转换的自然语言描述
        """
        self.graph.add_edge(
            from_node_id,
            to_node_id,
            action=action_taken,
            time=traversal_time,
            description=semantic_description,
            weight=np.linalg.norm(action_taken[:2])  # 基于距离的权重
        )

    def find_path(self,
                  start_node_id: str,
                  end_node_id: str,
                  method: str = "shortest") -> Optional[List[str]]:
        """
        查找两个节点间的路径。

        参数:
            start_node_id: 起始节点 ID
            end_node_id: 目标节点 ID
            method: "shortest" 使用 Dijkstra，"semantic" 使用基于标签的方法

        返回:
            形成路径的节点 ID 列表，如果没有路径则返回 None
        """
        try:
            if method == "shortest":
                return nx.shortest_path(self.graph, start_node_id, end_node_id, weight='weight')
            else:
                return nx.shortest_path(self.graph, start_node_id, end_node_id)
        except nx.NetworkXNoPath:
            return None

    def get_actions_for_path(self, path: List[str]) -> List[List[float]]:
        """
        获取遍历路径所需的动作序列。

        返回:
            [x, y, yaw] 动作列表
        """
        actions = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                actions.append(edge_data['action'])
        return actions

    def get_neighbors(self, node_id: str) -> List[Tuple[str, dict]]:
        """获取所有相邻节点及边属性。"""
        neighbors = []
        for neighbor in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            neighbors.append((neighbor, edge_data))
        return neighbors

    def semantic_search_path(self,
                             start_node_id: str,
                             target_description: str,
                             node_manager: 'MemoryNodeManager') -> Optional[List[str]]:
        """
        查找到匹配语义描述位置的路径。
        使用 GraphRAG 风格的语义匹配。

        参数:
            start_node_id: 起始节点
            target_description: 目标的自然语言描述
            node_manager: 节点管理器引用，用于语义匹配

        返回:
            到最佳匹配目的地的路径
        """
        # 查找匹配描述的节点
        matching_nodes = []
        for node_id in self.graph.nodes():
            node = node_manager.get_node(node_id)
            if node and self._semantic_match(target_description, node):
                matching_nodes.append(node_id)

        # 查找到任何匹配节点的最短路径
        best_path = None
        best_length = float('inf')

        for target_id in matching_nodes:
            path = self.find_path(start_node_id, target_id)
            if path and len(path) < best_length:
                best_path = path
                best_length = len(path)

        return best_path

    def _semantic_match(self, query: str, node: 'MemoryNode', threshold: float = 0.5) -> bool:
        """
        检查节点是否在语义上匹配查询描述。
        简单关键词匹配 - 可用嵌入增强。
        """
        query_lower = query.lower()

        # 检查语义标签
        for label in node.semantic_labels:
            if label.lower() in query_lower:
                return True

        # 检查场景描述
        if any(word in node.scene_description.lower() for word in query_lower.split()):
            return True

        return False

    def save_graph(self):
        """将图持久化到磁盘。"""
        nx.write_gpickle(self.graph, f"{self.index_path}/graph.gpickle")

    def load_graph(self):
        """从磁盘加载图。"""
        import os
        graph_file = f"{self.index_path}/graph.gpickle"
        if os.path.exists(graph_file):
            self.graph = nx.read_gpickle(graph_file)
```

### 2.5 视觉位置识别模块

**目的**：识别当前位置是否匹配任何已存储的记忆节点。

**技术栈**：
- **方法**：AnyLoc-VLAD-DINOv2（参考：`reference_code/AnyLoc/`）
- **匹配**：FAISS k-NN 搜索 + 重排序

```python
class VisualPlaceRecognizer:
    """
    基于特征匹配的视觉位置识别。
    基于 AnyLoc 方法论。
    """
    def __init__(self,
                 feature_extractor: VisualFeatureExtractor,
                 node_manager: MemoryNodeManager,
                 similarity_threshold: float = 0.7):
        self.feature_extractor = feature_extractor
        self.node_manager = node_manager
        self.similarity_threshold = similarity_threshold

    def recognize(self, images: Dict[str, np.ndarray]) -> Optional[Tuple[str, float]]:
        """
        尝试从记忆中识别当前位置。

        参数:
            images: 相机图像字典

        返回:
            如果识别成功返回 (node_id, confidence)，否则返回 None
        """
        # 提取特征
        features = self.feature_extractor.extract_features(images)
        global_descriptor = self.feature_extractor.compute_global_descriptor(features)

        # 在记忆中搜索
        matches = self.node_manager.search_similar(global_descriptor, k=5)

        if not matches:
            return None

        best_match_id, best_distance = matches[0]

        # 将距离转换为相似度（余弦相似度近似）
        # 对于归一化向量的 L2 距离：sim = 1 - dist^2 / 2
        similarity = 1 - (best_distance ** 2) / 2

        if similarity >= self.similarity_threshold:
            return (best_match_id, similarity)

        return None

    def re_rank_matches(self,
                        query_features: Dict[str, np.ndarray],
                        candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        使用逐相机特征匹配重新排序候选匹配。
        """
        scores = []
        for node_id in candidate_ids:
            node = self.node_manager.get_node(node_id)
            if not node:
                continue

            # 计算逐相机相似度
            cam_scores = []
            for cam_name, query_feat in query_features.items():
                if cam_name in node.visual_features:
                    node_feat = np.array(node.visual_features[cam_name])
                    sim = np.dot(query_feat, node_feat) / (
                        np.linalg.norm(query_feat) * np.linalg.norm(node_feat)
                    )
                    cam_scores.append(sim)

            avg_score = np.mean(cam_scores) if cam_scores else 0
            scores.append((node_id, avg_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

## 3. 技术实现细节

### 3.1 关键帧检测逻辑

当 InternNav 模型生成非 None 像素目标时，该帧被视为**记忆关键帧**：

```python
def should_create_keyframe(dual_sys_output) -> bool:
    """
    确定当前帧是否应保存为记忆关键帧。

    条件:
    1. 模型输出非 None 像素目标（表示重要决策点）
    2. 与上一个关键帧有足够的距离/时间间隔（避免冗余）
    """
    if dual_sys_output.output_pixel is not None:
        return True
    return False
```

### 3.2 多相机特征聚合

```python
def aggregate_panoramic_features(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    将 4 个环视相机的特征聚合为单一全景描述符。

    注意: 仅使用 camera_1~4 四个环视相机，不包含 front_1 前置相机。

    相机布局:
    - camera_1: 前右 (+37.5 度)
    - camera_2: 前左 (-37.5 度)
    - camera_3: 后左 (-142.5 度)
    - camera_4: 后右 (+142.5 度)

    聚合策略:
    1. 等权重融合（每个相机 0.25）
    2. L2 归一化
    """
    # 四个环视相机等权重
    weights = {
        'camera_1': 0.25,
        'camera_2': 0.25,
        'camera_3': 0.25,
        'camera_4': 0.25
    }

    weighted_features = []
    total_weight = 0.0
    for cam_name in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
        if cam_name in features and features[cam_name] is not None:
            weighted_features.append(features[cam_name] * weights[cam_name])
            total_weight += weights[cam_name]

    if not weighted_features:
        return None  # 无可用特征

    # 加权平均并归一化
    aggregated = np.sum(weighted_features, axis=0) / total_weight
    return aggregated / np.linalg.norm(aggregated)
```

### 3.3 支持返回的路径规划

```python
class PathPlanner:
    """
    使用记忆图规划导航路径。
    支持返回起点指令。
    """
    def __init__(self,
                 node_manager: MemoryNodeManager,
                 edge_manager: GraphEdgeManager,
                 recognizer: VisualPlaceRecognizer):
        self.node_manager = node_manager
        self.edge_manager = edge_manager
        self.recognizer = recognizer
        self.start_node_id: Optional[str] = None
        self.current_node_id: Optional[str] = None

    def set_start_position(self, node_id: str):
        """标记返回导航的起始位置。"""
        self.start_node_id = node_id
        self.current_node_id = node_id

    def update_position(self, images: Dict[str, np.ndarray]) -> Optional[str]:
        """
        基于视觉识别更新当前位置。
        """
        result = self.recognizer.recognize(images)
        if result:
            self.current_node_id = result[0]
            return self.current_node_id
        return None

    def plan_return_to_start(self) -> Optional[List[List[float]]]:
        """
        规划返回起始位置的动作。

        返回:
            [x, y, yaw] 动作列表，如果没有找到路径则返回 None
        """
        if not self.start_node_id or not self.current_node_id:
            return None

        if self.start_node_id == self.current_node_id:
            return [[0.0, 0.0, 0.0]]  # 已在起点

        path = self.edge_manager.find_path(self.current_node_id, self.start_node_id)
        if not path:
            return None

        return self.edge_manager.get_actions_for_path(path)

    def plan_to_description(self, target_description: str) -> Optional[List[List[float]]]:
        """
        规划到匹配语义描述位置的路径。
        """
        if not self.current_node_id:
            return None

        path = self.edge_manager.semantic_search_path(
            self.current_node_id,
            target_description,
            self.node_manager
        )

        if not path:
            return None

        return self.edge_manager.get_actions_for_path(path)
```

---

## 4. 数据结构

### 4.1 记忆节点模式

```json
{
  "node_id": "uuid-v4-string",
  "timestamp": 1704067200.0,
  "visual_features": {
    "camera_1": [0.123, -0.456, ...],
    "camera_2": [0.789, -0.012, ...],
    "camera_3": [0.345, -0.678, ...],
    "camera_4": [0.901, -0.234, ...]
  },
  "global_descriptor": [0.111, 0.222, ...],
  "scene_description": "明亮的走廊，左侧有玻璃门，前方是电梯。",
  "semantic_labels": ["走廊", "玻璃门", "电梯", "盆栽"],
  "pixel_target": [240.5, 320.0],
  "task_instruction": "导航到电梯",
  "position_estimate": [2.5, 1.3, 0.45]
}
```

### 4.2 边模式

```json
{
  "from_node": "node-uuid-1",
  "to_node": "node-uuid-2",
  "action": [0.5, 0.0, 0.13],
  "traversal_time": 2.3,
  "description": "沿走廊向前移动",
  "weight": 0.5
}
```

### 4.3 会话状态模式

```json
{
  "session_id": "session-uuid",
  "start_node_id": "node-uuid-start",
  "current_node_id": "node-uuid-current",
  "path_history": ["node-1", "node-2", "node-3"],
  "task_history": [
    {"task": "去电梯", "completed": true},
    {"task": "返回起点", "completed": false}
  ]
}
```

---

## 5. API 设计

### 5.1 WebSocket 消息扩展

**新请求字段**:
```json
{
  "id": "robot-001",
  "pts": 1704067200000,
  "task": "返回起点",
  "images": {
    "front_1": "base64...",
    "camera_1": "base64...",
    "camera_2": "base64...",
    "camera_3": "base64...",
    "camera_4": "base64..."
  },
  "memory_mode": "enabled",
  "return_to_start": true
}
```

**新响应字段**:
```json
{
  "status": "success",
  "id": "robot-001",
  "pts": 1704067200000,
  "task_status": "executing",
  "action": [[0.5, 0.0, 0.13]],
  "pixel_target": [0.5, 0.5],
  "memory_info": {
    "current_node_id": "node-uuid",
    "recognized_location": "A走廊玻璃门附近",
    "path_to_start": 5,
    "confidence": 0.92
  }
}
```

### 5.2 记忆管理命令

```json
// 将记忆保存到磁盘
{"command": "save_memory"}

// 从磁盘加载记忆
{"command": "load_memory"}

// 清除所有记忆
{"command": "clear_memory"}

// 通过描述查询记忆
{
  "command": "query_memory",
  "description": "电梯区域"
}

// 获取到指定位置的路径
{
  "command": "plan_path",
  "target": "起点"
}
```

---

## 6. 与现有 InternNav 的集成

### 6.1 对 ws_proxy.py 的最小改动

记忆系统设计为围绕现有导航服务的**可选增强**：

```python
# 在 ws_proxy_with_memory.py 中
from visual_memory_system import VisualMemorySystem

class InferenceWithMemory:
    """
    为 InternNav 推理添加记忆功能的包装器。
    """
    def __init__(self, base_agent, memory_system: VisualMemorySystem):
        self.base_agent = base_agent
        self.memory = memory_system

    async def process_inference(self, message_data, session_state):
        # 检查记忆相关命令
        if message_data.get('return_to_start'):
            return await self._handle_return_to_start(message_data, session_state)

        # 带记忆记录的标准推理
        result = await original_process_inference(message_data, session_state, self.base_agent)

        # 如果适用则记录关键帧
        if self._should_record_keyframe(result):
            await self._record_keyframe(message_data, result, session_state)

        # 向响应添加记忆信息
        result['memory_info'] = self._get_memory_info(session_state)

        return result
```

### 6.2 特征提取器复用

复用 InternNav 现有的 LongCLIP 编码器：

```python
# 从 internnav/model/encoder/image_clip_encoder.py 复用
from internnav.model.encoder.image_clip_encoder import ImageEncoder
from internnav.model.basemodel.LongCLIP.model import longclip

class InternNavFeatureExtractor:
    """
    复用 InternNav LongCLIP 集成的特征提取器。
    """
    def __init__(self, model_path: str):
        self.model, self.processor = longclip.load(model_path)

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 使用与 InternNav 相同的预处理
        processed = self.processor(Image.fromarray(image))
        with torch.no_grad():
            features = self.model.encode_image(processed.unsqueeze(0))
        return features.cpu().numpy().flatten()
```

---

## 7. 性能优化

### 7.1 GPU 加速

```python
# 使用 GPU 加速的 FAISS
import faiss

# 初始化 GPU 资源
gpu_resources = faiss.StandardGpuResources()

# 创建 GPU 索引
cpu_index = faiss.IndexFlatL2(feature_dim)
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
```

### 7.2 异步处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMemorySystem:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def record_keyframe_async(self, data):
        """非阻塞关键帧记录。"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._record_keyframe_sync,
            data
        )
```

### 7.3 VLM 批处理

```python
class BatchedSceneDescriptor:
    """
    批量处理多帧以实现高效 VLM 处理。
    """
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.pending_frames = []

    async def process_batch(self):
        if len(self.pending_frames) >= self.batch_size:
            batch = self.pending_frames[:self.batch_size]
            self.pending_frames = self.pending_frames[self.batch_size:]
            # 使用 VLM 处理批次
            return await self._vlm_batch_inference(batch)
```

### 7.4 记忆持久化策略

```python
class PersistenceManager:
    """
    管理记忆状态的定期保存。
    """
    def __init__(self, save_interval: int = 60):
        self.save_interval = save_interval
        self.last_save = time.time()

    async def maybe_save(self, memory_system):
        if time.time() - self.last_save > self.save_interval:
            await memory_system.save_async()
            self.last_save = time.time()
```

---

## 8. 参考实现

### 8.1 关键参考仓库

| 仓库 | 用途 | 位置 |
|------|------|------|
| AnyLoc | 基于 DINOv2+VLAD 的通用 VPR | `reference_code/AnyLoc/` |
| LightRAG | 基于图的 RAG 系统 | `reference_code/LightRAG/` |
| FAISS | 向量相似度搜索 | PyPI: `faiss-gpu` |
| DINOv2 | 自监督视觉特征 | PyTorch Hub |
| LightGlue | 特征匹配 | `reference_code/LightGlue/` |

### 8.2 关键代码参考

**AnyLoc VLAD 实现** (`reference_code/AnyLoc/utilities.py`):
- `DinoV2ExtractFeatures`: DINOv2 特征提取
- `VLAD`: 位置识别的向量聚合

**LightRAG 图存储** (`reference_code/LightRAG/lightrag/`):
- `NetworkXStorage`: 基于图的知识存储
- `NanoVectorDBStorage`: 轻量级向量存储

---

## 9. 实施路线图

### 第一阶段：核心基础设施（第 1-2 周）
- [ ] 使用 LongCLIP 实现 `VisualFeatureExtractor`
- [ ] 使用 FAISS 实现 `MemoryNodeManager`
- [ ] 使用 NetworkX 实现 `GraphEdgeManager`
- [ ] 所有组件的单元测试

### 第二阶段：集成（第 3 周）
- [ ] 创建 `ws_proxy_with_memory.py`
- [ ] 与现有 InternNav agent 集成
- [ ] 添加多相机支持
- [ ] WebSocket API 扩展

### 第三阶段：VLM 集成（第 4 周）
- [ ] 使用 Qwen3-VL 实现 `SceneDescriptionGenerator`
- [ ] 添加语义标签
- [ ] 优化批处理

### 第四阶段：路径规划（第 5 周）
- [ ] 实现 `VisualPlaceRecognizer`
- [ ] 实现带返回起点功能的 `PathPlanner`
- [ ] 语义路径规划

### 第五阶段：优化与测试（第 6 周）
- [ ] FAISS GPU 加速
- [ ] 异步处理优化
- [ ] 端到端测试
- [ ] 性能基准测试

---

## 附录 A：依赖项

```
# 核心依赖
torch>=2.0.0
numpy>=1.24.0
faiss-gpu>=1.7.4  # 或 faiss-cpu
networkx>=3.0
Pillow>=9.0.0

# VLM 依赖（可选，用于场景描述）
transformers>=4.37.0
accelerate>=0.25.0

# 存储
sqlite3  # 内置

# 现有 InternNav 依赖
# (LongCLIP 等已可用)
```

## 附录 B：配置模板

```yaml
# memory_config.yaml
memory:
  enabled: true
  feature_dim: 768
  index_path: "./memory_index"

  faiss:
    index_type: "IVF-PQ"
    nlist: 100
    m: 8
    use_gpu: true

  recognition:
    similarity_threshold: 0.7
    top_k: 5

  keyframe:
    min_interval_seconds: 2.0
    pixel_target_required: true

  vlm:
    enabled: true
    model: "Qwen/Qwen2.5-VL-7B-Instruct"
    batch_size: 4

  persistence:
    auto_save: true
    save_interval_seconds: 60
```

---

*文档版本: 1.0*
*最后更新: 2026-01-16*
*作者: Jianxiong*
