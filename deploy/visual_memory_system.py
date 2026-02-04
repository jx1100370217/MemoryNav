#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternNav 视觉记忆导航系统

完整的视觉记忆导航系统实现，包含：
1. LongCLIP 视觉特征提取器
2. Qwen2.5-VL 场景描述生成器
3. FAISS 向量索引的记忆节点管理器
4. NetworkX 图边管理器
5. 视觉位置识别模块
6. 路径规划器

作者: Jianxiong
日期: 2026-01-19
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import base64
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import cv2

import torch
from torch import nn
from torch.nn import functional as F

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 项目根目录
project_root = Path(__file__).parent.parent

# 尝试导入 FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS 已加载")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 不可用，将使用 numpy 相似度搜索")

# 尝试导入 NetworkX
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    logger.info("NetworkX 已加载")
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX 不可用，拓扑图功能受限")


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class VisualMemoryConfig:
    """视觉记忆系统配置"""
    # LongCLIP 特征提取器
    longclip_model_path: str = str(project_root / "checkpoints/longclip-B.pt")
    feature_extractor_device: str = "cuda:0"
    feature_dim: int = 768  # LongCLIP-B 输出维度

    # VLM 场景描述器 (Qwen2.5-VL)
    vlm_enabled: bool = True
    vlm_model_path: str = "/home/ubuntu/Disk/models/vlm/Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_device: str = "cuda:0"
    vlm_max_new_tokens: int = 256

    # FAISS 索引配置
    faiss_index_type: str = "Flat"  # "Flat", "IVF", "IVFPQ"
    faiss_nlist: int = 100  # IVF 聚类数
    faiss_m: int = 8  # PQ 子量化器数
    faiss_use_gpu: bool = True

    # 视觉位置识别
    similarity_threshold: float = 0.85
    recognition_top_k: int = 5
    min_time_gap: float = 30.0  # 回环检测最小时间间隔

    # 记忆节点管理
    max_nodes: int = 1000
    node_merge_threshold: float = 0.90
    keyframe_interval: int = 10

    # 持久化
    memory_save_path: str = str(project_root / "memory_data")
    auto_save_interval: int = 300  # 秒

    # 环视相机融合
    use_surround_cameras: bool = True
    surround_camera_ids: List[str] = field(default_factory=lambda: ['camera_1', 'camera_2', 'camera_3', 'camera_4'])


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class MemoryNode:
    """记忆节点 - 代表一个关键帧位置"""
    node_id: str
    timestamp: float
    visual_features: Dict[str, List[float]]  # camera_id -> feature
    global_descriptor: List[float]  # 融合后的全局描述符
    scene_description: str = ""  # VLM 生成的场景描述
    semantic_labels: List[str] = field(default_factory=list)  # 语义标签
    pixel_target: Optional[List[float]] = None  # 导航像素目标
    task_instruction: str = ""  # 当前任务指令
    position_estimate: Optional[List[float]] = None  # [x, y, yaw]
    visit_count: int = 1
    is_landmark: bool = False

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "scene_description": self.scene_description,
            "semantic_labels": self.semantic_labels,
            "task_instruction": self.task_instruction,
            "pixel_target": self.pixel_target,
            "position_estimate": self.position_estimate,
            "visit_count": self.visit_count,
            "is_landmark": self.is_landmark
        }


@dataclass
class GraphEdge:
    """图边 - 表示两个节点间的连接"""
    from_node_id: str
    to_node_id: str
    action: List[float]  # [x, y, yaw]
    traversal_time: float
    description: str = ""
    weight: float = 1.0


# ============================================================================
# 特征序列化工具 (使用 JSON + base64 替代 pickle)
# ============================================================================

def serialize_features(features: Dict[str, Any]) -> str:
    """将特征字典序列化为 JSON 字符串"""
    serializable = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            # 将 numpy 数组转换为 base64 编码的字符串
            serializable[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode('ascii')
            }
        elif isinstance(value, list):
            serializable[key] = value
        else:
            serializable[key] = value
    return json.dumps(serializable)


def deserialize_features(data_str: str) -> Dict[str, Any]:
    """从 JSON 字符串反序列化特征字典"""
    data = json.loads(data_str)
    result = {}
    for key, value in data.items():
        if isinstance(value, dict) and "dtype" in value and "data" in value:
            # 从 base64 还原 numpy 数组
            arr_bytes = base64.b64decode(value["data"])
            arr = np.frombuffer(arr_bytes, dtype=np.dtype(value["dtype"]))
            arr = arr.reshape(value["shape"])
            result[key] = arr.tolist()
        else:
            result[key] = value
    return result


# ============================================================================
# 尝试导入 OpenCLIP 作为备用方案
# ============================================================================
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
    logger.info("OpenCLIP 已加载")
except ImportError:
    OPENCLIP_AVAILABLE = False
    logger.warning("OpenCLIP 不可用")


# ============================================================================
# LongCLIP 视觉特征提取器
# ============================================================================

class LongCLIPFeatureExtractor:
    """
    基于 LongCLIP 的视觉特征提取器

    使用 InternNav 已集成的 LongCLIP 模型提取图像特征，
    用于视觉位置识别 (VPR) 和回环检测。
    如果 LongCLIP 不可用，自动回退到 OpenCLIP。
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        初始化 LongCLIP 特征提取器

        Args:
            model_path: LongCLIP 模型权重路径
            device: 推理设备
        """
        self.device = torch.device(device)
        self.is_available = False
        self.use_openclip = False
        self.feature_dim = 768  # 默认特征维度

        logger.info(f"加载 LongCLIP 模型: {model_path}")

        try:
            # 添加项目路径以导入 longclip
            sys.path.insert(0, str(project_root))
            from internnav.model.basemodel.LongCLIP.model import longclip

            # 加载 LongCLIP 模型
            self.model, self.preprocess = longclip.load(model_path, device=device)

            # 仅保留视觉编码器，删除文本部分以节省内存
            for attr in ['token_embedding', 'transformer', 'positional_embedding', 'ln_final']:
                if hasattr(self.model, attr):
                    try:
                        delattr(self.model, attr)
                    except:
                        pass

            # 设置为评估模式并冻结参数
            self.model.visual.requires_grad_(False)

            self.is_available = True
            logger.info("LongCLIP 特征提取器初始化成功")

        except Exception as e:
            logger.error(f"LongCLIP 加载失败: {e}")
            self.model = None
            self.preprocess = None

            # 尝试使用 OpenCLIP 作为备用
            if OPENCLIP_AVAILABLE:
                logger.info("尝试使用 OpenCLIP 作为备用特征提取器...")
                self._init_openclip()

    def extract_feature(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        提取单张图像的视觉特征

        Args:
            rgb_image: RGB图像 [H, W, 3], uint8 或 float

        Returns:
            feature: 归一化特征向量 [feature_dim]
        """
        if not self.is_available:
            return self._simple_fallback_extract(rgb_image)

        # 如果使用 OpenCLIP，调用专门的方法
        if self.use_openclip:
            return self._openclip_extract(rgb_image)

        # LongCLIP 特征提取
        try:
            # 确保是 uint8 格式
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)

            # 确保是 RGB 格式
            if len(rgb_image.shape) == 2:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
            elif rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[:, :, :3]

            pil_image = Image.fromarray(rgb_image)

            # 预处理
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                feature = self._encode_image(image_tensor)

            # 转换为 numpy 并归一化
            feature = feature.cpu().numpy().flatten().astype('float32')
            feature = feature / (np.linalg.norm(feature) + 1e-8)

            return feature

        except Exception as e:
            logger.warning(f"LongCLIP 特征提取失败: {e}")
            return self._simple_fallback_extract(rgb_image)

    def _encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """使用 LongCLIP 视觉编码器提取特征"""
        visual = self.model.visual

        try:
            data_type = visual.conv1.weight.dtype
        except:
            data_type = torch.float32

        x = image_tensor.type(data_type)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # CLS token + 位置编码
        x = torch.cat([
            visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        # CLS token 输出
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj

        return x

    def _init_openclip(self):
        """初始化 OpenCLIP 作为备用特征提取器"""
        try:
            # 使用 ViT-B/32 模型，与 LongCLIP 类似的架构
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k',
                device=self.device
            )
            self.model.requires_grad_(False)
            self.feature_dim = 512  # ViT-B/32 输出维度
            self.is_available = True
            self.use_openclip = True
            logger.info("OpenCLIP (ViT-B/32) 初始化成功，特征维度: 512")
        except Exception as e:
            logger.error(f"OpenCLIP 初始化失败: {e}")
            self.model = None
            self.preprocess = None

    def _openclip_extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """使用 OpenCLIP 提取特征"""
        try:
            # 确保是 uint8 格式
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)

            # 确保是 RGB 格式
            if len(rgb_image.shape) == 2:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
            elif rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[:, :, :3]

            pil_image = Image.fromarray(rgb_image)
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model.encode_image(image_tensor)
                feature = feature.cpu().numpy().flatten().astype('float32')
                feature = feature / (np.linalg.norm(feature) + 1e-8)

            return feature

        except Exception as e:
            logger.warning(f"OpenCLIP 特征提取失败: {e}")
            return self._simple_fallback_extract(rgb_image)

    def _fallback_extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """回退方案：优先使用 OpenCLIP，否则使用简化特征"""
        if self.use_openclip and self.is_available:
            return self._openclip_extract(rgb_image)
        return self._simple_fallback_extract(rgb_image)

    def _simple_fallback_extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """最简化的回退方案：基于图像直方图和颜色特征"""
        # 缩小图像
        image_small = cv2.resize(rgb_image, (64, 64))

        # 计算颜色直方图特征
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([image_small], [i], None, [64], [0, 256])
            hist_features.append(hist.flatten())
        color_hist = np.concatenate(hist_features)  # 192 维

        # 计算 HOG 类特征 (简化版)
        gray = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3).flatten()[:128]
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3).flatten()[:128]

        # 组合特征 - 使用当前特征维度
        feature = np.zeros(self.feature_dim, dtype='float32')
        feature[:192] = color_hist
        if self.feature_dim >= 448:
            feature[192:320] = grad_x / (np.max(np.abs(grad_x)) + 1e-8)
            feature[320:448] = grad_y / (np.max(np.abs(grad_y)) + 1e-8)

            # 添加图像均值和标准差
            for i in range(3):
                if 448 + i*2 + 1 < self.feature_dim:
                    feature[448 + i*2] = np.mean(image_small[:, :, i]) / 255.0
                    feature[448 + i*2 + 1] = np.std(image_small[:, :, i]) / 255.0
        else:
            # 对于较小的特征维度，使用更紧凑的表示
            feature[192:min(320, self.feature_dim)] = grad_x[:min(128, self.feature_dim - 192)] / (np.max(np.abs(grad_x)) + 1e-8)
            if self.feature_dim > 320:
                feature[320:min(448, self.feature_dim)] = grad_y[:min(128, self.feature_dim - 320)] / (np.max(np.abs(grad_y)) + 1e-8)

        # 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        features = [self.extract_feature(img) for img in images]
        return np.array(features, dtype='float32')


# ============================================================================
# Qwen2.5-VL 场景描述生成器
# ============================================================================

class SceneDescriptionGenerator:
    """
    使用 Qwen2.5-VL 生成场景描述和语义标签
    """

    def __init__(self, model_path: str, device: str = "cuda:0", max_new_tokens: int = 256):
        """
        初始化场景描述生成器

        Args:
            model_path: Qwen2.5-VL 模型路径
            device: 推理设备
            max_new_tokens: 最大生成 token 数
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.is_available = False

        logger.info(f"加载 Qwen2.5-VL 模型: {model_path}")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

            self.model.requires_grad_(False)
            self.is_available = True
            logger.info("Qwen2.5-VL 场景描述生成器初始化成功")

        except Exception as e:
            logger.error(f"Qwen2.5-VL 加载失败: {e}")
            self.model = None
            self.processor = None

    def generate_description(self, images: Dict[str, np.ndarray]) -> str:
        """
        从全景视图生成综合场景描述

        Args:
            images: 相机图像字典 {camera_id: image}

        Returns:
            description: 场景描述文本
        """
        if not self.is_available:
            return "场景描述生成器不可用"

        try:
            # 准备图像 - 使用前两个相机视图
            pil_images = []
            for cam_id in ['camera_1', 'camera_2', 'front_1']:
                if cam_id in images and images[cam_id] is not None:
                    img = images[cam_id]
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img))
                    if len(pil_images) >= 2:
                        break

            if not pil_images:
                return "无可用图像"

            # 构建消息
            prompt = """基于提供的相机视图，用2-3句话描述当前位置。重点关注：
1. 关键地标和物体
2. 环境类型（走廊、房间、室外等）
3. 可用于后续识别此位置的显著特征

请用中文描述。"""

            content = []
            for img in pil_images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=pil_images,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )

            # 解码输出
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            description = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return description.strip()

        except Exception as e:
            logger.warning(f"场景描述生成失败: {e}")
            return f"场景描述生成失败: {str(e)}"

    def extract_semantic_labels(self, images: Dict[str, np.ndarray]) -> List[str]:
        """
        从图像中提取语义标签

        Args:
            images: 相机图像字典

        Returns:
            labels: 语义标签列表
        """
        if not self.is_available:
            return []

        try:
            # 准备图像
            pil_images = []
            for cam_id in ['camera_1', 'camera_2', 'front_1']:
                if cam_id in images and images[cam_id] is not None:
                    img = images[cam_id]
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img))
                    if len(pil_images) >= 2:
                        break

            if not pil_images:
                return []

            prompt = """列出这些图像中可见的关键物体和场景元素。
仅输出逗号分隔的中文名词列表，例如：走廊,门,窗户,电梯,标识牌"""

            content = []
            for img in pil_images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=pil_images,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 解析标签
            labels = [label.strip() for label in response.split(',') if label.strip()]
            return labels[:10]  # 最多返回10个标签

        except Exception as e:
            logger.warning(f"语义标签提取失败: {e}")
            return []


# ============================================================================
# FAISS 记忆节点管理器
# ============================================================================

class MemoryNodeManager:
    """
    使用 FAISS 管理记忆节点，实现高效相似度搜索
    """

    def __init__(self, config: VisualMemoryConfig):
        """
        初始化记忆节点管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.feature_dim = config.feature_dim

        # 创建索引目录
        self.index_path = Path(config.memory_save_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # 初始化 FAISS 索引
        self._init_faiss_index()

        # 节点存储
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_id_to_idx: Dict[str, int] = {}
        self.idx_to_node_id: Dict[int, str] = {}
        self.next_idx = 0

        # SQLite 持久化
        self._init_database()

        logger.info(f"MemoryNodeManager 初始化完成 (index_path={self.index_path})")

    def _init_faiss_index(self):
        """初始化 FAISS 索引"""
        if not FAISS_AVAILABLE:
            self.index = None
            self.is_trained = True
            return

        # 使用 FlatIP 索引（内积，用于余弦相似度）
        if self.config.faiss_index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self.is_trained = True
        elif self.config.faiss_index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.feature_dim,
                self.config.faiss_nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self.is_trained = False
        elif self.config.faiss_index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.feature_dim,
                self.config.faiss_nlist,
                self.config.faiss_m, 8
            )
            self.is_trained = False
        else:
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self.is_trained = True

        # GPU 加速
        if self.config.faiss_use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS GPU 加速已启用")
            except Exception as e:
                logger.warning(f"FAISS GPU 加速失败: {e}")

    def _init_database(self):
        """初始化 SQLite 数据库"""
        db_path = self.index_path / "nodes.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                timestamp REAL,
                scene_description TEXT,
                semantic_labels TEXT,
                task_instruction TEXT,
                pixel_target TEXT,
                position_estimate TEXT,
                visit_count INTEGER,
                is_landmark INTEGER
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS features (
                node_id TEXT PRIMARY KEY,
                global_descriptor TEXT,
                visual_features TEXT
            )
        ''')
        self.conn.commit()

    def add_node(self, node: MemoryNode) -> str:
        """
        添加新的记忆节点

        Args:
            node: 记忆节点

        Returns:
            node_id: 节点 ID
        """
        # 归一化全局描述符
        descriptor = np.array(node.global_descriptor, dtype=np.float32)
        descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-8)
        node.global_descriptor = descriptor.tolist()

        # 训练索引（如果需要）
        if not self.is_trained and FAISS_AVAILABLE:
            if len(self.nodes) >= self.config.faiss_nlist:
                all_descriptors = np.array([
                    n.global_descriptor for n in self.nodes.values()
                ], dtype=np.float32)
                self.index.train(all_descriptors)
                self.index.add(all_descriptors)
                for i, nid in enumerate(self.nodes.keys()):
                    self.node_id_to_idx[nid] = i
                    self.idx_to_node_id[i] = nid
                self.next_idx = len(self.nodes)
                self.is_trained = True

        # 添加到 FAISS 索引
        if FAISS_AVAILABLE and self.is_trained:
            self.index.add(descriptor.reshape(1, -1))
            self.node_id_to_idx[node.node_id] = self.next_idx
            self.idx_to_node_id[self.next_idx] = node.node_id
            self.next_idx += 1

        # 存储节点
        self.nodes[node.node_id] = node

        # 持久化到数据库
        self._save_node_to_db(node)

        logger.debug(f"添加记忆节点: {node.node_id} (总数={len(self.nodes)})")
        return node.node_id

    def search_similar(self, query_descriptor: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        搜索 k 个最相似的节点

        Args:
            query_descriptor: 查询特征
            k: 返回数量

        Returns:
            results: [(node_id, similarity), ...]
        """
        if len(self.nodes) == 0:
            return []

        # 归一化查询
        query = query_descriptor.astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        if FAISS_AVAILABLE and self.is_trained and self.index.ntotal > 0:
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query.reshape(1, -1), k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self.idx_to_node_id:
                    node_id = self.idx_to_node_id[idx]
                    results.append((node_id, float(dist)))
            return results
        else:
            return self._brute_force_search(query, k)

    def _brute_force_search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """暴力搜索回退"""
        results = []
        for node_id, node in self.nodes.items():
            node_desc = np.array(node.global_descriptor)
            similarity = np.dot(query, node_desc)
            results.append((node_id, similarity))
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """获取节点"""
        return self.nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs):
        """更新节点属性"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            for key, value in kwargs.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            self._save_node_to_db(node)

    def _save_node_to_db(self, node: MemoryNode):
        """保存节点到数据库 (使用 JSON 序列化)"""
        self.conn.execute('''
            INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.node_id,
            node.timestamp,
            node.scene_description,
            json.dumps(node.semantic_labels),
            node.task_instruction,
            json.dumps(node.pixel_target) if node.pixel_target else None,
            json.dumps(node.position_estimate) if node.position_estimate else None,
            node.visit_count,
            1 if node.is_landmark else 0
        ))

        # 保存特征 (使用 JSON 序列化)
        self.conn.execute('''
            INSERT OR REPLACE INTO features VALUES (?, ?, ?)
        ''', (
            node.node_id,
            json.dumps(node.global_descriptor),
            json.dumps(node.visual_features)
        ))
        self.conn.commit()

    def save_index(self, path: Optional[str] = None):
        """保存 FAISS 索引"""
        save_path = Path(path) if path else self.index_path

        if FAISS_AVAILABLE and self.is_trained:
            # 如果是 GPU 索引，先转到 CPU
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            except:
                cpu_index = self.index
            faiss.write_index(cpu_index, str(save_path / "faiss.index"))

        # 保存映射
        with open(save_path / "mappings.json", 'w') as f:
            json.dump({
                'node_id_to_idx': self.node_id_to_idx,
                'idx_to_node_id': {str(k): v for k, v in self.idx_to_node_id.items()},
                'next_idx': self.next_idx
            }, f)

        logger.info(f"索引已保存到 {save_path}")

    def load_index(self, path: Optional[str] = None):
        """加载 FAISS 索引"""
        load_path = Path(path) if path else self.index_path

        index_file = load_path / "faiss.index"
        mappings_file = load_path / "mappings.json"

        if index_file.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(index_file))
            if self.config.faiss_use_gpu and faiss.get_num_gpus() > 0:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except:
                    pass
            self.is_trained = True

        if mappings_file.exists():
            with open(mappings_file, 'r') as f:
                mappings = json.load(f)
                self.node_id_to_idx = mappings['node_id_to_idx']
                self.idx_to_node_id = {int(k): v for k, v in mappings['idx_to_node_id'].items()}
                self.next_idx = mappings.get('next_idx', 0)

        # 从数据库加载节点
        self._load_nodes_from_db()

        logger.info(f"索引已从 {load_path} 加载 (节点数={len(self.nodes)})")

    def _load_nodes_from_db(self):
        """从数据库加载节点"""
        cursor = self.conn.execute('SELECT * FROM nodes')
        for row in cursor:
            node_id = row[0]

            # 加载特征
            feat_cursor = self.conn.execute(
                'SELECT global_descriptor, visual_features FROM features WHERE node_id = ?',
                (node_id,)
            )
            feat_row = feat_cursor.fetchone()

            if feat_row:
                global_descriptor = json.loads(feat_row[0])
                visual_features = json.loads(feat_row[1])
            else:
                global_descriptor = [0.0] * self.feature_dim
                visual_features = {}

            node = MemoryNode(
                node_id=node_id,
                timestamp=row[1],
                visual_features=visual_features,
                global_descriptor=global_descriptor,
                scene_description=row[2] or "",
                semantic_labels=json.loads(row[3]) if row[3] else [],
                task_instruction=row[4] or "",
                pixel_target=json.loads(row[5]) if row[5] else None,
                position_estimate=json.loads(row[6]) if row[6] else None,
                visit_count=row[7] or 1,
                is_landmark=bool(row[8])
            )
            self.nodes[node_id] = node

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_nodes": len(self.nodes),
            "index_size": self.index.ntotal if FAISS_AVAILABLE and self.index else 0,
            "is_trained": self.is_trained
        }

    def clear(self):
        """清空所有数据"""
        self.nodes.clear()
        self.node_id_to_idx.clear()
        self.idx_to_node_id.clear()
        self.next_idx = 0

        if FAISS_AVAILABLE:
            self._init_faiss_index()

        self.conn.execute('DELETE FROM nodes')
        self.conn.execute('DELETE FROM features')
        self.conn.commit()


# ============================================================================
# NetworkX 图边管理器
# ============================================================================

class GraphEdgeManager:
    """
    管理记忆节点间的图边关系
    实现 GraphRAG 风格的导航知识图谱
    """

    def __init__(self, config: VisualMemoryConfig):
        """
        初始化图边管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.index_path = Path(config.memory_save_path)

        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            self.edges: Dict[str, List[GraphEdge]] = {}  # 简单后备

        logger.info("GraphEdgeManager 初始化完成")

    def add_edge(self, from_node_id: str, to_node_id: str,
                 action: List[float], traversal_time: float = 1.0,
                 description: str = ""):
        """
        添加有向边

        Args:
            from_node_id: 源节点 ID
            to_node_id: 目标节点 ID
            action: 动作 [x, y, yaw]
            traversal_time: 遍历时间
            description: 边描述
        """
        weight = np.linalg.norm(action[:2]) if len(action) >= 2 else 1.0

        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_edge(
                from_node_id, to_node_id,
                action=action,
                time=traversal_time,
                description=description,
                weight=weight
            )
        else:
            edge = GraphEdge(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                action=action,
                traversal_time=traversal_time,
                description=description,
                weight=weight
            )
            if from_node_id not in self.edges:
                self.edges[from_node_id] = []
            self.edges[from_node_id].append(edge)

    def find_path(self, start_node_id: str, end_node_id: str,
                  method: str = "shortest") -> Optional[List[str]]:
        """
        查找两节点间的路径

        Args:
            start_node_id: 起始节点
            end_node_id: 目标节点
            method: 路径方法 ("shortest", "semantic")

        Returns:
            path: 节点 ID 列表
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return self._simple_path_search(start_node_id, end_node_id)

        try:
            if method == "shortest":
                return nx.shortest_path(self.graph, start_node_id, end_node_id, weight='weight')
            else:
                return nx.shortest_path(self.graph, start_node_id, end_node_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _simple_path_search(self, start: str, end: str) -> Optional[List[str]]:
        """简单的 BFS 路径搜索"""
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            node, path = queue.pop(0)
            if node in self.edges:
                for edge in self.edges[node]:
                    next_node = edge.to_node_id
                    if next_node == end:
                        return path + [next_node]
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [next_node]))

        return None

    def get_actions_for_path(self, path: List[str]) -> List[List[float]]:
        """获取路径对应的动作序列"""
        actions = []

        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i+1]

            if NETWORKX_AVAILABLE and self.graph is not None:
                if self.graph.has_edge(from_node, to_node):
                    edge_data = self.graph.get_edge_data(from_node, to_node)
                    actions.append(edge_data.get('action', [0.0, 0.0, 0.0]))
            elif from_node in self.edges:
                for edge in self.edges[from_node]:
                    if edge.to_node_id == to_node:
                        actions.append(edge.action)
                        break

        return actions

    def semantic_search_path(self, start_node_id: str, target_description: str,
                            node_manager: MemoryNodeManager) -> Optional[List[str]]:
        """
        基于语义描述搜索路径

        Args:
            start_node_id: 起始节点
            target_description: 目标描述
            node_manager: 节点管理器

        Returns:
            path: 到匹配目标的路径
        """
        # 查找匹配描述的节点
        matching_nodes = []
        query_lower = target_description.lower()

        for node_id, node in node_manager.nodes.items():
            # 检查语义标签
            for label in node.semantic_labels:
                if label.lower() in query_lower or query_lower in label.lower():
                    matching_nodes.append(node_id)
                    break
            else:
                # 检查场景描述
                if any(word in node.scene_description.lower() for word in query_lower.split()):
                    matching_nodes.append(node_id)

        # 找到最短路径
        best_path = None
        best_length = float('inf')

        for target_id in matching_nodes:
            path = self.find_path(start_node_id, target_id)
            if path and len(path) < best_length:
                best_path = path
                best_length = len(path)

        return best_path

    def get_neighbors(self, node_id: str) -> List[Tuple[str, Dict]]:
        """获取节点的所有邻居"""
        neighbors = []

        if NETWORKX_AVAILABLE and self.graph is not None:
            for neighbor in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                neighbors.append((neighbor, edge_data))
        elif node_id in self.edges:
            for edge in self.edges[node_id]:
                neighbors.append((edge.to_node_id, {
                    'action': edge.action,
                    'time': edge.traversal_time,
                    'weight': edge.weight
                }))

        return neighbors

    def save_graph(self, path: Optional[str] = None):
        """保存图到磁盘"""
        save_path = Path(path) if path else self.index_path

        if NETWORKX_AVAILABLE and self.graph is not None:
            # 使用 JSON 格式保存 (兼容性更好)
            data = nx.node_link_data(self.graph)
            with open(save_path / "graph.json", 'w') as f:
                json.dump(data, f)
        else:
            with open(save_path / "edges.json", 'w') as f:
                edges_data = {k: [asdict(e) for e in v] for k, v in self.edges.items()}
                json.dump(edges_data, f)

    def load_graph(self, path: Optional[str] = None):
        """从磁盘加载图"""
        load_path = Path(path) if path else self.index_path

        graph_file = load_path / "graph.json"
        edges_file = load_path / "edges.json"

        if graph_file.exists() and NETWORKX_AVAILABLE:
            with open(graph_file, 'r') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        elif edges_file.exists():
            with open(edges_file, 'r') as f:
                edges_data = json.load(f)
                self.edges = {
                    k: [GraphEdge(**e) for e in v]
                    for k, v in edges_data.items()
                }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if NETWORKX_AVAILABLE and self.graph is not None:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            }
        else:
            total_edges = sum(len(v) for v in self.edges.values())
            return {"edges": total_edges}

    def clear(self):
        """清空图"""
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.clear()
        else:
            self.edges.clear()


# ============================================================================
# 视觉位置识别模块
# ============================================================================

class VisualPlaceRecognizer:
    """
    视觉位置识别 (VPR) 模块
    基于特征匹配识别当前位置是否匹配已存储的记忆节点
    """

    def __init__(self, feature_extractor: LongCLIPFeatureExtractor,
                 node_manager: MemoryNodeManager,
                 config: VisualMemoryConfig):
        """
        初始化 VPR 模块

        Args:
            feature_extractor: 特征提取器
            node_manager: 节点管理器
            config: 配置
        """
        self.feature_extractor = feature_extractor
        self.node_manager = node_manager
        self.config = config

        # 缓存
        self.last_query_time = 0.0
        self.last_match_result = None

    def recognize(self, images: Dict[str, np.ndarray]) -> Optional[Tuple[str, float]]:
        """
        尝试从记忆中识别当前位置

        Args:
            images: 相机图像字典

        Returns:
            (node_id, confidence) 或 None
        """
        # 提取特征
        features = {}
        for cam_id in self.config.surround_camera_ids:
            if cam_id in images and images[cam_id] is not None:
                features[cam_id] = self.feature_extractor.extract_feature(images[cam_id])

        # 计算全局描述符
        global_descriptor = self._compute_global_descriptor(features)
        if global_descriptor is None:
            return None

        # 在记忆中搜索
        matches = self.node_manager.search_similar(
            global_descriptor,
            k=self.config.recognition_top_k
        )

        if not matches:
            return None

        best_match_id, best_similarity = matches[0]

        if best_similarity >= self.config.similarity_threshold:
            return (best_match_id, best_similarity)

        return None

    def is_revisited(self, query_feature: np.ndarray,
                     current_time: float) -> Optional[Tuple[str, float]]:
        """
        判断是否为已访问位置 (回环检测)

        Args:
            query_feature: 查询特征
            current_time: 当前时间

        Returns:
            (node_id, similarity) 或 None
        """
        matches = self.node_manager.search_similar(
            query_feature,
            k=self.config.recognition_top_k * 2
        )

        for node_id, similarity in matches:
            node = self.node_manager.get_node(node_id)
            if node is None:
                continue

            time_gap = current_time - node.timestamp
            if time_gap > self.config.min_time_gap and similarity >= self.config.similarity_threshold:
                return (node_id, similarity)

        return None

    def _compute_global_descriptor(self, features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        计算全局描述符 (融合多相机特征)

        Args:
            features: 相机特征字典

        Returns:
            global_descriptor: 融合后的全局描述符
        """
        if not features:
            return None

        # 等权重融合
        fused = None
        count = 0

        for cam_id in self.config.surround_camera_ids:
            if cam_id in features and features[cam_id] is not None:
                if fused is None:
                    fused = features[cam_id].copy()
                else:
                    fused += features[cam_id]
                count += 1

        if fused is None or count == 0:
            return None

        fused = fused / count
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

    def re_rank_matches(self, query_features: Dict[str, np.ndarray],
                        candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        使用逐相机特征匹配重排序候选

        Args:
            query_features: 查询特征
            candidate_ids: 候选节点 ID

        Returns:
            sorted_results: 排序后的结果
        """
        scores = []

        for node_id in candidate_ids:
            node = self.node_manager.get_node(node_id)
            if not node:
                continue

            cam_scores = []
            for cam_id, query_feat in query_features.items():
                if cam_id in node.visual_features:
                    node_feat = np.array(node.visual_features[cam_id])
                    sim = np.dot(query_feat, node_feat) / (
                        np.linalg.norm(query_feat) * np.linalg.norm(node_feat) + 1e-8
                    )
                    cam_scores.append(sim)

            avg_score = np.mean(cam_scores) if cam_scores else 0
            scores.append((node_id, avg_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)


# ============================================================================
# 路径规划器
# ============================================================================

class PathPlanner:
    """
    路径规划器 - 支持返回导航和语义导航
    """

    def __init__(self, node_manager: MemoryNodeManager,
                 edge_manager: GraphEdgeManager,
                 recognizer: VisualPlaceRecognizer):
        """
        初始化路径规划器

        Args:
            node_manager: 节点管理器
            edge_manager: 边管理器
            recognizer: 位置识别器
        """
        self.node_manager = node_manager
        self.edge_manager = edge_manager
        self.recognizer = recognizer

        self.start_node_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        self.path_history: List[str] = []

    def set_start_position(self, node_id: str):
        """设置起始位置"""
        self.start_node_id = node_id
        self.current_node_id = node_id
        self.path_history = [node_id]
        logger.info(f"设置起始位置: {node_id}")

    def update_position(self, images: Dict[str, np.ndarray]) -> Optional[str]:
        """
        基于视觉识别更新当前位置

        Args:
            images: 相机图像

        Returns:
            node_id: 当前节点 ID 或 None
        """
        result = self.recognizer.recognize(images)
        if result:
            self.current_node_id = result[0]
            if self.current_node_id not in self.path_history:
                self.path_history.append(self.current_node_id)
            return self.current_node_id
        return None

    def plan_return_to_start(self) -> Optional[List[List[float]]]:
        """
        规划返回起点的路径

        Returns:
            actions: 动作序列 或 None
        """
        if not self.start_node_id or not self.current_node_id:
            logger.warning("无法规划返回: 起点或当前位置未知")
            return None

        if self.start_node_id == self.current_node_id:
            return [[0.0, 0.0, 0.0]]  # 已在起点

        # 尝试使用图路径
        path = self.edge_manager.find_path(self.current_node_id, self.start_node_id)

        if path:
            actions = self.edge_manager.get_actions_for_path(path)
            logger.info(f"找到返回路径: {len(path)} 节点, {len(actions)} 动作")
            return actions

        # 回退: 使用路径历史反向
        logger.info("使用路径历史反向返回")
        return self._reverse_path_history()

    def _reverse_path_history(self) -> List[List[float]]:
        """从路径历史生成反向动作"""
        if len(self.path_history) < 2:
            return [[0.0, 0.0, 0.0]]

        # 反向路径
        reversed_path = list(reversed(self.path_history))
        actions = self.edge_manager.get_actions_for_path(reversed_path)

        # 反转动作
        reversed_actions = []
        for action in actions:
            reversed_actions.append(self._reverse_action(action))

        return reversed_actions if reversed_actions else [[0.0, 0.0, 0.0]]

    def _reverse_action(self, action: List[float]) -> List[float]:
        """反转动作"""
        if len(action) >= 3:
            return [-action[0], -action[1], -action[2]]
        return [0.0, 0.0, 0.0]

    def plan_to_description(self, target_description: str) -> Optional[List[List[float]]]:
        """
        规划到语义描述位置的路径

        Args:
            target_description: 目标描述

        Returns:
            actions: 动作序列
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

    def get_progress(self) -> Dict:
        """获取规划进度"""
        return {
            "start_node": self.start_node_id,
            "current_node": self.current_node_id,
            "path_length": len(self.path_history),
            "path_history": self.path_history[-10:]  # 最近10个
        }


# ============================================================================
# 环视相机融合器
# ============================================================================

class SurroundCameraFusion:
    """
    环视相机特征融合
    仅使用 camera_1~4 四个环视相机
    """

    def __init__(self, config: VisualMemoryConfig):
        self.config = config
        self.camera_ids = config.surround_camera_ids

    def fuse_features(self, features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        融合环视相机特征

        Args:
            features: {camera_id: feature}

        Returns:
            fused_feature: 融合特征
        """
        if not features:
            return None

        fused = None
        count = 0

        for cam_id in self.camera_ids:
            if cam_id in features and features[cam_id] is not None:
                if fused is None:
                    fused = features[cam_id].copy()
                else:
                    fused += features[cam_id]
                count += 1

        if fused is None or count == 0:
            return None

        fused = fused / count
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused


# ============================================================================
# 视觉记忆导航系统 (集成类)
# ============================================================================

class VisualMemorySystem:
    """
    视觉记忆导航系统 - 集成所有组件
    """

    def __init__(self, config: Optional[VisualMemoryConfig] = None):
        """
        初始化视觉记忆系统

        Args:
            config: 配置对象
        """
        self.config = config or VisualMemoryConfig()

        # 创建存储目录
        Path(self.config.memory_save_path).mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("初始化视觉记忆导航系统")
        logger.info("=" * 60)

        # 初始化组件
        self.feature_extractor = LongCLIPFeatureExtractor(
            self.config.longclip_model_path,
            self.config.feature_extractor_device
        )

        # 更新配置中的特征维度以匹配实际特征提取器
        actual_feature_dim = self.feature_extractor.feature_dim
        if actual_feature_dim != self.config.feature_dim:
            logger.info(f"更新特征维度: {self.config.feature_dim} -> {actual_feature_dim}")
            self.config.feature_dim = actual_feature_dim

        self.scene_generator = None
        if self.config.vlm_enabled:
            try:
                self.scene_generator = SceneDescriptionGenerator(
                    self.config.vlm_model_path,
                    self.config.vlm_device,
                    self.config.vlm_max_new_tokens
                )
            except Exception as e:
                logger.warning(f"VLM 加载失败: {e}")

        self.node_manager = MemoryNodeManager(self.config)
        self.edge_manager = GraphEdgeManager(self.config)

        self.recognizer = VisualPlaceRecognizer(
            self.feature_extractor,
            self.node_manager,
            self.config
        )

        self.path_planner = PathPlanner(
            self.node_manager,
            self.edge_manager,
            self.recognizer
        )

        self.surround_fusion = SurroundCameraFusion(self.config)

        # 状态
        self.last_node_id: Optional[str] = None
        self.last_action: List[float] = [0.0, 0.0, 0.0]
        self.is_recording = False
        self.recording_start_node: Optional[str] = None

        # 异步处理
        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info("视觉记忆导航系统初始化完成")

    def process_observation(self,
                           front_image: np.ndarray,
                           surround_images: Dict[str, np.ndarray] = None,
                           action: List[float] = None,
                           instruction: str = "",
                           generate_description: bool = False) -> Dict[str, Any]:
        """
        处理新观测

        Args:
            front_image: 前置相机图像 (front_1)
            surround_images: 环视相机图像 {camera_1~4: image}
            action: 当前动作 [x, y, yaw]
            instruction: 任务指令
            generate_description: 是否生成场景描述

        Returns:
            result: 处理结果
        """
        current_time = time.time()

        # 提取环视相机特征
        surround_features = {}
        if surround_images:
            for cam_id in self.config.surround_camera_ids:
                if cam_id in surround_images and surround_images[cam_id] is not None:
                    surround_features[cam_id] = self.feature_extractor.extract_feature(
                        surround_images[cam_id]
                    )

        # 融合特征
        global_descriptor = self.surround_fusion.fuse_features(surround_features)

        # 如果没有环视特征，使用前置相机
        if global_descriptor is None:
            global_descriptor = self.feature_extractor.extract_feature(front_image)

        # 检查是否为已访问位置 (回环检测)
        revisit_info = self.recognizer.is_revisited(global_descriptor, current_time)

        result = {
            "timestamp": current_time,
            "is_new_node": True,
            "is_revisited": revisit_info is not None,
            "revisit_info": None,
            "node_id": None,
            "scene_description": "",
            "semantic_labels": []
        }

        if revisit_info:
            matched_node_id, similarity = revisit_info
            result["is_new_node"] = False
            result["revisit_info"] = {
                "matched_node_id": matched_node_id,
                "similarity": similarity
            }
            result["node_id"] = matched_node_id

            # 更新访问计数
            self.node_manager.update_node(matched_node_id, visit_count=
                self.node_manager.get_node(matched_node_id).visit_count + 1
            )

            # 添加边
            if self.last_node_id and self.last_node_id != matched_node_id:
                self.edge_manager.add_edge(
                    self.last_node_id, matched_node_id,
                    self.last_action, 1.0
                )

            self.last_node_id = matched_node_id

        else:
            # 创建新节点
            node_id = str(uuid.uuid4())

            # 生成场景描述
            scene_description = ""
            semantic_labels = []

            if generate_description and self.scene_generator and self.scene_generator.is_available:
                all_images = {"front_1": front_image}
                if surround_images:
                    all_images.update(surround_images)
                scene_description = self.scene_generator.generate_description(all_images)
                semantic_labels = self.scene_generator.extract_semantic_labels(all_images)

            # 创建节点
            node = MemoryNode(
                node_id=node_id,
                timestamp=current_time,
                visual_features={k: v.tolist() for k, v in surround_features.items()},
                global_descriptor=global_descriptor.tolist(),
                scene_description=scene_description,
                semantic_labels=semantic_labels,
                task_instruction=instruction
            )

            self.node_manager.add_node(node)

            # 添加边
            if self.last_node_id:
                self.edge_manager.add_edge(
                    self.last_node_id, node_id,
                    self.last_action, 1.0
                )

            # 设置起始节点
            if self.recording_start_node is None and self.is_recording:
                self.recording_start_node = node_id
                self.path_planner.set_start_position(node_id)

            self.last_node_id = node_id

            result["node_id"] = node_id
            result["scene_description"] = scene_description
            result["semantic_labels"] = semantic_labels

        # 更新状态
        self.last_action = action if action else [0.0, 0.0, 0.0]

        # 更新路径规划器
        self.path_planner.current_node_id = result["node_id"]

        # 添加统计信息
        result["stats"] = {
            "node_stats": self.node_manager.get_stats(),
            "edge_stats": self.edge_manager.get_stats()
        }

        return result

    def start_recording(self, instruction: str = ""):
        """开始记录导航"""
        self.is_recording = True
        self.recording_start_node = self.last_node_id
        if self.last_node_id:
            self.path_planner.set_start_position(self.last_node_id)
        logger.info(f"开始记录导航 (起点={self.recording_start_node})")

    def stop_recording(self):
        """停止记录导航"""
        self.is_recording = False
        logger.info(f"停止记录导航 (节点数={len(self.node_manager.nodes)})")

    def plan_return_to_start(self) -> Optional[List[List[float]]]:
        """规划返回起点"""
        return self.path_planner.plan_return_to_start()

    def recognize_location(self, images: Dict[str, np.ndarray]) -> Optional[Tuple[str, float]]:
        """识别当前位置"""
        return self.recognizer.recognize(images)

    def save(self, path: Optional[str] = None):
        """保存记忆到磁盘"""
        self.node_manager.save_index(path)
        self.edge_manager.save_graph(path)
        logger.info("记忆已保存")

    def load(self, path: Optional[str] = None):
        """从磁盘加载记忆"""
        self.node_manager.load_index(path)
        self.edge_manager.load_graph(path)
        logger.info(f"记忆已加载 (节点数={len(self.node_manager.nodes)})")

    def clear(self):
        """清空所有记忆"""
        self.node_manager.clear()
        self.edge_manager.clear()
        self.last_node_id = None
        self.recording_start_node = None
        self.is_recording = False
        self.path_planner.start_node_id = None
        self.path_planner.current_node_id = None
        self.path_planner.path_history.clear()
        logger.info("记忆已清空")

    def get_stats(self) -> Dict:
        """获取系统统计"""
        return {
            "node_stats": self.node_manager.get_stats(),
            "edge_stats": self.edge_manager.get_stats(),
            "planner_progress": self.path_planner.get_progress(),
            "is_recording": self.is_recording,
            "feature_extractor_available": self.feature_extractor.is_available,
            "scene_generator_available": self.scene_generator.is_available if self.scene_generator else False
        }


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    # 简单测试
    config = VisualMemoryConfig()
    config.vlm_enabled = False  # 测试时禁用 VLM 以加快速度

    system = VisualMemorySystem(config)

    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 处理观测
    result = system.process_observation(
        front_image=test_image,
        surround_images={
            'camera_1': test_image,
            'camera_2': test_image,
            'camera_3': test_image,
            'camera_4': test_image
        },
        action=[0.5, 0.0, 0.1],
        instruction="测试导航"
    )

    print("处理结果:", result)
    print("系统统计:", system.get_stats())
