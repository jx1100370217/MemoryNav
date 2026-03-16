#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav v2.0 - 记忆构建器

从标注数据构建记忆拓扑图：
1. 加载 merged_labeled_data 目录结构
2. 解析 node_position_info.json
3. 提取环视图特征
4. 构建节点和边
5. 生成 VPR 索引
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .memory_models import MemoryNode, MemoryEdge
from .memory_graph import MemoryGraph
from .memory_vpr import MemoryVPR
from .anyloc_extractor import AnyLocExtractor
from .vpr_config_loader import load_vpr_config, get_threshold, get_order_invariant
from .vpr_factory import create_vpr_extractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    特征提取器基类
    
    可以被替换为不同的特征提取器
    """
    
    def __init__(self, feature_dim: int = 768, device: str = "cuda:0"):
        self.feature_dim = feature_dim
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型（子类实现）"""
        pass
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像特征
        
        Args:
            image: BGR/RGB 图像
            
        Returns:
            特征向量 (feature_dim,)
        """
        # 默认返回随机特征（用于测试）
        return np.random.randn(self.feature_dim).astype(np.float32)
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            images: 图像列表
            
        Returns:
            特征矩阵 (N, feature_dim)
        """
        features = [self.extract(img) for img in images]
        return np.array(features)



class MemoryBuilder:
    """
    记忆构建器
    
    从标注数据目录构建完整的记忆拓扑图
    """
    
    def __init__(self, 
                 feature_extractor: FeatureExtractor = None,
                 feature_dim: int = 768,
                 device: str = "cuda:0",
                 vpr_method: str = "selavpr",
                 anyloc_config: Dict = None):
        """
        初始化构建器
        
        Args:
            feature_extractor: 特征提取器，None则根据vpr_method自动选择
            feature_dim: 特征维度
            device: 计算设备
            vpr_method: VPR方法 'anyloc', 'megaloc', 'effovpr', 'selavpr'
            anyloc_config: AnyLoc配置参数 (可选)
                - dino_model: DINOv2模型名 (默认 'dinov2_vitb14')
                - desc_facet: 特征facet (默认 'value')
                - agg_mode: 聚合模式 'vlad'/'gem' (默认 'vlad')
                - num_clusters: VLAD聚类数 (默认 32)
                - domain: 预训练域 (默认 'indoor')
                - max_img_size: 最大图像边长 (默认 630)
        """
        self.vpr_method = vpr_method
        self.device = device
        
        # 特征提取器
        if feature_extractor is not None:
            self.extractor = feature_extractor
            self.feature_dim = feature_dim
            order_invariant = get_order_invariant()
        else:
            self.extractor, self.feature_dim, order_invariant = create_vpr_extractor(
                vpr_method=vpr_method, device=device, config=anyloc_config)
        
        # 记忆图和VPR
        self.graph = MemoryGraph()
        # 从统一配置获取相似度阈值
        _cfg = load_vpr_config()
        _threshold = get_threshold(_cfg)
        self.vpr = MemoryVPR(feature_dim=self.feature_dim, similarity_threshold=_threshold, order_invariant=order_invariant)
        
        logger.info(f"[MemoryBuilder] 初始化完成: method={vpr_method}, dim={self.feature_dim}, device={device}")
    
    def build_from_directory(self, data_dir: str, 
                             extract_features: bool = True,
                             save_path: str = None) -> Tuple[MemoryGraph, MemoryVPR]:
        """
        从标注数据目录构建记忆图
        
        Args:
            data_dir: 数据目录路径 (如 merged_labeled_data/)
            extract_features: 是否提取视觉特征
            save_path: 保存路径（可选）
            
        Returns:
            (MemoryGraph, MemoryVPR)
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        logger.info(f"[MemoryBuilder] 开始构建记忆图: {data_dir}")
        
        # 清空现有数据
        self.graph.clear()
        self.vpr.clear()
        
        # 遍历所有节点目录
        node_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()], 
                          key=lambda x: int(x.name) if x.name.isdigit() else 0)
        
        total_nodes = len(node_dirs)
        logger.info(f"[MemoryBuilder] 发现 {total_nodes} 个节点目录")
        
        for idx, node_dir in enumerate(node_dirs):
            try:
                node = self._build_node(node_dir, extract_features)
                if node:
                    self.graph.add_node(node)
                    
                    # 添加到VPR索引
                    self.vpr.add_node_features(
                        node_id=node.node_id,
                        node_name=node.node_name,
                        node_name_eng=getattr(node, 'node_name_eng', ''),
                        camera_features=node.camera_features,
                        fused_feature=node.fused_feature
                    )
                    
                    logger.info(f"[MemoryBuilder] [{idx+1}/{total_nodes}] "
                               f"节点 {node.node_id} ({node.node_name}) 构建完成")
            except Exception as e:
                logger.error(f"[MemoryBuilder] 节点 {node_dir.name} 构建失败: {e}")
        
        # 如果使用AnyLoc VLAD模式，训练词汇表并重新提取特征
        if self.vpr_method == 'anyloc' and hasattr(self.extractor, 'agg_mode'):
            if self.extractor.agg_mode == 'vlad' and not self.extractor.is_vlad_ready():
                logger.info("[MemoryBuilder] 训练 AnyLoc VLAD 词汇表...")
                self.extractor.fit_vlad_vocabulary()
                # 重新提取所有节点的 VLAD 特征
                logger.info("[MemoryBuilder] 使用 VLAD 重新提取特征...")
                self.vpr.clear()
                for node_id, node in self.graph.nodes.items():
                    camera_features = {}
                    features_list = []
                    for cam_id, img_path in node.camera_images.items():
                        if os.path.exists(img_path):
                            try:
                                image = cv2.imread(img_path) if CV2_AVAILABLE else None
                                if image is not None:
                                    feat = self.extractor.extract(image)
                                    camera_features[cam_id] = feat
                                    features_list.append(feat)
                            except Exception as e:
                                logger.warning(f"[MemoryBuilder] VLAD特征重提取失败 {img_path}: {e}")
                    if features_list:
                        fused = np.mean(features_list, axis=0).astype(np.float32)
                        fused = fused / (np.linalg.norm(fused) + 1e-8)
                        node.camera_features = camera_features
                        node.fused_feature = fused
                    self.vpr.add_node_features(
                        node_id=node.node_id,
                        node_name=node.node_name,
                        node_name_eng=getattr(node, 'node_name_eng', ''),
                        camera_features=node.camera_features,
                        fused_feature=node.fused_feature
                    )
        
        # 保存
        if save_path:
            self.save(save_path)
        
        stats = self.graph.get_stats()
        logger.info(f"[MemoryBuilder] 记忆图构建完成: {stats['total_nodes']} 节点, "
                   f"{stats['total_edges']} 边")
        
        return self.graph, self.vpr
    
    def _build_node(self, node_dir: Path, extract_features: bool) -> Optional[MemoryNode]:
        """
        构建单个节点
        
        Args:
            node_dir: 节点目录
            extract_features: 是否提取特征
            
        Returns:
            MemoryNode 或 None
        """
        # 读取节点信息
        info_file = node_dir / "node_position_info.json"
        if not info_file.exists():
            logger.warning(f"[MemoryBuilder] 节点信息文件不存在: {info_file}")
            return None
        
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        self_pos = info.get('self_position', {})
        next_positions = info.get('next_positions', [])
        
        node_id = str(self_pos.get('position_id', node_dir.name))
        node_name = self_pos.get('position_name', f'Node_{node_id}')
        node_name_eng = self_pos.get('position_name_eng', '')
        
        # 相机图路径
        camera_images = {}
        for cam_key in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            img_name = self_pos.get(cam_key, '')
            if img_name:
                camera_images[cam_key] = str(node_dir / img_name)
        
        # 提取时间戳
        timestamp = ""
        for img_name in camera_images.values():
            if img_name:
                base = os.path.basename(img_name)
                timestamp = base.split('_')[0]
                break
        
        # 构建边（新方案：camera_name + crop 子图）
        edges = []
        for next_pos in next_positions:
            camera_name = next_pos.get("camera_name", "")

            # camera_name 必须存在
            if not camera_name:
                logger.debug(f"[MemoryBuilder] 跳过无 camera_name 的边: "
                            f"{node_id} -> {next_pos.get('position_id', '?')}")
                continue

            # 解析三级子图路径 crop_image_paths (big/mid/small)
            raw_crop_paths = next_pos.get("crop_image_paths", {})
            crop_image_paths = {}
            for scale_key, rel_path in raw_crop_paths.items():
                if rel_path:
                    crop_image_paths[scale_key] = str(node_dir / rel_path)

            # crop_image_path 指向 big（兼容引用）
            crop_image_path_rel = next_pos.get("crop_image_path", "")
            crop_path = str(node_dir / crop_image_path_rel) if crop_image_path_rel else ""

            # 解析 normalized boxes (逗号分隔浮点数字符串)
            def _parse_norm_box(raw):
                if isinstance(raw, str) and raw:
                    return tuple(float(x) for x in raw.split(','))
                elif isinstance(raw, (list, tuple)):
                    return tuple(float(x) for x in raw)
                return ()

            big_box = _parse_norm_box(next_pos.get("big_box", ""))
            mid_box = _parse_norm_box(next_pos.get("mid_box", ""))
            small_box = _parse_norm_box(next_pos.get("small_box", ""))

            edge = MemoryEdge(
                target_node_id=str(next_pos.get("position_id", "")),
                target_node_name=next_pos.get("position_name", ""),
                target_node_name_eng=next_pos.get("position_name_eng", ""),
                camera_name=camera_name,
                landmark_name=next_pos.get("landmark_name", ""),
                landmark_name_eng=next_pos.get("landmark_name_eng", ""),
                crop_image_paths=crop_image_paths,
                crop_image_path=crop_path,
                big_box=big_box,
                mid_box=mid_box,
                small_box=small_box,
            )
            edges.append(edge)

        # 提取视觉特征
        camera_features = {}
        fused_feature = None
        
        if extract_features:
            features_list = []
            
            for cam_id, img_path in camera_images.items():
                if os.path.exists(img_path):
                    try:
                        image = cv2.imread(img_path) if CV2_AVAILABLE else None
                        if image is not None:
                            feat = self.extractor.extract(image)
                            camera_features[cam_id] = feat
                            features_list.append(feat)
                    except Exception as e:
                        logger.warning(f"[MemoryBuilder] 特征提取失败 {img_path}: {e}")
            
            # 对AnyLoc VLAD模式：收集patch描述子用于训练词汇表
            if self.vpr_method == 'anyloc' and hasattr(self.extractor, 'collect_descriptors'):
                for cam_id, img_path in camera_images.items():
                    if os.path.exists(img_path):
                        try:
                            image = cv2.imread(img_path) if CV2_AVAILABLE else None
                            if image is not None:
                                self.extractor.collect_descriptors(image)
                        except Exception as e:
                            pass
            
            # 融合特征（平均）
            if features_list:
                fused_feature = np.mean(features_list, axis=0).astype(np.float32)
                fused_feature = fused_feature / (np.linalg.norm(fused_feature) + 1e-8)
        
        # 创建节点
        node = MemoryNode(
            node_id=node_id,
            node_name=node_name,
            node_name_eng=node_name_eng,
            camera_images=camera_images,
            camera_features=camera_features,
            fused_feature=fused_feature,
            edges=edges,
            base_path=str(node_dir),
            timestamp=timestamp
        )
        
        return node
    
    def save(self, path: str):
        """
        保存记忆图和VPR索引
        
        Args:
            path: 保存路径（目录或文件前缀）
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图
        graph_path = str(save_path) + "_graph.pkl" if not str(save_path).endswith('.pkl') else str(save_path)
        self.graph.save(graph_path)
        
        logger.info(f"[MemoryBuilder] 记忆数据已保存到 {path}")
    
    def load(self, path: str) -> Tuple[MemoryGraph, MemoryVPR]:
        """
        加载记忆图和VPR索引
        
        Args:
            path: 加载路径
            
        Returns:
            (MemoryGraph, MemoryVPR)
        """
        graph_path = str(path) + "_graph.pkl" if not str(path).endswith('.pkl') else str(path)
        
        if os.path.exists(graph_path):
            self.graph.load(graph_path)
            
            # 校验缓存特征维度与当前VPR方案是否匹配
            for _nid, _node in self.graph.nodes.items():
                if _node.camera_features:
                    _sample_feat = list(_node.camera_features.values())[0]
                    import numpy as np
                    _cached_dim = np.array(_sample_feat).flatten().shape[0]
                    if _cached_dim != self.feature_dim:
                        raise ValueError(
                            f"[MemoryBuilder] 缓存特征维度({_cached_dim})与当前VPR方案"
                            f"({self.vpr_method}, dim={self.feature_dim})不匹配! "
                            f"请用当前方案重新构建: bash deploy/build_memory.sh"
                        )
                break
            
            # 重建VPR索引
            self.vpr.clear()
            for node_id, node in self.graph.nodes.items():
                self.vpr.add_node_features(
                    node_id=node.node_id,
                    node_name=node.node_name,
                    node_name_eng=getattr(node, 'node_name_eng', ''),
                    camera_features=node.camera_features,
                    fused_feature=node.fused_feature
                )
            
            logger.info(f"[MemoryBuilder] 记忆数据已从 {path} 加载 (dim={self.feature_dim}, method={self.vpr_method})")
        else:
            logger.warning(f"[MemoryBuilder] 文件不存在: {graph_path}")
        
        return self.graph, self.vpr
    
    def get_stats(self) -> Dict:
        """获取构建器统计信息"""
        return {
            'graph': self.graph.get_stats(),
            'vpr': self.vpr.get_stats()
        }
