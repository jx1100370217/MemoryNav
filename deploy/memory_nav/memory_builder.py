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

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    特征提取器基类
    
    可以被替换为 LongCLIP、CLIP、DinoV2 等不同的特征提取器
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


class LongCLIPExtractor(FeatureExtractor):
    """
    LongCLIP 特征提取器
    
    使用 LongCLIP 模型提取视觉特征
    """
    
    def __init__(self, model_path: str = None, feature_dim: int = 768, device: str = "cuda:0"):
        self.model_path = model_path
        super().__init__(feature_dim, device)
    
    def _load_model(self):
        """加载 LongCLIP 模型"""
        try:
            import torch
            import sys
            
            # 尝试导入 LongCLIP
            longclip_path = "/home/ubuntu/Disk/codes/jianxiong/MemoryNav/internnav/model/basemodel/LongCLIP"
            if os.path.exists(longclip_path):
                sys.path.insert(0, longclip_path)
            
            import sys; sys.path.insert(0, "/home/ubuntu/Disk/codes/jianxiong/MemoryNav"); from internnav.model.basemodel.LongCLIP.model import longclip
            
            # 默认模型路径
            if self.model_path is None:
                self.model_path = "/home/ubuntu/Disk/codes/jianxiong/MemoryNav/checkpoints/longclip-B.pt"
            
            if os.path.exists(self.model_path):
                self.model, self.preprocess = longclip.load(self.model_path, device=self.device)
                self.model.eval()
                logger.info(f"[LongCLIP] 模型加载成功: {self.model_path}")
            else:
                logger.warning(f"[LongCLIP] 模型文件不存在: {self.model_path}，使用随机特征")
                self.model = None
                
        except Exception as e:
            logger.warning(f"[LongCLIP] 模型加载失败: {e}，使用随机特征")
            self.model = None
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取 LongCLIP 特征"""
        if self.model is None:
            return super().extract(image)
        
        try:
            import torch
            from PIL import Image as PILImage
            
            # 转换为 PIL Image
            if image.shape[2] == 3:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = PILImage.fromarray(image)
            
            # 预处理
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            logger.warning(f"[LongCLIP] 特征提取失败: {e}")
            return super().extract(image)


class MemoryBuilder:
    """
    记忆构建器
    
    从标注数据目录构建完整的记忆拓扑图
    """
    
    def __init__(self, 
                 feature_extractor: FeatureExtractor = None,
                 feature_dim: int = 768,
                 device: str = "cuda:0"):
        """
        初始化构建器
        
        Args:
            feature_extractor: 特征提取器，None则使用默认的LongCLIP
            feature_dim: 特征维度
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.device = device
        
        # 特征提取器
        if feature_extractor is None:
            self.extractor = LongCLIPExtractor(feature_dim=feature_dim, device=device)
        else:
            self.extractor = feature_extractor
        
        # 记忆图和VPR
        self.graph = MemoryGraph()
        self.vpr = MemoryVPR(feature_dim=feature_dim)
        
        logger.info(f"[MemoryBuilder] 初始化完成: dim={feature_dim}, device={device}")
    
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
        
        # 构建边
        edges = []
        for next_pos in next_positions:
            pixel_pos = next_pos.get('pixel_position', '0.5,0.5')
            if isinstance(pixel_pos, str):
                x, y = map(float, pixel_pos.split(','))
            else:
                x, y = float(pixel_pos[0]), float(pixel_pos[1])
            
            stitch_image = next_pos.get('stitch_image', '')
            stitch_path = str(node_dir / stitch_image) if stitch_image else ""
            
            edge = MemoryEdge(
                target_node_id=str(next_pos.get('position_id', '')),
                target_node_name=next_pos.get('position_name', ''),
                target_node_name_eng=next_pos.get('position_name_eng', ''),
                angle=float(next_pos.get('angle', 0)),
                pixel_position=(x, y),
                stitch_image_path=stitch_path
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
            
            logger.info(f"[MemoryBuilder] 记忆数据已从 {path} 加载")
        else:
            logger.warning(f"[MemoryBuilder] 文件不存在: {graph_path}")
        
        return self.graph, self.vpr
    
    def get_stats(self) -> Dict:
        """获取构建器统计信息"""
        return {
            'graph': self.graph.get_stats(),
            'vpr': self.vpr.get_stats()
        }
