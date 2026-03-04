#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnyLoc VPR 特征提取器

基于 AnyLoc (RA-L 2023) 的视觉位置识别特征提取器。
使用 DINOv2 + VLAD 聚合生成全局描述子。

支持两种模式：
1. VLAD模式 (默认): DINOv2 patch features → VLAD聚合 → 全局描述子
2. GeM模式 (轻量): DINOv2 patch features → GeM池化 → 全局描述子

参考:
- AnyLoc: Towards Universal Visual Place Recognition (RA-L 2023)
- https://github.com/AnyLoc/AnyLoc
"""

import os
import logging
from typing import Dict, List, Optional, Literal, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision import transforms as tvf
    import einops as ein
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("[AnyLoc] PyTorch/einops 不可用")

try:
    import fast_pytorch_kmeans as fpk
    FPK_AVAILABLE = True
except ImportError:
    FPK_AVAILABLE = False
    logger.warning("[AnyLoc] fast_pytorch_kmeans 不可用，VLAD词汇表需从缓存加载")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================
# DINOv2 特征提取器 (来自 AnyLoc demo/utilities.py)
# ============================================================
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14",
                           "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2ExtractFeatures:
    """从 DINOv2 中间层提取 patch 特征"""

    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int,
                 facet: _DINO_FACETS = "token", use_cls=False,
                 norm_descs=True, device: str = "cpu") -> None:
        self.vit_type = dino_model
        # 优先使用本地缓存，避免网络问题
        try:
            self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', dino_model)
        except Exception:
            # 尝试从本地缓存加载
            import glob
            cache_dirs = glob.glob(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_*'))
            if cache_dirs:
                self.dino_model: nn.Module = torch.hub.load(
                    cache_dirs[0], dino_model, source='local')
            else:
                raise
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                register_forward_hook(self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer]. \
                attn.qkv.register_forward_hook(
                    self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook

    def __call__(self, img: 'torch.Tensor') -> 'torch.Tensor':
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None
        return res

    def __del__(self):
        if hasattr(self, 'fh_handle'):
            self.fh_handle.remove()


# ============================================================
# VLAD 聚合器 (来自 AnyLoc demo/utilities.py，精简版)
# ============================================================
class VLAD:
    """VLAD 全局描述子生成器"""

    def __init__(self, num_clusters: int,
                 desc_dim: Union[int, None] = None,
                 intra_norm: bool = True, norm_descs: bool = True,
                 dist_mode: str = "cosine", vlad_mode: str = "hard",
                 soft_temp: float = 1.0,
                 cache_dir: Union[str, None] = None) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ['soft', 'hard']
        self.soft_temp = soft_temp
        self.c_centers = None
        self.kmeans = None
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(self.cache_dir))
            os.makedirs(self.cache_dir, exist_ok=True)

    def can_use_cache_vlad(self):
        if self.cache_dir is None:
            return False
        return os.path.exists(f"{self.cache_dir}/c_centers.pt")

    def fit(self, train_descs: Union[np.ndarray, 'torch.Tensor', None]):
        """训练或加载词汇表"""
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        if self.can_use_cache_vlad():
            logger.info("[VLAD] 使用缓存的聚类中心")
            self.c_centers = torch.load(f"{self.cache_dir}/c_centers.pt",
                                         map_location='cpu')
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
        else:
            if train_descs is None:
                raise ValueError("无训练描述子且无缓存")
            if isinstance(train_descs, np.ndarray):
                train_descs = torch.from_numpy(train_descs).to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                torch.save(self.c_centers, f"{self.cache_dir}/c_centers.pt")
                logger.info(f"[VLAD] 聚类中心已缓存: {self.cache_dir}/c_centers.pt")

    def generate(self, query_descs: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """生成单张图的 VLAD 向量"""
        assert self.c_centers is not None
        if isinstance(query_descs, np.ndarray):
            query_descs = torch.from_numpy(query_descs).to(torch.float32)
        if self.norm_descs:
            query_descs = F.normalize(query_descs)
        # Residuals: [q, c, d]
        residuals = ein.rearrange(query_descs, "q d -> q 1 d") \
                    - ein.rearrange(self.c_centers, "c d -> 1 c d")

        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == 'hard':
            labels = self.kmeans.predict(query_descs)
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                cd_sum = residuals[labels == k, k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k * self.desc_dim:(k + 1) * self.desc_dim] = cd_sum
        else:
            cos_sims = F.cosine_similarity(
                ein.rearrange(query_descs, "q d -> q 1 d"),
                ein.rearrange(self.c_centers, "c d -> 1 c d"),
                dim=2)
            soft_assign = F.softmax(self.soft_temp * cos_sims, dim=1)
            for k in range(self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                cd_sum = ein.rearrange(w * residuals,
                                       "q c d -> (q c) d").sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k * self.desc_dim:(k + 1) * self.desc_dim] = cd_sum
        return F.normalize(un_vlad, dim=0)

    def generate_multi(self, multi_query) -> 'torch.Tensor':
        res = [self.generate(q) for q in multi_query]
        return torch.stack(res)


# ============================================================
# AnyLoc 特征提取器 (集成到 MemoryNav)
# ============================================================
class AnyLocExtractor:
    """
    AnyLoc VPR 特征提取器

    将 DINOv2 + VLAD/GeM 封装为与 LongCLIPExtractor 兼容的接口。

    Args:
        dino_model: DINOv2模型名称
        desc_layer: 提取特征的层号
        desc_facet: 特征facet类型
        agg_mode: 聚合模式 'vlad' 或 'gem'
        num_clusters: VLAD聚类数 (仅vlad模式)
        domain: 预训练域 (仅vlad模式，用于加载预训练词汇表)
        vlad_cache_dir: VLAD缓存目录
        max_img_size: 最大图像边长
        device: 计算设备
    """

    # 不同模型对应的 desc_layer 和 desc_dim
    MODEL_CONFIGS = {
        'dinov2_vits14': {'desc_layer': 11, 'desc_dim': 384},
        'dinov2_vitb14': {'desc_layer': 11, 'desc_dim': 768},
        'dinov2_vitl14': {'desc_layer': 23, 'desc_dim': 1024},
        'dinov2_vitg14': {'desc_layer': 31, 'desc_dim': 1536},
    }

    def __init__(self,
                 dino_model: str = "dinov2_vitb14",
                 desc_layer: int = None,
                 desc_facet: str = "value",
                 agg_mode: str = "vlad",
                 num_clusters: int = 8,
                 domain: str = "indoor",
                 vlad_cache_dir: str = None,
                 max_img_size: int = 630,
                 device: str = "cuda:0"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用，无法使用 AnyLoc")

        self.dino_model_name = dino_model
        self.agg_mode = agg_mode
        self.num_clusters = num_clusters
        self.domain = domain
        self.max_img_size = max_img_size
        self.device = device

        # 自动设置层号和维度
        config = self.MODEL_CONFIGS.get(dino_model, self.MODEL_CONFIGS['dinov2_vitb14'])
        self.desc_layer = desc_layer or config['desc_layer']
        self.desc_dim = config['desc_dim']

        # 计算输出特征维度
        if agg_mode == 'vlad':
            self.feature_dim = num_clusters * self.desc_dim
        else:  # gem
            self.feature_dim = self.desc_dim

        # VLAD缓存目录
        if vlad_cache_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            self.vlad_cache_dir = os.path.join(base, 'anyloc_cache',
                                                f'{dino_model}_l{self.desc_layer}_{desc_facet}_c{num_clusters}',
                                                domain)
        else:
            self.vlad_cache_dir = vlad_cache_dir
        os.makedirs(self.vlad_cache_dir, exist_ok=True)

        # 图像预处理
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])

        # 加载模型
        self.dino_extractor = None
        self.vlad = None
        self._vlad_fitted = False
        self._patch_descs_buffer = []  # 用于收集训练描述子
        self._load_model(desc_facet)

        logger.info(f"[AnyLoc] 初始化完成: model={dino_model}, layer={self.desc_layer}, "
                    f"facet={desc_facet}, agg={agg_mode}, dim={self.feature_dim}, "
                    f"device={device}")

    def _load_model(self, desc_facet: str):
        """加载 DINOv2 和 VLAD"""
        try:
            self.dino_extractor = DinoV2ExtractFeatures(
                self.dino_model_name, self.desc_layer,
                desc_facet, device=self.device)
            logger.info(f"[AnyLoc] DINOv2 模型加载成功: {self.dino_model_name}")
        except Exception as e:
            logger.error(f"[AnyLoc] DINOv2 加载失败: {e}")
            raise

        if self.agg_mode == 'vlad':
            self.vlad = VLAD(self.num_clusters, desc_dim=None,
                             cache_dir=self.vlad_cache_dir)
            # 尝试加载缓存的词汇表
            if self.vlad.can_use_cache_vlad():
                self.vlad.fit(None)
                self._vlad_fitted = True
                logger.info("[AnyLoc] VLAD 词汇表从缓存加载成功")
            else:
                logger.info("[AnyLoc] VLAD 词汇表未找到，需要先调用 fit_vlad_vocabulary()")

    def _preprocess_image(self, image: np.ndarray) -> 'torch.Tensor':
        """预处理图像为 DINOv2 输入"""
        # BGR → RGB → PIL
        if CV2_AVAILABLE and len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        pil_img = PILImage.fromarray(rgb)

        img_pt = self.base_tf(pil_img).to(self.device)

        # 限制最大尺寸
        c, h, w = img_pt.shape
        if max(h, w) > self.max_img_size:
            if h >= w:
                w_new = int(w * self.max_img_size / h)
                h_new = self.max_img_size
            else:
                h_new = int(h * self.max_img_size / w)
                w_new = self.max_img_size
            img_pt = tvf.functional.resize(img_pt, (h_new, w_new),
                                            interpolation=tvf.InterpolationMode.BICUBIC)
            c, h, w = img_pt.shape

        # 使尺寸可被14整除 (patch size)
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]

        return img_pt

    def _extract_patch_descriptors(self, image: np.ndarray) -> 'torch.Tensor':
        """提取 DINOv2 patch 描述子"""
        img_pt = self._preprocess_image(image)
        with torch.no_grad():
            ret = self.dino_extractor(img_pt)  # [1, num_patches, desc_dim]
        return ret.cpu().squeeze(0)  # [num_patches, desc_dim]

    def _gem_pool(self, descs: 'torch.Tensor', p: float = 3.0) -> np.ndarray:
        """GeM (Generalized Mean) 池化"""
        # descs: [num_patches, desc_dim]
        gem = (descs.clamp(min=1e-6) ** p).mean(dim=0) ** (1.0 / p)
        gem = F.normalize(gem, dim=0)
        return gem.numpy().astype(np.float32)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像的全局描述子

        Args:
            image: BGR 图像 (numpy array)

        Returns:
            全局描述子 (feature_dim,)
        """
        if self.dino_extractor is None:
            return np.random.randn(self.feature_dim).astype(np.float32)

        try:
            descs = self._extract_patch_descriptors(image)

            if self.agg_mode == 'gem':
                return self._gem_pool(descs)
            else:  # vlad
                if not self._vlad_fitted:
                    # 词汇表未训练，暂时用 GeM
                    logger.warning("[AnyLoc] VLAD未训练，暂用GeM池化")
                    return self._gem_pool(descs)
                gd = self.vlad.generate(descs)
                return gd.numpy().astype(np.float32)

        except Exception as e:
            logger.error(f"[AnyLoc] 特征提取失败: {e}")
            return np.random.randn(self.feature_dim).astype(np.float32)

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        features = [self.extract(img) for img in images]
        return np.array(features)

    def collect_descriptors(self, image: np.ndarray):
        """
        收集 patch 描述子用于训练 VLAD 词汇表

        在构建记忆库时，对每张图调用此方法收集描述子，
        然后调用 fit_vlad_vocabulary() 训练词汇表。
        """
        if self.agg_mode != 'vlad':
            return
        try:
            descs = self._extract_patch_descriptors(image)
            self._patch_descs_buffer.append(descs)
        except Exception as e:
            logger.warning(f"[AnyLoc] 描述子收集失败: {e}")

    def fit_vlad_vocabulary(self):
        """
        使用收集的描述子训练 VLAD 词汇表

        应在 collect_descriptors() 收集完所有图像后调用。
        训练后的词汇表会自动缓存到 vlad_cache_dir。
        """
        if self.agg_mode != 'vlad':
            logger.info("[AnyLoc] GeM模式无需训练词汇表")
            return

        if not self._patch_descs_buffer:
            logger.warning("[AnyLoc] 无描述子可用于训练词汇表")
            return

        logger.info(f"[AnyLoc] 开始训练VLAD词汇表: "
                    f"{len(self._patch_descs_buffer)} 张图的描述子")

        all_descs = torch.cat(self._patch_descs_buffer, dim=0)
        logger.info(f"[AnyLoc] 总描述子数: {all_descs.shape[0]}, 维度: {all_descs.shape[1]}")

        self.vlad.fit(all_descs)
        self._vlad_fitted = True
        self._patch_descs_buffer.clear()

        logger.info("[AnyLoc] VLAD 词汇表训练完成并已缓存")

    def is_vlad_ready(self) -> bool:
        """检查 VLAD 是否已就绪"""
        if self.agg_mode == 'gem':
            return True
        return self._vlad_fitted

    @property
    def output_dim(self) -> int:
        """输出特征维度"""
        return self.feature_dim
