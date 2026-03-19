#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 兜底打点模块

利用 DINOv3 密集 patch 特征在相机图中定位 crop 参考图对应区域的中心点。
比 Qwen3.5 快 20 倍以上（~50ms vs ~1.5s），作为第一优先级兜底方案。

原理:
  1. 提取 crop 参考图的 patch 特征 → 取平均得到 "目标原型" 向量
  2. 提取相机图的 patch 特征网格
  3. 计算每个 patch 与目标原型的余弦相似度 → 热力图
  4. 在热力图上找峰值区域 → 中心点即为打点结果
  5. 峰值相似度作为置信度
"""

import logging
import time
import os
from typing import Dict, Optional, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DINOv3PointGrounder:
    """
    基于 DINOv3 patch 特征的兜底打点器

    与 Qwen3.5PointGrounder 接口一致:
    - predict(image, landmark_name, crop_paths) → result dict
    - predict_on_camera(camera_images, landmark_name, crop_paths, target_camera) → result dict
    """

    def __init__(self, device: str = "cuda:0", confidence_threshold: float = 0.45):
        """
        Args:
            device: 推理设备
            confidence_threshold: 打点置信度阈值 (低于此值认为未找到)
        """
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._patch_size = 16
        self._feature_dim = 768
        self._confidence_threshold = confidence_threshold
        self._crop_cache: Dict[str, torch.Tensor] = {}  # path -> prototype feature

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def start(self):
        """加载 DINOv3 模型"""
        if self._model is not None:
            return
        import timm
        model_name = "vit_base_patch16_dinov3"
        logger.info(f"[DINOv3Grounder] 加载 {model_name} ...")
        t0 = time.time()
        self._model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self._model = self._model.eval().to(self._device)

        if hasattr(self._model, 'patch_embed'):
            ps = getattr(self._model.patch_embed, 'patch_size', None)
            if ps is not None:
                self._patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        if hasattr(self._model, 'embed_dim'):
            self._feature_dim = self._model.embed_dim

        logger.info(f"[DINOv3Grounder] 加载完成: {time.time()-t0:.1f}s, "
                    f"patch_size={self._patch_size}, dim={self._feature_dim}")

    def share_model(self, model, patch_size: int = 16, feature_dim: int = 768):
        """共享已加载的 DINOv3 模型 (避免重复加载, 从 SubImageMatcher 共享)"""
        self._model = model
        self._patch_size = patch_size
        self._feature_dim = feature_dim
        logger.info(f"[DINOv3Grounder] 共享模型: patch_size={patch_size}, dim={feature_dim}")

    def _prepare_image(self, image: np.ndarray, target_size: int = 518):
        """预处理: BGR→RGB→resize→normalize→tensor"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        scale = target_size / max(h, w)
        new_h = max(self._patch_size, (int(h * scale) // self._patch_size) * self._patch_size)
        new_w = max(self._patch_size, (int(w * scale) // self._patch_size) * self._patch_size)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        return tensor, new_h, new_w

    def _extract_patch_features(self, image: np.ndarray, target_size: int = 518):
        """提取 patch 特征网格

        Returns:
            (features [n_h, n_w, dim], n_h, n_w)
        """
        tensor, rh, rw = self._prepare_image(image, target_size)
        n_h = rh // self._patch_size
        n_w = rw // self._patch_size

        with torch.no_grad():
            feats = self._model.forward_features(tensor)
            n_patches = n_h * n_w
            n_prefix = feats.shape[1] - n_patches
            patch_tokens = feats[:, max(0, n_prefix):, :]

        grid = patch_tokens[0].reshape(n_h, n_w, -1)
        grid = F.normalize(grid, dim=-1)
        return grid, n_h, n_w

    def _get_crop_prototype(self, crop_image: np.ndarray, crop_path: str = "") -> torch.Tensor:
        """提取 crop 参考图的原型特征 (带缓存)

        原型 = crop 所有 patch 特征的 L2 归一化平均
        """
        if crop_path and crop_path in self._crop_cache:
            return self._crop_cache[crop_path]

        # crop 图较小，用较小 target_size
        crop_h, crop_w = crop_image.shape[:2]
        target = max(self._patch_size * 2, min(280, max(crop_h, crop_w)))
        grid, _, _ = self._extract_patch_features(crop_image, target_size=target)

        # 平均池化 → 原型向量
        prototype = grid.reshape(-1, self._feature_dim).mean(dim=0)
        prototype = F.normalize(prototype, dim=0)

        if crop_path:
            self._crop_cache[crop_path] = prototype
        return prototype

    def _find_peak_point(self, sim_map: np.ndarray, n_h: int, n_w: int,
                         orig_h: int, orig_w: int, top_k: int = 5):
        """从相似度热力图中找峰值点 → 返回归一化坐标

        使用 top-k patch 的加权平均位置 (比单点 argmax 更鲁棒)
        """
        flat = sim_map.flatten()
        k = min(top_k, len(flat))
        top_indices = np.argpartition(flat, -k)[-k:]
        top_sims = flat[top_indices]

        # softmax 权重 (temperature=0.1 → sharp)
        top_sims_shifted = top_sims - top_sims.max()
        weights = np.exp(top_sims_shifted * 10)
        weights /= weights.sum()

        # 加权平均位置
        ys = (top_indices // n_w).astype(float)
        xs = (top_indices % n_w).astype(float)

        center_patch_y = (weights * ys).sum() + 0.5  # patch 中心
        center_patch_x = (weights * xs).sum() + 0.5

        # patch → 归一化 [0, 1]
        point_y = center_patch_y / n_h
        point_x = center_patch_x / n_w

        peak_confidence = float(flat[top_indices[np.argmax(top_sims)]])

        return (round(point_x, 4), round(point_y, 4)), peak_confidence

    def predict(self, image: np.ndarray, landmark_name: str,
                crop_paths: Dict[str, str] = None) -> Dict:
        """
        对单张相机图执行 DINOv3 兜底打点

        Args:
            image: 相机图 (BGR)
            landmark_name: 地标名 (仅用于日志)
            crop_paths: {"big": path, "mid": path, "small": path}

        Returns:
            与 Qwen35PointGrounder.predict 格式一致
        """
        if not self.is_ready:
            return {"success": False, "point": None, "point_pixel": None,
                    "confidence": 0.0, "error": "DINOv3 模型未加载"}

        if not crop_paths:
            return {"success": False, "point": None, "point_pixel": None,
                    "confidence": 0.0, "error": "无 crop 参考图路径"}

        t0 = time.perf_counter()
        orig_h, orig_w = image.shape[:2]

        # 提取相机图 patch 特征 (只做一次)
        cam_grid, n_h, n_w = self._extract_patch_features(image, target_size=518)
        cam_flat = cam_grid.reshape(-1, self._feature_dim)  # [n_h*n_w, dim]

        # 对每个尺度的 crop 计算相似度，取最佳结果
        best_point = None
        best_conf = -1.0
        best_scale = None

        for scale in ["mid", "big", "small"]:
            crop_path = crop_paths.get(scale, "")
            if not crop_path or not os.path.exists(crop_path):
                continue

            crop_img = cv2.imread(crop_path)
            if crop_img is None:
                continue

            prototype = self._get_crop_prototype(crop_img, crop_path)

            # 余弦相似度热力图: [n_h*n_w] → [n_h, n_w]
            sim = (cam_flat @ prototype).cpu().numpy().reshape(n_h, n_w)

            point, peak_conf = self._find_peak_point(sim, n_h, n_w, orig_h, orig_w)

            if peak_conf > best_conf:
                best_conf = peak_conf
                best_point = point
                best_scale = scale

        latency = time.perf_counter() - t0

        if best_point is None or best_conf < self._confidence_threshold:
            return {
                "success": False,
                "point": None,
                "point_pixel": None,
                "confidence": round(best_conf, 4) if best_conf > 0 else 0.0,
                "latency": round(latency, 4),
                "error": f"DINOv3 置信度不足: {best_conf:.4f} < {self._confidence_threshold}",
                "method": "dinov3",
            }

        px = int(best_point[0] * orig_w)
        py = int(best_point[1] * orig_h)

        return {
            "success": True,
            "point": list(best_point),   # [x_norm, y_norm] in [0,1]
            "point_pixel": [px, py],
            "confidence": round(best_conf, 4),
            "latency": round(latency, 4),
            "scale": best_scale,
            "method": "dinov3",
        }

    def predict_on_camera(self, camera_images: Dict[str, np.ndarray],
                          landmark_name: str,
                          crop_paths: Dict[str, str] = None,
                          target_camera: str = None) -> Dict:
        """
        在多相机上执行 DINOv3 兜底打点

        Args:
            camera_images: {"camera_1": img, ...}
            landmark_name: 地标名
            crop_paths: {"big": path, "mid": path, "small": path}
            target_camera: 优先相机

        Returns:
            与 Qwen35PointGrounder.predict_on_camera 格式一致
        """
        if target_camera:
            cameras_to_try = [target_camera] + [c for c in sorted(camera_images.keys()) if c != target_camera]
        else:
            cameras_to_try = sorted(camera_images.keys())

        best_result = None
        best_confidence = -1.0
        tried_cameras = []

        for cam_name in cameras_to_try:
            if cam_name not in camera_images:
                continue
            tried_cameras.append(cam_name)
            result = self.predict(camera_images[cam_name], landmark_name, crop_paths)
            result["camera_name"] = cam_name

            if result["success"] and result["confidence"] > best_confidence:
                best_result = result
                best_confidence = result["confidence"]
                if best_confidence >= 0.5:
                    break

        if best_result:
            logger.info(f"[DINOv3Grounder] 打点成功: camera={best_result['camera_name']}, "
                       f"landmark='{landmark_name}', conf={best_confidence:.4f}, "
                       f"scale={best_result.get('scale')}, tried={tried_cameras}")
            return best_result

        logger.info(f"[DINOv3Grounder] 打点失败: landmark='{landmark_name}', tried={tried_cameras}")
        return {
            "success": False,
            "camera_name": cameras_to_try[0] if cameras_to_try else "",
            "point": None,
            "point_pixel": None,
            "confidence": 0.0,
            "error": f"All cameras failed for landmark '{landmark_name}' (DINOv3)",
            "method": "dinov3",
        }
