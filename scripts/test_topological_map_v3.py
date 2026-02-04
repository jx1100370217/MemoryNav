#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试拓扑图构建 - 使用v3场景描述生成器

目的：
1. 测试v3场景描述生成器与拓扑图的集成
2. 验证生成的节点信息质量
3. 评估整体记忆系统的效果

测试环境: internvla conda环境
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deploy"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_images(data_dir: str, max_samples: int = 10):
    """加载测试图像"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    all_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    timestamps = set()

    for f in all_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            timestamps.add(parts[0])

    samples = []
    for ts in sorted(timestamps)[:max_samples]:
        camera_images = {}
        for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            img_path = data_path / f"{ts}_{cam_id}.jpg"
            if not img_path.exists():
                img_path = data_path / f"{ts}_{cam_id}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path))
                camera_images[cam_id] = img

        if camera_images:
            samples.append({
                'timestamp': ts,
                'camera_images': camera_images
            })

    return samples


def test_integrated_system():
    """测试集成系统"""
    print("\n" + "=" * 70)
    print("拓扑图构建集成测试 (v3场景描述生成器)")
    print("=" * 70)

    from deploy.memory_modules.config import MemoryNavigationConfig
    from deploy.memory_modules.topological_map import TopologicalMapManager
    from deploy.memory_modules.scene_description_v3 import SceneDescriptionGeneratorV3
    from deploy.memory_modules.feature_extraction import LongCLIPFeatureExtractor
    from deploy.memory_modules.surround_fusion import SurroundCameraFusion

    # 配置
    config = MemoryNavigationConfig(
        memory_enabled=True,
        vlm_enabled=True,
        gpu_id=None,
        vlm_device="cuda:2",
        feature_extractor_device="cuda:1",
        vlm_version="v3"
    )

    # 加载测试数据
    all_samples = []
    for data_dir in ["memory_test_data", "memory_test_data2"]:
        samples = load_test_images(data_dir, max_samples=5)
        all_samples.extend(samples)

    if not all_samples:
        logger.error("没有找到测试数据")
        return 1

    print(f"\n加载了 {len(all_samples)} 个测试样本")

    # 初始化组件
    print("\n[1] 初始化组件...")
    feature_extractor = LongCLIPFeatureExtractor(
        model_path=config.longclip_model_path,
        device=config.feature_extractor_device if config.feature_extractor_device.startswith("cuda") else f"cuda:{config.feature_extractor_device}",
        feature_dim=config.feature_dim
    )
    scene_generator = SceneDescriptionGeneratorV3(config)
    topo_map = TopologicalMapManager(config)
    surround_fusion = SurroundCameraFusion(config)

    print("  ✓ 组件初始化完成")

    # 模拟导航过程
    print("\n[2] 模拟导航过程...")

    results = []
    for i, sample in enumerate(all_samples):
        print(f"\n--- 处理样本 {i+1}/{len(all_samples)} ---")
        images = sample['camera_images']

        # 提取特征
        surround_features = {}
        for cam_id, img in images.items():
            # 确保是numpy数组
            if isinstance(img, Image.Image):
                img = np.array(img)
            feature = feature_extractor.extract_feature(img)
            surround_features[cam_id] = feature

        # 融合特征
        fused_feature = surround_fusion.fuse_features(surround_features)

        # 生成场景信息
        is_keyframe = (i % 3 == 0)  # 每3帧一个关键帧

        if is_keyframe:
            scene_desc, labels, node_name = scene_generator.generate_complete_scene_info(images)
        else:
            scene_desc = None
            labels = []
            node_name = None

        # 添加到拓扑图
        node_id, is_new, revisit_info = topo_map.add_observation(
            visual_feature=fused_feature,
            surround_images=images,
            is_keyframe=is_keyframe,
            scene_description=scene_desc,
            semantic_labels=labels,
            node_name=node_name
        )

        result = {
            'timestamp': sample['timestamp'],
            'node_id': node_id,
            'is_new': is_new,
            'is_keyframe': is_keyframe,
            'node_name': node_name,
            'scene_description': scene_desc,
            'semantic_labels': labels
        }
        results.append(result)

        status = "新节点" if is_new else f"合并到节点{revisit_info[0] if revisit_info else '?'}"
        keyframe_str = " [关键帧]" if is_keyframe else ""
        print(f"  样本{i+1}: node={node_id}, {status}{keyframe_str}")
        if node_name:
            print(f"    名称: {node_name}")
        if labels:
            print(f"    标签: {labels[:5]}")

    # 统计结果
    print("\n" + "=" * 70)
    print("[3] 拓扑图统计")
    print("=" * 70)

    stats = topo_map.get_stats()
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  关键帧节点数: {stats['keyframe_nodes']}")
    print(f"  VPR索引大小: {stats['vpr_size']}")

    # 获取所有节点信息
    unique_names = set()
    unique_descriptions = set()
    all_labels = []

    for node_id in topo_map.nodes:
        node = topo_map.nodes[node_id]
        if node.node_name:
            unique_names.add(node.node_name)
        if node.scene_description:
            unique_descriptions.add(node.scene_description)
        if node.semantic_labels:
            all_labels.extend(node.semantic_labels)

    print(f"\n唯一节点名称数: {len(unique_names)}")
    print(f"唯一场景描述数: {len(unique_descriptions)}")

    label_counts = defaultdict(int)
    for label in all_labels:
        label_counts[label.lower()] += 1

    print(f"唯一标签数: {len(label_counts)}")

    print("\n节点名称列表:")
    for name in sorted(unique_names):
        print(f"  - {name}")

    print("\n高频标签 (前10):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {label}: {count}")

    # 保存结果
    output = {
        "results": results,
        "stats": stats,
        "unique_names": list(unique_names),
        "unique_descriptions_count": len(unique_descriptions),
        "unique_labels_count": len(label_counts),
        "label_frequency": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:20])
    }

    output_file = project_root / "scripts" / "topological_map_v3_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存到: {output_file}")

    # 评估
    print("\n" + "=" * 70)
    print("[4] 质量评估")
    print("=" * 70)

    total_keyframes = sum(1 for r in results if r['is_keyframe'])
    names_with_unique = len(unique_names)

    quality_score = 0
    quality_checks = []

    # 检查1: 唯一名称比例
    if total_keyframes > 0:
        name_uniqueness = names_with_unique / total_keyframes
        if name_uniqueness >= 0.7:
            quality_checks.append(f"✓ 名称唯一性良好 ({names_with_unique}/{total_keyframes})")
            quality_score += 1
        else:
            quality_checks.append(f"✗ 名称唯一性需改进 ({names_with_unique}/{total_keyframes})")

    # 检查2: 标签多样性
    if len(label_counts) >= 15:
        quality_checks.append(f"✓ 标签多样性良好 ({len(label_counts)}种)")
        quality_score += 1
    else:
        quality_checks.append(f"○ 标签多样性一般 ({len(label_counts)}种)")

    # 检查3: 节点数量合理
    if stats['total_nodes'] > 1:
        quality_checks.append(f"✓ 节点数量合理 ({stats['total_nodes']}个)")
        quality_score += 1
    else:
        quality_checks.append(f"✗ 节点数量过少 ({stats['total_nodes']}个)")

    for check in quality_checks:
        print(f"  {check}")

    print(f"\n质量得分: {quality_score}/3")

    if quality_score >= 2:
        print("\n>>> v3集成测试通过 <<<")
        return 0
    else:
        print("\n>>> 需要进一步优化 <<<")
        return 1


if __name__ == "__main__":
    sys.exit(test_integrated_system())
