#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试场景描述生成质量

目的：
1. 分析当前VLM生成的场景描述、语义标签和节点名称
2. 评估不同节点之间的区分度
3. 使用真实图像测试生成效果

基于以下最佳实践优化:
1. SENT-Map: 语义增强拓扑图，使用结构化JSON描述
2. ROOT: 分层语义场景理解
3. Visual Landmark Sequence: 单对象/多对象/场景地标分类

测试环境: internvla conda环境
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

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
    """
    加载测试图像

    Args:
        data_dir: 测试数据目录
        max_samples: 最大样本数

    Returns:
        list of dict: 每个样本包含 {timestamp, camera_images}
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return []

    # 收集所有时间戳
    all_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    timestamps = set()

    for f in all_files:
        # 解析文件名: timestamp_camera_x.jpg
        parts = f.stem.split('_')
        if len(parts) >= 2:
            try:
                ts = parts[0]
                timestamps.add(ts)
            except:
                pass

    # 为每个时间戳收集相机图像
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

    logger.info(f"加载了 {len(samples)} 个测试样本")
    return samples


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def calculate_label_overlap(labels1: list, labels2: list) -> float:
    """计算两个标签列表的重叠率"""
    if not labels1 or not labels2:
        return 0.0
    set1 = set(l.lower() for l in labels1)
    set2 = set(l.lower() for l in labels2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def evaluate_discrimination(results: list) -> dict:
    """
    评估生成结果的区分度

    Args:
        results: 生成结果列表，每个包含 {scene_description, semantic_labels, node_name}

    Returns:
        区分度评估指标
    """
    n = len(results)
    if n < 2:
        return {"error": "样本数不足"}

    # 计算成对相似度
    desc_similarities = []
    label_overlaps = []
    name_similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            # 场景描述相似度
            desc_sim = calculate_similarity(
                results[i].get('scene_description', ''),
                results[j].get('scene_description', '')
            )
            desc_similarities.append(desc_sim)

            # 标签重叠率
            label_overlap = calculate_label_overlap(
                results[i].get('semantic_labels', []),
                results[j].get('semantic_labels', [])
            )
            label_overlaps.append(label_overlap)

            # 节点名称相似度
            name_sim = calculate_similarity(
                results[i].get('node_name', ''),
                results[j].get('node_name', '')
            )
            name_similarities.append(name_sim)

    # 计算统计指标
    return {
        "description_similarity": {
            "mean": np.mean(desc_similarities),
            "max": np.max(desc_similarities),
            "min": np.min(desc_similarities),
            "std": np.std(desc_similarities)
        },
        "label_overlap": {
            "mean": np.mean(label_overlaps),
            "max": np.max(label_overlaps),
            "min": np.min(label_overlaps),
            "std": np.std(label_overlaps)
        },
        "name_similarity": {
            "mean": np.mean(name_similarities),
            "max": np.max(name_similarities),
            "min": np.min(name_similarities),
            "std": np.std(name_similarities)
        },
        "unique_names": len(set(r.get('node_name', '') for r in results)),
        "total_samples": n
    }


def count_unique_elements(results: list) -> dict:
    """统计唯一元素数量"""
    unique_descriptions = set()
    unique_names = set()
    all_labels = []

    for r in results:
        if r.get('scene_description'):
            unique_descriptions.add(r['scene_description'])
        if r.get('node_name'):
            unique_names.add(r['node_name'])
        if r.get('semantic_labels'):
            all_labels.extend(r['semantic_labels'])

    label_counts = defaultdict(int)
    for label in all_labels:
        label_counts[label.lower()] += 1

    return {
        "unique_descriptions": len(unique_descriptions),
        "unique_names": len(unique_names),
        "unique_labels": len(label_counts),
        "label_frequency": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:20])
    }


def test_current_generator(samples: list, config):
    """
    测试当前的场景描述生成器
    """
    from deploy.memory_modules.scene_description import SceneDescriptionGenerator

    generator = SceneDescriptionGenerator(config)

    if not generator.is_available:
        logger.error("VLM模型不可用")
        return []

    results = []
    for i, sample in enumerate(samples):
        logger.info(f"\n处理样本 {i+1}/{len(samples)}: timestamp={sample['timestamp']}")

        images = sample['camera_images']
        scene_desc, labels, name = generator.generate_complete_scene_info(images)

        result = {
            'timestamp': sample['timestamp'],
            'scene_description': scene_desc,
            'semantic_labels': labels,
            'node_name': name
        }
        results.append(result)

        logger.info(f"  场景描述: {scene_desc[:100] if scene_desc else 'None'}...")
        logger.info(f"  语义标签: {labels}")
        logger.info(f"  节点名称: {name}")

    return results


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("场景描述生成质量测试")
    print("=" * 70)

    # 配置
    from deploy.memory_modules.config import MemoryNavigationConfig

    config = MemoryNavigationConfig(
        memory_enabled=True,
        vlm_enabled=True,
        gpu_id=None,  # 使用多GPU
        vlm_device="cuda:2"  # VLM使用GPU 2
    )

    # 加载测试数据
    data_dirs = [
        "memory_test_data",
        "memory_test_data2",
        "assets/realworld_sample_data1",
        "assets/realworld_sample_data2"
    ]

    all_samples = []
    for data_dir in data_dirs:
        samples = load_test_images(data_dir, max_samples=5)
        all_samples.extend(samples)

    if not all_samples:
        logger.error("没有找到测试数据")
        return 1

    logger.info(f"总共加载 {len(all_samples)} 个测试样本")

    # 测试当前生成器
    print("\n[1] 测试当前场景描述生成器...")
    results = test_current_generator(all_samples, config)

    if not results:
        logger.error("没有生成结果")
        return 1

    # 评估区分度
    print("\n[2] 评估生成结果区分度...")
    discrimination_metrics = evaluate_discrimination(results)

    print("\n--- 区分度评估 ---")
    print(f"场景描述相似度: mean={discrimination_metrics['description_similarity']['mean']:.3f}, "
          f"max={discrimination_metrics['description_similarity']['max']:.3f}")
    print(f"标签重叠率: mean={discrimination_metrics['label_overlap']['mean']:.3f}, "
          f"max={discrimination_metrics['label_overlap']['max']:.3f}")
    print(f"节点名称相似度: mean={discrimination_metrics['name_similarity']['mean']:.3f}, "
          f"max={discrimination_metrics['name_similarity']['max']:.3f}")
    print(f"唯一节点名称数: {discrimination_metrics['unique_names']}/{discrimination_metrics['total_samples']}")

    # 统计唯一元素
    unique_stats = count_unique_elements(results)
    print("\n--- 唯一性统计 ---")
    print(f"唯一场景描述数: {unique_stats['unique_descriptions']}")
    print(f"唯一节点名称数: {unique_stats['unique_names']}")
    print(f"唯一标签数: {unique_stats['unique_labels']}")
    print(f"\n高频标签 (前10):")
    for label, count in list(unique_stats['label_frequency'].items())[:10]:
        print(f"  {label}: {count}")

    # 保存结果
    output_file = project_root / "scripts" / "scene_description_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "discrimination_metrics": {
                k: {kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in discrimination_metrics.items()
            },
            "unique_stats": unique_stats
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 判断是否需要优化
    print("\n[3] 优化建议...")
    needs_optimization = False

    if discrimination_metrics['description_similarity']['mean'] > 0.5:
        print("  ! 场景描述相似度过高，需要增加描述多样性")
        needs_optimization = True

    if discrimination_metrics['label_overlap']['mean'] > 0.5:
        print("  ! 标签重叠率过高，需要提取更具体的特征")
        needs_optimization = True

    if discrimination_metrics['name_similarity']['mean'] > 0.5:
        print("  ! 节点名称相似度过高，需要改进命名策略")
        needs_optimization = True

    if discrimination_metrics['unique_names'] < len(results) * 0.7:
        print(f"  ! 唯一名称比例过低 ({discrimination_metrics['unique_names']}/{len(results)})")
        needs_optimization = True

    if needs_optimization:
        print("\n>>> 建议优化提示词以提高区分度 <<<")
    else:
        print("\n>>> 当前生成质量良好 <<<")

    return 0 if not needs_optimization else 1


if __name__ == "__main__":
    sys.exit(main())
