#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试场景描述生成器 v3.0 vs v2.x 对比

目的：
1. 对比新旧版本的生成质量
2. 评估节点区分度的改进
3. 验证优化效果

测试环境: internvla conda环境
"""

import os
import sys
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
    """评估生成结果的区分度"""
    n = len(results)
    if n < 2:
        return {"error": "样本数不足"}

    desc_similarities = []
    label_overlaps = []
    name_similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            desc_sim = calculate_similarity(
                results[i].get('scene_description', ''),
                results[j].get('scene_description', '')
            )
            desc_similarities.append(desc_sim)

            label_overlap = calculate_label_overlap(
                results[i].get('semantic_labels', []),
                results[j].get('semantic_labels', [])
            )
            label_overlaps.append(label_overlap)

            name_sim = calculate_similarity(
                results[i].get('node_name', ''),
                results[j].get('node_name', '')
            )
            name_similarities.append(name_sim)

    return {
        "description_similarity": {
            "mean": float(np.mean(desc_similarities)),
            "max": float(np.max(desc_similarities)),
            "min": float(np.min(desc_similarities)),
            "std": float(np.std(desc_similarities))
        },
        "label_overlap": {
            "mean": float(np.mean(label_overlaps)),
            "max": float(np.max(label_overlaps)),
            "min": float(np.min(label_overlaps)),
            "std": float(np.std(label_overlaps))
        },
        "name_similarity": {
            "mean": float(np.mean(name_similarities)),
            "max": float(np.max(name_similarities)),
            "min": float(np.min(name_similarities)),
            "std": float(np.std(name_similarities))
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
        "label_frequency": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:15])
    }


def test_generator(generator, samples: list, version_name: str):
    """测试生成器"""
    print(f"\n{'='*60}")
    print(f"测试 {version_name}")
    print('='*60)

    results = []
    for i, sample in enumerate(samples):
        logger.info(f"处理样本 {i+1}/{len(samples)}: timestamp={sample['timestamp']}")

        images = sample['camera_images']
        scene_desc, labels, name = generator.generate_complete_scene_info(images)

        result = {
            'timestamp': sample['timestamp'],
            'scene_description': scene_desc,
            'semantic_labels': labels,
            'node_name': name
        }
        results.append(result)

        print(f"\n样本 {i+1}:")
        print(f"  节点名称: {name}")
        print(f"  语义标签: {labels}")
        print(f"  场景描述: {scene_desc[:100] if scene_desc else 'None'}...")

    return results


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("场景描述生成器 v3.0 vs v2.x 对比测试")
    print("=" * 70)

    from deploy.memory_modules.config import MemoryNavigationConfig

    config = MemoryNavigationConfig(
        memory_enabled=True,
        vlm_enabled=True,
        gpu_id=None,
        vlm_device="cuda:2"
    )

    # 加载测试数据
    all_samples = []
    for data_dir in ["memory_test_data", "memory_test_data2"]:
        samples = load_test_images(data_dir, max_samples=5)
        all_samples.extend(samples)

    if not all_samples:
        logger.error("没有找到测试数据")
        return 1

    print(f"\n总共加载 {len(all_samples)} 个测试样本")

    # 测试 v3.0 版本
    print("\n" + "=" * 70)
    print("测试 v3.0 版本 (新版)")
    print("=" * 70)

    from deploy.memory_modules.scene_description_v3 import SceneDescriptionGeneratorV3
    generator_v3 = SceneDescriptionGeneratorV3(config)

    if not generator_v3.is_available:
        logger.error("v3 VLM模型不可用")
        return 1

    results_v3 = test_generator(generator_v3, all_samples, "v3.0")

    # 评估 v3.0
    print("\n" + "-" * 40)
    print("v3.0 评估结果")
    print("-" * 40)

    metrics_v3 = evaluate_discrimination(results_v3)
    unique_v3 = count_unique_elements(results_v3)

    print(f"场景描述相似度: mean={metrics_v3['description_similarity']['mean']:.3f}, "
          f"max={metrics_v3['description_similarity']['max']:.3f}")
    print(f"标签重叠率: mean={metrics_v3['label_overlap']['mean']:.3f}, "
          f"max={metrics_v3['label_overlap']['max']:.3f}")
    print(f"节点名称相似度: mean={metrics_v3['name_similarity']['mean']:.3f}, "
          f"max={metrics_v3['name_similarity']['max']:.3f}")
    print(f"唯一节点名称数: {metrics_v3['unique_names']}/{metrics_v3['total_samples']}")
    print(f"唯一场景描述数: {unique_v3['unique_descriptions']}")
    print(f"唯一标签数: {unique_v3['unique_labels']}")

    print("\n高频标签:")
    for label, count in list(unique_v3['label_frequency'].items())[:10]:
        print(f"  {label}: {count}")

    # 保存结果
    output = {
        "v3_results": results_v3,
        "v3_metrics": metrics_v3,
        "v3_unique_stats": unique_v3
    }

    output_file = project_root / "scripts" / "scene_description_v3_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 判断改进效果
    print("\n" + "=" * 70)
    print("改进效果评估")
    print("=" * 70)

    # 读取v2结果
    v2_results_file = project_root / "scripts" / "scene_description_test_results.json"
    if v2_results_file.exists():
        with open(v2_results_file, 'r', encoding='utf-8') as f:
            v2_data = json.load(f)
            metrics_v2 = v2_data.get('discrimination_metrics', {})

        print("\n对比 v2.x vs v3.0:")
        print("-" * 40)

        comparisons = [
            ("场景描述相似度",
             metrics_v2.get('description_similarity', {}).get('mean', 0),
             metrics_v3['description_similarity']['mean'],
             "越低越好"),
            ("标签重叠率",
             metrics_v2.get('label_overlap', {}).get('mean', 0),
             metrics_v3['label_overlap']['mean'],
             "越低越好"),
            ("节点名称相似度",
             metrics_v2.get('name_similarity', {}).get('mean', 0),
             metrics_v3['name_similarity']['mean'],
             "越低越好"),
            ("唯一名称比例",
             metrics_v2.get('unique_names', 0) / max(metrics_v2.get('total_samples', 1), 1),
             metrics_v3['unique_names'] / metrics_v3['total_samples'],
             "越高越好"),
        ]

        improved = 0
        for name, v2_val, v3_val, target in comparisons:
            if target == "越低越好":
                delta = v2_val - v3_val
                improved_flag = "✓ 改进" if delta > 0.05 else ("○ 持平" if abs(delta) < 0.05 else "✗ 退步")
            else:
                delta = v3_val - v2_val
                improved_flag = "✓ 改进" if delta > 0.05 else ("○ 持平" if abs(delta) < 0.05 else "✗ 退步")

            print(f"{name}: v2={v2_val:.3f} -> v3={v3_val:.3f} [{improved_flag}]")

            if "改进" in improved_flag:
                improved += 1

        print(f"\n改进指标数: {improved}/4")

        if improved >= 2:
            print("\n>>> v3.0 版本改进效果显著 <<<")
        else:
            print("\n>>> v3.0 版本需要进一步优化 <<<")

    return 0


if __name__ == "__main__":
    sys.exit(main())
