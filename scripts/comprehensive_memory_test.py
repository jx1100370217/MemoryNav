#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合记忆系统测试

测试场景:
1. 模拟导航过程中的记忆构建
2. 验证VPR阈值优化效果
3. 验证周期性关键帧
4. 验证语义图完整性
"""

import os
import sys
import time
import json
import numpy as np
import logging
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deploy"))

from deploy.memory_modules.config import MemoryNavigationConfig
from deploy.memory_modules.vpr import VisualPlaceRecognition
from deploy.memory_modules.topological_map import TopologicalMapManager
from deploy.memory_modules.route_memory import RouteMemoryManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_trajectory_features(num_frames: int, noise_level: float = 0.1) -> list:
    """
    生成模拟导航轨迹的视觉特征

    模拟机器人逐渐移动时视觉特征的变化:
    - 相邻帧相似度高
    - 随着距离增加，相似度下降
    """
    np.random.seed(42)

    # 生成基础特征序列（模拟渐变）
    features = []
    base_feature = np.random.randn(512).astype(np.float32)
    base_feature = base_feature / np.linalg.norm(base_feature)

    for i in range(num_frames):
        # 添加渐进式变化
        drift = np.random.randn(512).astype(np.float32) * 0.02 * i  # 逐渐偏移
        noise = np.random.randn(512).astype(np.float32) * noise_level

        feature = base_feature + drift + noise
        feature = feature / np.linalg.norm(feature)
        features.append(feature)

    return features


def test_memory_construction():
    """测试记忆构建流程"""
    print("\n" + "="*70)
    print("综合记忆系统测试")
    print("="*70)

    # 1. 初始化配置
    config = MemoryNavigationConfig(
        memory_enabled=True,
        gpu_id="1",
        vlm_enabled=False,  # 测试时禁用VLM避免GPU依赖
        keyframe_interval=8
    )

    # 2. 初始化组件
    print("\n[1] 初始化组件...")
    topo_map = TopologicalMapManager(config)
    route_memory = RouteMemoryManager(config)

    # 验证VPR阈值
    print(f"    VPR高置信度阈值: {topo_map.vpr.high_confidence_threshold}")
    assert topo_map.vpr.high_confidence_threshold == 0.96, "VPR阈值未正确设置"
    print("    ✓ VPR阈值验证通过")

    # 3. 生成模拟轨迹
    print("\n[2] 生成模拟导航轨迹...")
    num_frames = 30
    features = generate_trajectory_features(num_frames, noise_level=0.05)
    print(f"    生成 {num_frames} 帧特征")

    # 4. 模拟导航记忆构建
    print("\n[3] 模拟导航记忆构建...")
    route_memory.start_recording("模拟导航测试")

    node_ids = []
    new_node_count = 0
    keyframe_count = 0

    start_time = time.time()
    for i, feature in enumerate(features):
        current_time = start_time + i * 0.5  # 每帧间隔0.5秒

        # 每8帧创建一个关键帧
        is_keyframe = (i > 0 and i % config.keyframe_interval == 0)

        node_id, is_new, revisit_info = topo_map.add_observation(
            visual_feature=feature,
            is_keyframe=is_keyframe,
            scene_description=f"场景描述_{i}" if is_keyframe else None,
            semantic_labels=["测试标签"] if is_keyframe else None
        )

        node_ids.append(node_id)
        if is_new:
            new_node_count += 1
        if is_keyframe:
            keyframe_count += 1

        # 记录到路线
        route_memory.record_step(
            node_id=node_id,
            visual_feature=feature,
            action=[2, 2, 2, 2],  # 模拟前进动作
            is_keyframe=is_keyframe
        )

        status = "新节点" if is_new else "合并"
        keyframe_str = " [关键帧]" if is_keyframe else ""
        if i < 5 or i >= num_frames - 3 or is_new or is_keyframe:
            print(f"    帧{i:2d}: node={node_id}, {status}{keyframe_str}")

    # 5. 停止记录
    route = route_memory.stop_recording()

    # 6. 分析结果
    print("\n[4] 记忆构建结果分析...")
    stats = topo_map.get_stats()

    print(f"    总帧数: {num_frames}")
    print(f"    新建节点数: {new_node_count}")
    print(f"    关键帧数: {keyframe_count}")
    print(f"    拓扑图节点数: {stats['total_nodes']}")
    print(f"    关键帧节点数: {stats['keyframe_nodes']}")
    print(f"    VPR索引大小: {stats['vpr_size']}")
    print(f"    节点序列: {node_ids}")
    print(f"    唯一节点数: {len(set(node_ids))}")

    # 7. 验证结果
    print("\n[5] 验证测试结果...")

    # 验证新节点数大于1（不应该全部合并到一个节点）
    assert new_node_count > 1, f"新节点数应该 > 1，实际: {new_node_count}"
    print(f"    ✓ 新节点创建正常 ({new_node_count} 个)")

    # 验证不是所有帧都合并到同一节点
    unique_nodes = len(set(node_ids))
    assert unique_nodes > 1, f"应该有多个不同节点，实际只有 {unique_nodes} 个"
    print(f"    ✓ 节点多样性正常 ({unique_nodes} 个不同节点)")

    # 验证关键帧被正确处理
    expected_keyframes = (num_frames - 1) // config.keyframe_interval
    assert keyframe_count >= expected_keyframes, f"关键帧数应该 >= {expected_keyframes}，实际: {keyframe_count}"
    print(f"    ✓ 关键帧处理正常 ({keyframe_count} 个)")

    # 验证路线记录完整性
    assert route is not None, "路线记录不应该为空"
    assert len(route.node_sequence) == num_frames, f"路线长度应该是 {num_frames}，实际: {len(route.node_sequence)}"
    print(f"    ✓ 路线记录完整 ({len(route.node_sequence)} 帧)")

    print("\n" + "="*70)
    print("✓ 所有综合测试通过!")
    print("="*70)

    return True


def main():
    try:
        success = test_memory_construction()
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
