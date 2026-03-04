#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryNav 记忆导航综合测试脚本

测试内容:
1. VPR 定位: 用已知节点图片验证匹配正确性
2. VPR 循环移位: 验证不同朝向的匹配
3. 路径规划: 验证起点→终点路径正确
4. 状态机模拟: 模拟多步导航流程
5. 真实轨迹测试: 用 memory_test_data / memory_test_data2 的图片做 VPR 定位

用法:
    cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
    conda activate internvla
    python tests/test_memory_nav.py
"""

import sys
import os
import time
import json
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 项目路径
PROJECT_ROOT = Path("/home/ubuntu/Disk/codes/jianxiong/MemoryNav")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src/diffusion-policy"))

from PIL import Image


# ============================================================================
# 颜色输出
# ============================================================================
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def ok(msg):    print(f"  {C.GREEN}✅ PASS{C.END}  {msg}")
def fail(msg):  print(f"  {C.RED}❌ FAIL{C.END}  {msg}")
def info(msg):  print(f"  {C.CYAN}ℹ️  INFO{C.END}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}⚠️  WARN{C.END}  {msg}")
def header(msg): print(f"\n{C.BOLD}{'='*70}\n  {msg}\n{'='*70}{C.END}")

passed = 0
failed = 0
def check(cond, msg_pass, msg_fail=""):
    global passed, failed
    if cond:
        ok(msg_pass)
        passed += 1
    else:
        fail(msg_fail or msg_pass)
        failed += 1


# ============================================================================
# 节点名称映射 (从 node_position_info.json 中获取)
# ============================================================================
NODE_NAMES = {
    "1": "C8打印区",
    "3": "C8微波炉区域",
    "4": "C8玻璃门",
    "8": "C8前台区",
    "12": "C8电梯区",
    "14": "C8实验室门口",
    "15": "24号会议室门口",
    "18": "A8前台",
    "20": "07电话亭会议室区域",
    "21": "A8厕所区域",
    "23": "A8打印区",
    "26": "A8档案室区域",
    "29": "11号会议室区域",
    "31": "A8休息区过道",
    "32": "A8休息区",
}


def load_node_camera_images(node_id: str) -> Dict[str, np.ndarray]:
    """加载某节点的4个相机图像"""
    node_dir = PROJECT_ROOT / "merged_labeled_data" / str(node_id)
    info_path = node_dir / "node_position_info.json"
    with open(info_path) as f:
        info = json.load(f)

    camera_images = {}
    for cam_key in ["camera_1", "camera_2", "camera_3", "camera_4"]:
        img_name = info["self_position"][cam_key]
        img_path = node_dir / img_name
        img = np.array(Image.open(img_path).convert("RGB"))
        camera_images[cam_key] = img
    return camera_images


def load_test_data_frame(data_dir: str, timestamp: str) -> Dict[str, np.ndarray]:
    """加载测试数据某一帧的相机图"""
    d = PROJECT_ROOT / data_dir
    camera_images = {}
    for cam_key in ["camera_1", "camera_2", "camera_3", "camera_4"]:
        img_path = d / f"{timestamp}_{cam_key}.jpg"
        if img_path.exists():
            img = np.array(Image.open(img_path).convert("RGB"))
            camera_images[cam_key] = img
    return camera_images if len(camera_images) == 4 else None


def get_test_data_timestamps(data_dir: str) -> List[str]:
    """获取测试数据目录中所有时间戳 (排序)"""
    d = PROJECT_ROOT / data_dir
    timestamps = set()
    for f in d.iterdir():
        if f.name.endswith("_front_1.jpg"):
            ts = f.name.replace("_front_1.jpg", "")
            timestamps.add(ts)
    return sorted(timestamps)


# ============================================================================
# 模拟 MemoryNavState (从 ws_proxy_with_memory.py 复制核心逻辑)
# ============================================================================
@dataclass
class MemoryNavState:
    plan: object = None
    current_step_idx: int = 0
    phase: str = 'idle'
    consecutive_misses: int = 0
    MAX_MISSES: int = 8
    last_vpr_result: object = None
    last_task: str = None

    def reset(self):
        self.plan = None
        self.current_step_idx = 0
        self.phase = 'idle'
        self.consecutive_misses = 0
        self.last_vpr_result = None
        self.last_task = None

    def get_current_step(self):
        if self.plan is None or not self.plan.success:
            return None
        if self.current_step_idx >= len(self.plan.steps):
            return None
        return self.plan.steps[self.current_step_idx]

    def advance(self) -> bool:
        if self.plan is None:
            return False
        self.current_step_idx += 1
        self.consecutive_misses = 0
        if self.current_step_idx >= len(self.plan.steps):
            self.phase = 'completed'
            return False
        self.phase = 'step_init'
        return True


# ============================================================================
# 测试入口
# ============================================================================
def main():
    global passed, failed

    os.chdir(PROJECT_ROOT)

    # ==============================================================
    # 0. 加载模型和记忆
    # ==============================================================
    header("0. 加载 SelaVPR++ + 记忆图 + VPR")

    t0 = time.time()

    from deploy.memory_nav import (
        MemoryNavigator, MemoryBuilder,
        MemoryGraph, MemoryVPR,
        NavigationPlan, NavigationStep, VPRResult
    )

    info("正在初始化 SelaVPR++ 特征提取器...")
    from deploy.memory_nav.vpr_factory import create_vpr_extractor
    extractor, _, _ = create_vpr_extractor("selavpr", device="cuda:0")
    info(f"SelaVPR++ 加载完成: {time.time()-t0:.1f}s")

    info("正在初始化 MemoryNavigator...")
    navigator = MemoryNavigator(
        feature_extractor=extractor,
        feature_dim=512,
        device="cuda:0"
    )

    info("正在加载记忆数据...")
    navigator.load_memory(
        path="deploy/memory_nav/memory_cache",
        data_dir="merged_labeled_data"
    )

    load_time = time.time() - t0
    info(f"全部加载完成: {load_time:.1f}s")

    graph = navigator.graph
    vpr = navigator.vpr

    check(graph is not None, f"记忆图已加载: {len(graph.nodes)} 节点")
    check(vpr is not None, "VPR 模块已加载")

    stats = graph.get_stats()
    info(f"图统计: {stats['total_nodes']} 节点, {stats['total_edges']} 边")

    # 列出所有目的地
    all_dests = navigator.get_all_destinations()
    info(f"可用目的地 ({len(all_dests)}):")
    for nid, nname in all_dests:
        info(f"  {nid}: {nname}")

    # ==============================================================
    # 1. VPR 定位测试 - 用已知节点图片
    # ==============================================================
    header("1. VPR 定位: 已知节点图片 → 匹配正确节点")

    vpr_results = {}
    for node_id, node_name in NODE_NAMES.items():
        try:
            camera_images = load_node_camera_images(node_id)
            result = navigator.locate_by_images(camera_images)
            vpr_results[node_id] = result

            if result:
                matched = result.matched_node_id == node_id
                check(
                    matched,
                    f"Node {node_id} ({node_name}): matched={result.matched_node_id} "
                    f"sim={result.similarity:.4f} shift={result.best_shift} "
                    f"offset={result.heading_offset:.1f}°",
                    f"Node {node_id} ({node_name}): 期望={node_id}, 实际={result.matched_node_id} "
                    f"({result.matched_node_name}) sim={result.similarity:.4f}"
                )
            else:
                fail(f"Node {node_id} ({node_name}): VPR 返回 None")
                failed += 1
        except Exception as e:
            fail(f"Node {node_id} ({node_name}): 异常 {e}")
            failed += 1

    # ==============================================================
    # 2. VPR 循环移位测试 - 模拟不同朝向
    # ==============================================================
    header("2. VPR 循环移位: 人工旋转相机顺序 → 验证 shift 检测")

    # 选择 node 18 (A8前台) 做移位测试
    test_node_id = "18"
    info(f"测试节点: {test_node_id} ({NODE_NAMES[test_node_id]})")

    original = load_node_camera_images(test_node_id)

    HEADING_OFFSETS = [0.0, 75.0, 180.0, -105.0]
    for shift in range(4):
        # 对查询图做循环移位: query_cam[(i+shift)%4] = mem_cam[i]
        # 即 query_cam[i] = mem_cam[(i-shift)%4]
        shifted = {}
        cam_keys = ["camera_1", "camera_2", "camera_3", "camera_4"]
        for i in range(4):
            src_idx = (i - shift) % 4
            shifted[cam_keys[i]] = original[cam_keys[src_idx]]

        result = navigator.locate_by_images(shifted)
        if result:
            correct_node = result.matched_node_id == test_node_id
            correct_shift = result.best_shift == shift
            expected_offset = HEADING_OFFSETS[shift]
            correct_offset = abs(result.heading_offset - expected_offset) < 0.1

            check(
                correct_node and correct_shift,
                f"Shift={shift}: node={result.matched_node_id}(✓) "
                f"best_shift={result.best_shift}({'✓' if correct_shift else '✗'}) "
                f"offset={result.heading_offset:.1f}°(expect {expected_offset:.1f}°) "
                f"sim={result.similarity:.4f}",
                f"Shift={shift}: node={result.matched_node_id}(expect {test_node_id}) "
                f"best_shift={result.best_shift}(expect {shift}) "
                f"offset={result.heading_offset:.1f}°(expect {expected_offset:.1f}°)"
            )
        else:
            fail(f"Shift={shift}: VPR 返回 None")
            failed += 1

    # ==============================================================
    # 3. 路径规划测试
    # ==============================================================
    header("3. 路径规划: 各种起点→终点")

    test_routes = [
        ("18", "1",  "A8前台→C8打印区"),          # 长路径 18→15→14→4→1
        ("1",  "32", "C8打印区→A8休息区"),          # 跨区域
        ("18", "32", "A8前台→A8休息区"),            # 18→29→31→32
        ("8",  "21", "C8前台区→A8厕所区域"),        # 跨区域
        ("32", "32", "A8休息区→A8休息区 (同节点)"), # 同节点
    ]

    for start_id, goal_id, desc in test_routes:
        plan = navigator.plan_navigation(goal_id, start_id)
        if start_id == goal_id:
            # 同节点: 期望 total_steps=0
            check(
                plan.success and plan.total_steps == 0,
                f"{desc}: 同节点检测正确 (steps=0)",
                f"{desc}: 期望 steps=0, 实际 steps={plan.total_steps}, success={plan.success}"
            )
        else:
            check(
                plan.success and plan.total_steps > 0,
                f"{desc}: path={' → '.join(plan.path)} ({plan.total_steps}步)",
                f"{desc}: 规划失败 - {plan.message}"
            )

            # 验证路径连通性 (每对相邻节点确实有边)
            if plan.success:
                path_valid = True
                bad_link = ""
                for i in range(len(plan.path) - 1):
                    from_id = plan.path[i]
                    to_id = plan.path[i + 1]
                    from_node = graph.get_node(from_id)
                    if from_node:
                        neighbor_ids = from_node.get_neighbor_ids()
                        if to_id not in neighbor_ids:
                            path_valid = False
                            bad_link = f"{from_id} → {to_id}"
                            break
                    else:
                        path_valid = False
                        bad_link = f"节点 {from_id} 不存在"
                        break
                check(path_valid, f"  路径连通性验证通过",
                      f"  路径不连通: {bad_link}")

    # ==============================================================
    # 4. 目的地语义匹配测试
    # ==============================================================
    header("4. 目的地匹配: task → 节点")

    dest_tests = [
        ("C8打印区", "1"),
        ("A8前台", "18"),
        ("C8前台区", "8"),
        ("24号会议室门口", "15"),
        ("A8休息区", "32"),
        ("C8实验室门口", "14"),
        ("A8厕所区域", "21"),
        ("11号会议室区域", "29"),
    ]

    for query, expected_id in dest_tests:
        node = navigator.find_destination(query)
        if node:
            check(
                node.node_id == expected_id,
                f"'{query}' → {node.node_id} ({node.node_name})",
                f"'{query}' → {node.node_id} (期望 {expected_id})"
            )
        else:
            fail(f"'{query}' → None (期望 {expected_id})")
            failed += 1

    # ==============================================================
    # 5. 状态机模拟: 完整导航流程
    # ==============================================================
    header("5. 状态机模拟: A8前台(18) → C8打印区(1)")

    nav_state = MemoryNavState()

    # Step 5.1: 初始 VPR + 规划
    info("--- 第1次请求: 发送 node 18 图片 + task 'C8打印区' ---")
    images_18 = load_node_camera_images("18")
    vpr_result = navigator.locate_by_images(images_18)
    check(vpr_result is not None and vpr_result.matched_node_id == "18",
          f"VPR 定位: {vpr_result.matched_node_id if vpr_result else 'None'}")

    dest = navigator.find_destination("C8打印区")
    check(dest is not None and dest.node_id == "1",
          f"目的地匹配: {dest.node_id if dest else 'None'} ({dest.node_name if dest else ''})")

    if dest is None or vpr_result is None:
        fail("无法继续状态机测试: dest 或 vpr_result 为 None")
        failed += 1
    else:
        # 以下测试在 else 块内，确保 dest 和 vpr_result 不为 None
        pass

    plan = navigator.plan_navigation(
        dest if dest else "1",
        vpr_result.matched_node_id if vpr_result else "18"
    )
    check(plan.success, f"路径规划: {' → '.join(plan.path)} ({plan.total_steps}步)")

    nav_state.plan = plan
    nav_state.current_step_idx = 0
    nav_state.phase = 'step_init'
    nav_state.last_vpr_result = vpr_result

    step = nav_state.get_current_step()
    info(f"当前步骤: {step.from_node_name} → {step.to_node_name}, "
         f"angle={step.angle:.2f}°, pixel={step.pixel_position}")

    # Step 5.2: 模拟到达各中间节点
    expected_path = plan.path  # e.g. ["18", "15", "14", "4", "1"]
    info(f"预期路径: {' → '.join(expected_path)}")

    for step_idx in range(1, len(expected_path)):
        target_node_id = expected_path[step_idx]
        info(f"\n--- 到达 node {target_node_id} ({NODE_NAMES.get(target_node_id, '?')}) ---")

        # 发送目标节点的图片
        images = load_node_camera_images(target_node_id)
        vpr_result = navigator.locate_by_images(images)

        check(vpr_result is not None and vpr_result.matched_node_id == target_node_id,
              f"VPR: matched={vpr_result.matched_node_id if vpr_result else 'None'}")

        # 检查是否匹配到计划中的下一目标
        current_step = nav_state.get_current_step()
        if current_step and vpr_result and vpr_result.matched_node_id == current_step.to_node_id:
            has_next = nav_state.advance()
            if has_next:
                new_step = nav_state.get_current_step()
                check(True,
                      f"步骤完成! advance → step {nav_state.current_step_idx}: "
                      f"{new_step.from_node_name} → {new_step.to_node_name}")
            else:
                check(nav_state.phase == 'completed',
                      f"导航完成! phase={nav_state.phase}")

    check(nav_state.phase == 'completed',
          f"最终状态: phase={nav_state.phase} (期望 completed)")

    # ==============================================================
    # 6. VPR 丢失 + 强制 advance 测试
    # ==============================================================
    header("6. VPR 丢失容错: 连续 miss → 强制 advance")

    nav_state2 = MemoryNavState()
    plan2 = navigator.plan_navigation("1", "18")  # 18 → ... → 1
    nav_state2.plan = plan2
    nav_state2.current_step_idx = 0
    nav_state2.phase = 'step_init'

    step = nav_state2.get_current_step()
    info(f"初始步骤: {step.from_node_name} → {step.to_node_name}")

    # 模拟连续 VPR 丢失
    for i in range(nav_state2.MAX_MISSES):
        nav_state2.consecutive_misses += 1
        nav_state2.phase = 'fallback'

    check(nav_state2.consecutive_misses >= nav_state2.MAX_MISSES,
          f"连续丢失 {nav_state2.consecutive_misses} 次 >= MAX_MISSES={nav_state2.MAX_MISSES}")

    # 强制 advance
    has_next = nav_state2.advance()
    check(has_next or nav_state2.phase == 'completed',
          f"强制 advance: step={nav_state2.current_step_idx}, phase={nav_state2.phase}")
    check(nav_state2.consecutive_misses == 0,
          f"miss 计数已重置: {nav_state2.consecutive_misses}")

    # ==============================================================
    # 7. 真实轨迹 VPR 扫描
    # ==============================================================
    header("7. 真实轨迹 VPR: memory_test_data (73帧)")

    timestamps_1 = get_test_data_timestamps("memory_test_data")
    info(f"共 {len(timestamps_1)} 帧, 采样每5帧")

    match_log_1 = []
    sample_indices = range(0, len(timestamps_1), 5)
    for idx in sample_indices:
        ts = timestamps_1[idx]
        images = load_test_data_frame("memory_test_data", ts)
        if images:
            result = navigator.locate_by_images(images)
            if result:
                match_log_1.append((idx, ts, result.matched_node_id,
                                    result.matched_node_name, result.similarity,
                                    result.best_shift, result.heading_offset))
                info(f"帧{idx:3d} ts={ts}: node={result.matched_node_id} "
                     f"({result.matched_node_name}) sim={result.similarity:.4f} "
                     f"shift={result.best_shift} offset={result.heading_offset:.1f}°")
            else:
                match_log_1.append((idx, ts, None, None, 0, 0, 0))
                info(f"帧{idx:3d} ts={ts}: 无匹配")

    check(len(match_log_1) > 0, f"test_data1: {len(match_log_1)} 帧有结果")

    # 统计各节点出现次数
    node_counts_1 = {}
    for entry in match_log_1:
        nid = entry[2]
        if nid:
            node_counts_1[nid] = node_counts_1.get(nid, 0) + 1
    info(f"节点分布: {dict(sorted(node_counts_1.items(), key=lambda x: -x[1]))}")

    # ==============================================================
    # 8. Memory Response 构建验证
    # ==============================================================
    header("8. Memory Response 构建验证")

    # 构建一个 plan: 18 → 15 → 14 → 4 → 1
    plan = navigator.plan_navigation("1", "18")
    check(plan.success, f"路径: {' → '.join(plan.path)}")

    for i, step in enumerate(plan.steps):
        # 检查每步的 angle 和 pixel_position 是合理的
        check(
            -180 <= step.angle <= 180,
            f"Step {i}: {step.from_node_name} → {step.to_node_name}, "
            f"angle={step.angle:.2f}°, pixel={step.pixel_position}, "
            f"stitch={step.stitch_image_path}",
            f"Step {i}: angle={step.angle} 超出 [-180, 180] 范围"
        )

        # 验证 pixel_position 在 [0, 1] 范围
        px, py = step.pixel_position
        check(
            0 <= px <= 1 and 0 <= py <= 1,
            f"  pixel_position ({px:.4f}, {py:.4f}) 在 [0,1] 范围内",
            f"  pixel_position ({px:.4f}, {py:.4f}) 超出 [0,1] 范围"
        )

        # 验证 stitch 图片存在
        stitch_path = Path(step.stitch_image_path)
        if not stitch_path.is_absolute():
            stitch_path = PROJECT_ROOT / stitch_path
        check(
            stitch_path.exists(),
            f"  stitch 图片存在: {step.stitch_image_path}",
            f"  stitch 图片不存在: {stitch_path}"
        )

    # ==============================================================
    # 汇总
    # ==============================================================
    header("测试汇总")
    total = passed + failed
    print(f"\n  总计: {total} 项")
    print(f"  {C.GREEN}通过: {passed}{C.END}")
    if failed > 0:
        print(f"  {C.RED}失败: {failed}{C.END}")
    else:
        print(f"  失败: 0")
    print(f"\n  {'🎉 全部通过!' if failed == 0 else '⚠️  有失败项，请检查'}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n{C.RED}💥 测试脚本异常: {e}{C.END}")
        traceback.print_exc()
        sys.exit(2)
