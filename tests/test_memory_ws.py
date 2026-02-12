#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_memory_ws.py - 三层记忆导航策略 WebSocket 集成测试

使用 memory_test_data/ 真实轨迹数据，完整回放 146 帧，
验证从 C8打印区(1) 经 C8微波炉区域(3) 到达 C8前台区(8) 的完整记忆导航过程。

三层策略:
  Layer 1: 记忆引导 - 每步首次请求返回 angle + pixel_goal
  Layer 2: VPR持续验证 - 匹配目标节点→advance, 匹配源节点→重复引导
  Layer 3: 模型兜底 - VPR失败或匹配路径外节点时的处理
  稀疏容错: 连续N次VPR失败→强制advance

用法:
  1. 启动 ws_proxy_with_memory.py:
     cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
     python3 deploy/ws_proxy_with_memory.py

  2. 运行测试:
     python3 tests/test_memory_ws.py

  3. 查看服务端完整日志:
     tail -f deploy/logs/ws_proxy_with_memory.log
"""

import asyncio
import json
import os
import sys
import base64
import time

WS_URL = "ws://localhost:9528"
PROJECT_ROOT = "/home/ubuntu/Disk/codes/jianxiong/MemoryNav"
DATA_DIR = os.path.join(PROJECT_ROOT, "memory_test_data")
TASK = "前往C8前台"
SAMPLE_STEP = 1 # 每 N 帧采样一次


def get_timestamps():
    """获取 memory_test_data 中所有帧的时间戳(排序)"""
    files = os.listdir(DATA_DIR)
    ts_set = set()
    for f in files:
        if 'camera_1' in f and f.endswith('.jpg'):
            ts_set.add(f.split('_camera')[0])
    return sorted(ts_set)


def load_frame(ts):
    """加载一帧图像 → base64 dict"""
    images = {}
    for key in ['camera_1', 'camera_2', 'camera_3', 'camera_4', 'front_1']:
        path = os.path.join(DATA_DIR, f"{ts}_{key}.jpg")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                images[key] = base64.b64encode(f.read()).decode()
    if 'front_1' not in images and 'camera_1' in images:
        images['front_1'] = images['camera_1']
    return images


async def send_frame(ws, task, images, pts=None):
    """发送一帧请求并接收响应"""
    msg = {"id": "test_robot", "task": task, "images": images}
    if pts is not None:
        msg["pts"] = pts
    await ws.send(json.dumps(msg))
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=30))


async def send_command(ws, command):
    """发送命令并接收响应"""
    await ws.send(json.dumps({"command": command}))
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=10))


async def run_test():
    """完整轨迹回放测试"""

    print("=" * 70)
    print("🧪 三层记忆导航策略 — 完整轨迹回放测试")
    print(f"   服务器: {WS_URL}")
    print(f"   任务: {TASK}")
    print(f"   数据: {DATA_DIR}")
    print("=" * 70)

    timestamps = get_timestamps()
    total_frames = len(timestamps)
    print(f"📁 测试数据: {total_frames} 帧, 每{SAMPLE_STEP}帧采样")

    if total_frames < 10:
        print(f"❌ 测试数据不足 ({total_frames} 帧), 需要至少 50 帧")
        sys.exit(1)

    try:
        import websockets
    except ImportError:
        print("❌ 需要安装 websockets: pip install websockets")
        sys.exit(1)

    # 连接
    print(f"\n🔗 连接 {WS_URL}...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, max_size=50 * 1024 * 1024),
            timeout=15
        )
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print(f"   请先启动: cd {PROJECT_ROOT} && python3 deploy/ws_proxy_with_memory.py")
        sys.exit(1)

    print("✅ 已连接")

    # 检查记忆导航状态
    status = await send_command(ws, 'memory_status')
    stats = status.get('memory_navigator_graph_stats', {})
    print(f"📊 记忆图: {stats.get('total_nodes', 0)} 节点, {stats.get('total_edges', 0)} 边")
    print(f"🧠 记忆导航: {'启用' if status.get('memory_enabled') else '禁用'}")

    # 重置状态
    await send_command(ws, 'reset_memory')

    # ====================================================================
    # 完整回放
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"▶️  开始任务: {TASK}")
    print(f"{'='*70}\n")

    sample_indices = list(range(0, total_frames, SAMPLE_STEP))
    completed = False
    last_phase = None
    last_step = -1
    start_time = time.time()

    for seq, frame_idx in enumerate(sample_indices):
        ts = timestamps[frame_idx]
        imgs = load_frame(ts)
        resp = await send_frame(ws, TASK, imgs, pts=int(ts))

        mi = resp.get('memory_info', {})
        phase = mi.get('phase', '')
        step = mi.get('current_step', -1)
        active = resp.get('memory_active', False)
        angle = resp.get('angle')
        pixel = resp.get('pixel_target')

        # 状态转换时打印
        if phase != last_phase or step != last_step:
            from_node = mi.get('from_node', '')
            to_node = mi.get('to_node', '')
            msg = resp.get('message', '')[:60]
            angle_str = f"angle={angle:.1f}°" if angle is not None else ""
            pixel_str = f"pixel=({pixel[0]:.2f},{pixel[1]:.2f})" if pixel else ""
            nav_str = f"{angle_str} {pixel_str}".strip()

            print(f"  帧{frame_idx:3d} | step={step} | {phase:10s} | "
                  f"{from_node:10s} → {to_node:10s} | {nav_str}")
            if msg:
                print(f"         └─ {msg}")
            last_phase = phase
            last_step = step

        if resp.get('task_status') == 'end':
            completed = True
            break

    elapsed = time.time() - start_time
    frames_sent = seq + 1

    # ====================================================================
    # 结果
    # ====================================================================
    print(f"\n{'='*70}")
    if completed:
        print(f"✅ 导航完成! 发送{frames_sent}帧(采样自{total_frames}帧), 耗时{elapsed:.1f}s")
    else:
        print(f"❌ 导航未完成. 发送{frames_sent}帧, 耗时{elapsed:.1f}s")

    print(f"{'='*70}")
    print(f"\n📋 服务端详细日志: deploy/logs/ws_proxy_with_memory.log")

    await ws.close()

    if not completed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_test())
