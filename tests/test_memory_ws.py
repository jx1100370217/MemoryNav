#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_memory_ws.py - 三层记忆导航策略 WebSocket 集成测试

使用 memory_test_data/ 真实轨迹数据，完整回放帧序列，
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
from collections import defaultdict

WS_URL = "ws://127.0.0.1:9528"
PROJECT_ROOT = "/home/ubuntu/Disk/codes/jianxiong/MemoryNav"
DATA_DIR = os.path.join(PROJECT_ROOT, "memory_test_data")
TASK = "前往C8前台"
SAMPLE_STEP = 1  # 每 N 帧采样一次

# ANSI 颜色
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"
C_BG_GREEN = "\033[42m"
C_BG_RED = "\033[41m"
C_BG_YELLOW = "\033[43m"
C_BG_BLUE = "\033[44m"


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


def fmt_similarity(sim, threshold=0.70):
    """格式化相似度值，带颜色标识"""
    if sim >= threshold:
        return f"{C_GREEN}{sim:.4f}{C_RESET}"
    elif sim >= threshold * 0.9:
        return f"{C_YELLOW}{sim:.4f}{C_RESET}"
    else:
        return f"{C_DIM}{sim:.4f}{C_RESET}"


def fmt_phase(phase):
    """格式化 phase 状态"""
    colors = {
        'step_init': C_CYAN,
        'verifying': C_BLUE,
        'model_fallback': C_YELLOW,
        'completed': C_GREEN,
        'trend_go_straight': C_MAGENTA,
    }
    color = colors.get(phase, C_WHITE)
    return f"{color}{phase:18s}{C_RESET}"


def fmt_decision(decision):
    """格式化决策类型"""
    colors = {
        'advance': f"{C_BG_GREEN}{C_WHITE} ADVANCE {C_RESET}",
        'skip_advance': f"{C_BG_GREEN}{C_WHITE} SKIP→ADV {C_RESET}",
        'continue': f"{C_BLUE}CONTINUE{C_RESET}",
        'miss': f"{C_RED}VPR MISS{C_RESET}",
        'replan': f"{C_MAGENTA}REPLAN{C_RESET}",
        'go_straight': f"{C_CYAN}GO STRAIGHT{C_RESET}",
        'force_advance': f"{C_BG_YELLOW}{C_WHITE} FORCE ADV {C_RESET}",
        'completed': f"{C_BG_GREEN}{C_WHITE} COMPLETED {C_RESET}",
        'init': f"{C_DIM}INIT{C_RESET}",
    }
    return colors.get(decision, decision)


def print_separator(char='─', width=120):
    print(f"{C_DIM}{char * width}{C_RESET}")


def print_header(title, width=120):
    print(f"\n{C_BOLD}{'═' * width}{C_RESET}")
    padding = (width - len(title) - 4) // 2
    print(f"{C_BOLD}{'═' * padding}  {title}  {'═' * padding}{C_RESET}")
    print(f"{C_BOLD}{'═' * width}{C_RESET}")


async def run_test():
    """完整轨迹回放测试"""

    print_header("🧪 三层记忆导航策略 — 完整轨迹回放测试")
    print(f"  {C_BOLD}服务器:{C_RESET}  {WS_URL}")
    print(f"  {C_BOLD}任务:{C_RESET}    {TASK}")
    print(f"  {C_BOLD}数据:{C_RESET}    {DATA_DIR}")
    print(f"  {C_BOLD}采样:{C_RESET}    每 {SAMPLE_STEP} 帧")

    timestamps = get_timestamps()
    total_frames = len(timestamps)
    print(f"  {C_BOLD}总帧数:{C_RESET}  {total_frames}")

    if total_frames < 10:
        print(f"\n{C_RED}❌ 测试数据不足 ({total_frames} 帧), 需要至少 10 帧{C_RESET}")
        sys.exit(1)

    try:
        import websockets
    except ImportError:
        print(f"\n{C_RED}❌ 需要安装 websockets: pip install websockets{C_RESET}")
        sys.exit(1)

    print(f"\n{C_CYAN}🔗 连接 {WS_URL}...{C_RESET}")

    # WebSocket 连接 (带重试)
    ws = None
    for _attempt in range(3):
        try:
            ws = await websockets.connect(
                WS_URL,
                max_size=50 * 1024 * 1024,
                open_timeout=30,
                ping_interval=30,
                ping_timeout=10,
            )
            break
        except Exception as e:
            if _attempt < 2:
                print(f"{C_YELLOW}⚠️  连接尝试 {_attempt+1}/3 失败: {e}, 3秒后重试...{C_RESET}")
                await asyncio.sleep(3)
            else:
                print(f"{C_RED}❌ 连接失败 (已重试3次): {e}{C_RESET}")
                print(f"   请先启动: cd {PROJECT_ROOT} && python deploy/ws_proxy_with_memory.py")
                sys.exit(1)

    print(f"{C_GREEN}✅ 已连接{C_RESET}")

    # 检查记忆导航状态
    status = await send_command(ws, 'memory_status')
    stats = status.get('memory_navigator_graph_stats') or {}
    mem_enabled = status.get('memory_enabled', False)

    print_separator('─')
    print(f"  {C_BOLD}记忆图:{C_RESET}    {stats.get('total_nodes', 0)} 节点, {stats.get('total_edges', 0)} 边")
    print(f"  {C_BOLD}记忆导航:{C_RESET}  {'✅ 启用' if mem_enabled else '❌ 禁用'}")
    if not mem_enabled:
        print(f"{C_RED}⚠️  记忆导航未启用，测试可能无法正常运行{C_RESET}")
    print_separator('─')

    # 重置状态
    await send_command(ws, 'reset_memory')
    print(f"  {C_DIM}🔄 已重置记忆导航状态{C_RESET}")

    # ====================================================================
    # 完整回放
    # ====================================================================
    print_header(f"▶️  开始任务: {TASK}")

    # 表头
    print(f"\n{C_BOLD}{'帧':>5s} │ {'步骤':>4s} │ {'Phase':^18s} │ {'决策':^14s} │ "
          f"{'VPR匹配节点':^20s} │ {'VPR sim':>8s} │ {'VPR conf':>8s} │ "
          f"{'from → to':^30s} │ {'angle':>7s} │ {'pixel_target':>14s} │ {'misses':>6s}{C_RESET}")
    print_separator('─')

    sample_indices = list(range(0, total_frames, SAMPLE_STEP))
    completed = False
    start_time = time.time()

    # 统计
    stat_decisions = defaultdict(int)
    stat_vpr_matches = defaultdict(int)  # node_id -> count
    stat_phases = defaultdict(int)
    stat_total_frames = 0
    stat_vpr_hits = 0
    stat_vpr_misses = 0
    stat_max_consecutive_misses = 0
    stat_steps_completed = 0
    stat_sim_history = []  # (frame_idx, vpr_sim, vpr_conf, matched_node)

    last_step = -1
    last_phase = None

    for seq, frame_idx in enumerate(sample_indices):
        ts = timestamps[frame_idx]
        imgs = load_frame(ts)
        t0 = time.time()
        resp = await send_frame(ws, TASK, imgs, pts=int(ts))
        latency_ms = (time.time() - t0) * 1000

        stat_total_frames += 1

        mi = resp.get('memory_info', {})
        phase = mi.get('phase', 'unknown')
        step_idx = mi.get('current_step', -1)
        total_steps = mi.get('total_steps', 0)
        from_node = mi.get('from_node', '')
        from_node_id = mi.get('from_node_id', '')
        to_node = mi.get('to_node', '')
        to_node_id = mi.get('to_node_id', '')
        vpr_sim = mi.get('vpr_similarity', 0.0)
        vpr_conf = mi.get('vpr_confidence', 0.0)
        vpr_matched = mi.get('vpr_matched_node', None)
        consecutive_misses = mi.get('consecutive_misses', 0)
        heading_offset = mi.get('heading_offset', 0.0)
        plan_path = mi.get('plan_path', [])

        active = resp.get('memory_active', False)
        angle = resp.get('angle')
        pixel = resp.get('pixel_target')
        task_status = resp.get('task_status', '')
        message = resp.get('message', '')

        # 统计
        stat_phases[phase] += 1
        stat_max_consecutive_misses = max(stat_max_consecutive_misses, consecutive_misses)

        if vpr_matched:
            stat_vpr_hits += 1
            stat_vpr_matches[vpr_matched] += 1
        else:
            stat_vpr_misses += 1

        stat_sim_history.append((frame_idx, vpr_sim, vpr_conf, vpr_matched))

        # 判断决策类型
        decision = 'continue'
        step_changed = (step_idx != last_step and last_step != -1)
        phase_changed = (phase != last_phase)

        if task_status == 'end':
            decision = 'completed'
        elif phase == 'step_init' and step_changed and step_idx > last_step:
            if step_idx > last_step + 1:
                decision = 'skip_advance'
            else:
                decision = 'advance'
            stat_steps_completed += 1
        elif vpr_matched is None and active:
            decision = 'miss'
        elif phase == 'step_init' and last_step == -1:
            decision = 'init'
        elif 'go straight' in message.lower() or 'go_straight' in phase:
            decision = 'go_straight'
        elif '重规划' in message or '重新规划' in message:
            decision = 'replan'

        stat_decisions[decision] += 1

        # ─── 逐帧详细日志 ───
        # VPR 匹配节点显示
        if vpr_matched:
            # 尝试从响应中获取匹配节点名称
            matched_name = vpr_matched
            if vpr_matched == to_node_id:
                matched_display = f"{C_GREEN}✓ {matched_name}{C_RESET}"
            elif vpr_matched == from_node_id:
                matched_display = f"{C_YELLOW}↺ {matched_name}{C_RESET}"
            else:
                matched_display = f"{C_RED}⚡{matched_name}{C_RESET}"
        else:
            matched_display = f"{C_DIM}──{C_RESET}"

        # 格式化各字段
        step_str = f"{step_idx+1}/{total_steps}" if total_steps > 0 else "─"
        angle_str = f"{angle:7.1f}°" if angle is not None else f"{'─':>7s}"
        pixel_str = f"({pixel[0]:.3f},{pixel[1]:.3f})" if pixel else f"{'─':>14s}"
        miss_str = f"{consecutive_misses}" if consecutive_misses > 0 else f"{C_DIM}0{C_RESET}"
        nav_str = f"{from_node} → {to_node}" if from_node else "─"

        print(f"{frame_idx:5d} │ {step_str:>4s} │ {fmt_phase(phase)} │ {fmt_decision(decision):>14s} │ "
              f"{matched_display:>20s} │ {fmt_similarity(vpr_sim):>8s} │ {fmt_similarity(vpr_conf):>8s} │ "
              f"{nav_str:^30s} │ {angle_str} │ {pixel_str:>14s} │ {miss_str:>6s}")

        # 关键事件详细信息
        if decision in ('advance', 'skip_advance'):
            print(f"      {C_GREEN}│ ✅ 步骤前进! VPR匹配到目标节点 {to_node_id}, "
                  f"sim={vpr_sim:.4f}, conf={vpr_conf:.4f}{C_RESET}")
            if plan_path:
                path_display = ' → '.join(plan_path)
                progress = f"[{'█' * (step_idx+1)}{'░' * (total_steps - step_idx - 1)}]"
                print(f"      {C_DIM}│ 📍 路径: {path_display}{C_RESET}")
                print(f"      {C_DIM}│ 📊 进度: {progress} {step_idx+1}/{total_steps}{C_RESET}")

        elif decision == 'replan':
            print(f"      {C_MAGENTA}│ 🔄 重新规划路径! 当前匹配: {vpr_matched}, "
                  f"新路径: {' → '.join(plan_path)}{C_RESET}")

        elif decision == 'miss' and consecutive_misses >= 3:
            print(f"      {C_RED}│ ⚠️ 连续VPR丢失 {consecutive_misses} 次!{C_RESET}")

        elif decision == 'completed':
            print(f"      {C_GREEN}│ 🎉 {message}{C_RESET}")

        # 补充: 有消息且是重要消息时打印
        if message and decision not in ('continue', 'miss', 'init', 'completed'):
            if '完成' in message or '重规划' in message or '强制' in message:
                print(f"      {C_DIM}│ 💬 {message}{C_RESET}")

        last_step = step_idx
        last_phase = phase

        if task_status == 'end':
            completed = True
            break

    elapsed = time.time() - start_time
    frames_sent = stat_total_frames

    # ====================================================================
    # 详细统计报告
    # ====================================================================
    print_header("📊 测试统计报告")

    # 基本信息
    print(f"\n  {C_BOLD}【基本信息】{C_RESET}")
    print(f"  {'总帧数':>16s}: {total_frames}")
    print(f"  {'发送帧数':>16s}: {frames_sent}")
    print(f"  {'采样间隔':>16s}: 每 {SAMPLE_STEP} 帧")
    print(f"  {'总耗时':>16s}: {elapsed:.1f}s")
    print(f"  {'平均延迟':>16s}: {elapsed/max(frames_sent,1)*1000:.0f}ms/帧")

    # 导航结果
    print(f"\n  {C_BOLD}【导航结果】{C_RESET}")
    if completed:
        print(f"  {'状态':>16s}: {C_GREEN}✅ 导航完成{C_RESET}")
    else:
        print(f"  {'状态':>16s}: {C_RED}❌ 导航未完成{C_RESET}")
    print(f"  {'完成步数':>16s}: {stat_steps_completed}")

    # VPR 统计
    total_vpr = stat_vpr_hits + stat_vpr_misses
    hit_rate = stat_vpr_hits / max(total_vpr, 1) * 100
    print(f"\n  {C_BOLD}【VPR 匹配统计】{C_RESET}")
    print(f"  {'VPR 匹配成功':>16s}: {stat_vpr_hits} / {total_vpr} ({hit_rate:.1f}%)")
    print(f"  {'VPR 匹配失败':>16s}: {stat_vpr_misses}")
    print(f"  {'最大连续丢失':>16s}: {stat_max_consecutive_misses}")

    if stat_sim_history:
        sims = [s[1] for s in stat_sim_history if s[3] is not None]
        confs = [s[2] for s in stat_sim_history if s[3] is not None]
        if sims:
            print(f"  {'相似度 (匹配帧)':>16s}: min={min(sims):.4f}  avg={sum(sims)/len(sims):.4f}  max={max(sims):.4f}")
        if confs:
            print(f"  {'置信度 (匹配帧)':>16s}: min={min(confs):.4f}  avg={sum(confs)/len(confs):.4f}  max={max(confs):.4f}")

    # VPR 匹配节点分布
    if stat_vpr_matches:
        print(f"\n  {C_BOLD}【VPR 匹配节点分布】{C_RESET}")
        for node_id, count in sorted(stat_vpr_matches.items(), key=lambda x: -x[1]):
            bar_len = int(count / max(stat_vpr_matches.values()) * 30)
            bar = '█' * bar_len
            print(f"    {node_id:>20s}: {count:4d} 次 {C_CYAN}{bar}{C_RESET}")

    # 决策分布
    print(f"\n  {C_BOLD}【决策类型分布】{C_RESET}")
    for dec, count in sorted(stat_decisions.items(), key=lambda x: -x[1]):
        bar_len = int(count / max(stat_decisions.values()) * 30)
        bar = '█' * bar_len
        print(f"    {dec:>16s}: {count:4d} 次 {C_BLUE}{bar}{C_RESET}")

    # Phase 分布
    print(f"\n  {C_BOLD}【Phase 状态分布】{C_RESET}")
    for ph, count in sorted(stat_phases.items(), key=lambda x: -x[1]):
        bar_len = int(count / max(stat_phases.values()) * 30)
        bar = '█' * bar_len
        print(f"    {ph:>18s}: {count:4d} 次 {C_MAGENTA}{bar}{C_RESET}")

    # VPR 相似度变化趋势 (ASCII 图)
    if stat_sim_history and len(stat_sim_history) > 5:
        print(f"\n  {C_BOLD}【VPR 相似度趋势】{C_RESET}")
        # 选取最多 60 个采样点
        step_size = max(1, len(stat_sim_history) // 60)
        sampled = stat_sim_history[::step_size]
        chart_height = 8
        max_sim = 1.0
        min_sim = 0.0

        # 每行输出
        for row in range(chart_height, -1, -1):
            threshold_val = min_sim + (max_sim - min_sim) * row / chart_height
            line = f"  {threshold_val:5.2f} │"
            for _, sim, _, matched in sampled:
                if matched is not None:
                    level = int((sim - min_sim) / (max_sim - min_sim) * chart_height)
                    if level == row:
                        line += f"{C_GREEN}●{C_RESET}"
                    elif level > row:
                        line += f"{C_DIM}│{C_RESET}"
                    else:
                        line += " "
                else:
                    if row == 0:
                        line += f"{C_RED}✕{C_RESET}"
                    else:
                        line += " "
            print(line)
        print(f"        └{'─' * len(sampled)}")
        print(f"         {C_DIM}帧序列 (● = VPR匹配, ✕ = 丢失){C_RESET}")

    print_separator('═')
    if completed:
        print(f"\n{C_GREEN}{C_BOLD}🎉 测试通过!{C_RESET}")
    else:
        print(f"\n{C_RED}{C_BOLD}💥 测试失败 — 导航未完成{C_RESET}")
    print(f"{C_DIM}📋 服务端详细日志: deploy/logs/ws_proxy_with_memory.log{C_RESET}\n")

    if ws:
        await ws.close()

    if not completed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_test())
