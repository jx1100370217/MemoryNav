#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_memory_ws.py - 记忆导航/在线建图 双模式 WebSocket 集成测试

使用 memory_test_data/ 真实轨迹数据，完整回放帧序列，
通过 --mode {nav,mapping} 选择测试导航模式或建图模式。

导航模式 (--mode nav, 默认):
  Layer 1: 记忆引导 - 每步返回 camera_name + landmark_name + pixel_target (子图匹配)
  Layer 2: VPR持续验证 - 匹配目标节点→advance, 匹配源节点→重复引导
  Layer 3: 模型兜底 - VPR失败或匹配路径外节点时走 InternVLA 推理
  趋势检测: 相似度趋势判断方向正确性
  稀疏容错: 连续偏离超限→强制advance

建图模式 (--mode mapping):
  发送 start_mapping 切换服务端到建图模式 → 逐帧喂入 4 相机 → stop_mapping 触发 finalize
  输出节点/拓扑/场景图/占据栅格/可视化 PNG

用法:
  1. 启动 ws_proxy_with_memory.py:
     cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
     python3 deploy/ws_proxy_with_memory.py

  2. 运行测试:
     python3 tests/test_memory_ws.py                  # 默认 nav 模式
     python3 tests/test_memory_ws.py --mode mapping   # 建图模式

  3. 查看服务端完整日志:
     tail -f deploy/logs/ws_proxy_with_memory.log
"""

import argparse
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
# TASK = "前往母婴室门口"
SAMPLE_STEP = 1  # 每 N 帧采样一次

# 从服务端导入阈值常量，确保测试与服务使用同一套阈值
try:
    sys.path.insert(0, PROJECT_ROOT)
    from deploy.ws_proxy_with_memory import SUB_MATCH_CONFIDENCE_THRESHOLD, FRAME_SIMILARITY_THRESHOLD
except ImportError:
    SUB_MATCH_CONFIDENCE_THRESHOLD = 0.35
    FRAME_SIMILARITY_THRESHOLD = 0.70

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


def fmt_similarity(sim, threshold=FRAME_SIMILARITY_THRESHOLD):
    """格式化相似度值，带颜色标识"""
    if sim >= threshold:
        return f"{C_GREEN}{sim:.4f}{C_RESET}"
    elif sim >= threshold * 0.9:
        return f"{C_YELLOW}{sim:.4f}{C_RESET}"
    else:
        return f"{C_DIM}{sim:.4f}{C_RESET}"


def fmt_confidence(conf, threshold=SUB_MATCH_CONFIDENCE_THRESHOLD):
    """格式化子图匹配置信度"""
    if conf >= threshold:
        return f"{C_GREEN}{conf:.3f}{C_RESET}"
    elif conf > 0:
        return f"{C_RED}{conf:.3f}{C_RESET}"
    else:
        return f"{C_DIM}  ─  {C_RESET}"


def fmt_phase(phase):
    """格式化 phase 状态"""
    colors = {
        'step_init': C_CYAN,
        'verifying': C_BLUE,
        'fallback': C_YELLOW,
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
        'fallback': f"{C_YELLOW}FALLBACK{C_RESET}",
        'vpr_held': f"{C_YELLOW}VPR HELD{C_RESET}",
    }
    return colors.get(decision, decision)


def print_separator(char='─', width=140):
    print(f"{C_DIM}{char * width}{C_RESET}")


def print_header(title, width=140):
    print(f"\n{C_BOLD}{'═' * width}{C_RESET}")
    padding = (width - len(title) - 4) // 2
    print(f"{C_BOLD}{'═' * padding}  {title}  {'═' * padding}{C_RESET}")
    print(f"{C_BOLD}{'═' * width}{C_RESET}")


async def run_test_nav():
    """完整轨迹回放测试 (导航模式)"""

    print_header("🧪 记忆导航 — 完整轨迹回放测试")
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
          f"{'from → to':^30s} │ {'camera':>10s} │ {'sub_conf':>8s} │ "
          f"{'la_conf':>8s} │ {'pixel_target':>14s}{C_RESET}")
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
    stat_sub_match_hits = 0
    stat_sub_match_misses = 0
    stat_cache_reused = 0         # 帧间相似度复用缓存次数
    stat_cache_cleared = 0        # 场景变化清除缓存次数
    stat_cache_no_cache = 0       # 无缓存可用次数
    stat_frame_sims = []          # 帧间相似度历史 (frame_idx, sim, reused)
    stat_cache_reused = 0         # 帧间相似度复用缓存次数
    stat_cache_cleared = 0        # 场景变化清除缓存次数
    stat_cache_no_cache = 0       # 无缓存可用次数
    stat_frame_sims = []          # 帧间相似度历史 (frame_idx, sim, reused)
    stat_lookahead_hits = 0       # lookahead 下一步子图匹配成功次数
    stat_lookahead_misses = 0     # lookahead 下一步子图匹配失败次数
    stat_lookahead_triggered_advance = 0  # lookahead 触发 advance 的次数
    stat_vpr_held_by_lookahead = 0        # VPR 到了但 lookahead 未确认，暂缓 advance 的次数

    last_step = -1
    last_phase = None
    last_to_node_id = None
    last_to_node = None

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
        plan_path = mi.get('plan_path', [])

        active = resp.get('memory_active', False)
        camera_name = resp.get('camera_name', None)
        landmark_name = resp.get('landmark_name', None)
        pixel_target = resp.get('pixel_target', None)
        task_status = resp.get('task_status', '')
        message = resp.get('message', '')

        # action + coord_transform 调试信息
        action = resp.get('action', [[0.0, 0.0, 0.0]])
        coord_debug = mi.get('coord_transform', None)

        # 帧间相似度 & 缓存复用信息（从 memory_info 提取，先于子图统计）
        frame_sim = mi.get('frame_similarity', None)
        cache_action = mi.get('cache_action', None)  # 'reused' / 'cleared' / 'no_cache' / 'accepted'

        # 子图匹配信息（缓存复用不计入成功率）
        sub_match = resp.get('sub_image_match')
        sub_conf = 0.0
        sub_found = False
        if sub_match and sub_match.get('match'):
            sub_conf = sub_match['match'].get('confidence', 0.0)
            sub_found = sub_match['match'].get('found', False)
            if cache_action != 'reused':
                if sub_conf >= SUB_MATCH_CONFIDENCE_THRESHOLD:
                    stat_sub_match_hits += 1
                else:
                    stat_sub_match_misses += 1
        if cache_action == 'reused':
            stat_cache_reused += 1
        elif cache_action == 'cleared':
            stat_cache_cleared += 1
        elif cache_action == 'no_cache':
            stat_cache_no_cache += 1
        if frame_sim is not None:
            stat_frame_sims.append((frame_idx, frame_sim, cache_action))

        # lookahead 信息（从 memory_info 中解析）
        # 服务端会在响应的 memory_info 中记录 lookahead 结果（如果有）
        # 但由于响应结构未直接暴露 lookahead，我们通过 message 和 advance 决策间接统计

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

        if task_status == 'end':
            decision = 'completed'
        elif phase == 'step_init' and step_changed and step_idx > last_step:
            if step_idx > last_step + 1:
                decision = 'skip_advance'
            else:
                decision = 'advance'
                # lookahead 触发的 advance 会在 message 中包含 "下一步子图匹配成功"
                if '下一步子图匹配成功' in message:
                    stat_lookahead_triggered_advance += 1
            stat_steps_completed += 1
        elif vpr_matched is None and active:
            decision = 'miss'
        elif phase == 'step_init' and last_step == -1:
            decision = 'init'
        elif 'go straight' in message.lower() or 'go_straight' in phase:
            decision = 'go_straight'
        elif '重规划' in message or '重新规划' in message:
            decision = 'replan'
        elif phase == 'fallback':
            decision = 'fallback'

        # 检测 VPR 到了目标但 lookahead 未确认（暂缓 advance）
        if (vpr_matched == to_node_id and not step_changed
                and phase == 'verifying' and active
                and '暂不切换' in message):
            decision = 'vpr_held'
            stat_vpr_held_by_lookahead += 1

        stat_decisions[decision] += 1

        # ─── 逐帧详细日志 ───
        # VPR 匹配节点显示
        if vpr_matched:
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
        camera_str = f"{camera_name}" if camera_name else f"{'─':>10s}"
        sub_conf_str = fmt_confidence(sub_conf)
        pixel_str = f"({pixel_target[0]:.3f},{pixel_target[1]:.3f})" if pixel_target else f"{'─':>14s}"
        nav_str = f"{from_node} → {to_node}" if from_node else "─"

        # 格式化 action
        if action and len(action) > 0 and len(action[0]) >= 2:
            a0 = action[0]
            ax, ay = a0[0], a0[1]
            if abs(ax) < 0.001 and abs(ay) < 0.001:
                action_str = f"{C_DIM}{'─':>14s}{C_RESET}"
            else:
                action_str = f"({ax:+.2f},{ay:+.2f})"
        else:
            action_str = f"{C_DIM}{'─':>14s}{C_RESET}"

        # lookahead 子图匹配置信度
        la_conf = mi.get('lookahead_conf', None)
        la_found = mi.get('lookahead_found', None)
        if la_conf is not None:
            if la_found:
                la_conf_str = f"{C_GREEN}{la_conf:.3f}{C_RESET}"
            elif la_conf > 0:
                la_conf_str = f"{C_RED}{la_conf:.3f}{C_RESET}"
            else:
                la_conf_str = f"{C_DIM}  ─  {C_RESET}"
        else:
            la_conf_str = f"{C_DIM}  ─  {C_RESET}"

        print(f"{frame_idx:5d} │ {step_str:>4s} │ {fmt_phase(phase)} │ {fmt_decision(decision):>14s} │ "
              f"{matched_display:>20s} │ {fmt_similarity(vpr_sim):>8s} │ {fmt_similarity(vpr_conf):>8s} │ "
              f"{nav_str:^30s} │ {camera_str:>10s} │ {sub_conf_str:>8s} │ "
              f"{la_conf_str:>8s} │ {action_str:>14s}")

        # 关键事件详细信息
        if decision in ('advance', 'skip_advance'):
            _la_tag = f" + 🔭 lookahead确认" if '下一步子图匹配成功' in message else ""
            _triggered_node = last_to_node_id or to_node_id  # advance 是由上一帧的目标节点触发的
            _triggered_name = last_to_node or to_node
            print(f"      {C_GREEN}│ ✅ 步骤前进! VPR匹配到目标节点 {_triggered_node} ({_triggered_name}), "
                  f"sim={vpr_sim:.4f}, conf={vpr_conf:.4f}{_la_tag}{C_RESET}")
            if plan_path:
                path_display = ' → '.join(plan_path)
                progress = f"[{'█' * (step_idx+1)}{'░' * (total_steps - step_idx - 1)}]"
                print(f"      {C_DIM}│ 📍 路径: {path_display}{C_RESET}")
                print(f"      {C_DIM}│ 📊 进度: {progress} {step_idx+1}/{total_steps}{C_RESET}")

        elif decision == 'replan':
            print(f"      {C_MAGENTA}│ 🔄 重新规划路径! 当前匹配: {vpr_matched}, "
                  f"新路径: {' → '.join(plan_path)}{C_RESET}")

        elif decision == 'vpr_held':
            print(f"      {C_YELLOW}│ ⏳ VPR匹配到目标节点 {to_node_id}，但下一步子图匹配未成功，暂不切换{C_RESET}")

        elif decision == 'miss' and consecutive_misses >= 3:
            print(f"      {C_RED}│ ⚠️ 连续VPR丢失 {consecutive_misses} 次!{C_RESET}")

        elif decision == 'completed':
            print(f"      {C_GREEN}│ 🎉 {message}{C_RESET}")

        # 子图匹配详情 (关键帧)
        if sub_match and landmark_name and decision in ('init', 'advance', 'skip_advance'):
            center = sub_match.get('match', {}).get('center_pct', {})
            print(f"      {C_DIM}│ 🎯 子图匹配: camera={camera_name}, landmark={landmark_name}, "
                  f"conf={sub_conf:.3f}, center=({center.get('x', 0):.3f},{center.get('y', 0):.3f}){C_RESET}")

        # 子图匹配缓存复用详情
        if cache_action and cache_action != 'accepted':
            if cache_action == 'reused':
                print(f"      {C_CYAN}│ 🔄 缓存复用: 帧间DINOv2相似度={frame_sim:.4f} >= {FRAME_SIMILARITY_THRESHOLD}, "
                      f"sub_conf={sub_conf:.3f} < threshold, 复用上次匹配结果{C_RESET}")
            elif cache_action == 'cleared':
                print(f"      {C_RED}│ 🗑️ 缓存清除: 帧间DINOv2相似度={frame_sim:.4f} < {FRAME_SIMILARITY_THRESHOLD}, "
                      f"场景变化大{C_RESET}")
            elif cache_action == 'no_cache':
                print(f"      {C_DIM}│ ❌ 无缓存: sub_conf={sub_conf:.3f} < threshold, 无历史缓存可用{C_RESET}")



        # 重要消息
        if message and decision not in ('continue', 'miss', 'init', 'completed'):
            if '完成' in message or '重规划' in message or '强制' in message or '偏离' in message:
                print(f"      {C_DIM}│ 💬 {message}{C_RESET}")

        last_step = step_idx
        last_phase = phase
        if to_node_id:  # VPR MISS 帧的 memory_info 可能缺少 to_node_id，保留上次有效值
            last_to_node_id = to_node_id
        if to_node:
            last_to_node = to_node

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

    # 子图匹配统计
    total_sub = stat_sub_match_hits + stat_sub_match_misses
    if total_sub > 0:
        sub_hit_rate = stat_sub_match_hits / total_sub * 100
        print(f"\n  {C_BOLD}【子图匹配统计】{C_RESET}")
        print(f"  {f'匹配成功 (≥{SUB_MATCH_CONFIDENCE_THRESHOLD})':>16s}: {stat_sub_match_hits} / {total_sub} ({sub_hit_rate:.1f}%)  (不含缓存复用)")
        print(f"  {f'匹配失败 (<{SUB_MATCH_CONFIDENCE_THRESHOLD})':>16s}: {stat_sub_match_misses}")
        print(f"  {'缓存复用 (不计入)':>16s}: {stat_cache_reused}")

    # 帧间相似度 & 缓存复用统计
    total_cache_events = stat_cache_reused + stat_cache_cleared + stat_cache_no_cache
    if total_cache_events > 0:
        print(f"\n  {C_BOLD}【帧间相似度 & 缓存复用】{C_RESET}")
        print(f"  {'缓存复用次数':>16s}: {stat_cache_reused} ({C_GREEN}帧间DINOv2相似度 >= {FRAME_SIMILARITY_THRESHOLD}{C_RESET})")
        print(f"  {'缓存清除次数':>16s}: {stat_cache_cleared} ({C_RED}帧间DINOv2相似度 < {FRAME_SIMILARITY_THRESHOLD}{C_RESET})")
        print(f"  {'无缓存可用':>16s}: {stat_cache_no_cache}")
        if stat_frame_sims:
            sims_only = [s[1] for s in stat_frame_sims]
            print(f"  {'帧间相似度':>16s}: min={min(sims_only):.4f}  avg={sum(sims_only)/len(sims_only):.4f}  max={max(sims_only):.4f}")

        # 帧间相似度趋势迷你图
        if len(stat_frame_sims) > 3:
            print(f"  {'帧间相似度序列':>16s}: ", end="")
            for _, sim, action in stat_frame_sims:
                if action == 'reused':
                    print(f"{C_GREEN}{'█' if sim >= FRAME_SIMILARITY_THRESHOLD else '▄'}{C_RESET}", end="")
                elif action == 'cleared':
                    print(f"{C_RED}▁{C_RESET}", end="")
                else:
                    print(f"{C_DIM}·{C_RESET}", end="")
            print(f"  {C_DIM}(█=复用 ▁=清除 ·=无缓存){C_RESET}")

        # Lookahead 双重确认统计
    if stat_lookahead_triggered_advance > 0 or stat_vpr_held_by_lookahead > 0:
        print(f"\n  {C_BOLD}【Lookahead 双重确认】{C_RESET}")
        print(f"  {'lookahead触发advance':>18s}: {stat_lookahead_triggered_advance} 次 (VPR匹配目标 + 下一步子图匹配成功)")
        print(f"  {'VPR到了但暂缓切换':>18s}: {stat_vpr_held_by_lookahead} 次 (VPR匹配目标, 但下一步子图未成功)")
        _total_vpr_at_target = stat_lookahead_triggered_advance + stat_vpr_held_by_lookahead
        if _total_vpr_at_target > 0:
            _la_rate = stat_lookahead_triggered_advance / _total_vpr_at_target * 100
            print(f"  {'lookahead通过率':>18s}: {_la_rate:.1f}% ({stat_lookahead_triggered_advance}/{_total_vpr_at_target})")

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
        step_size = max(1, len(stat_sim_history) // 60)
        sampled = stat_sim_history[::step_size]
        chart_height = 8
        max_sim = 1.0
        min_sim = 0.0

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


# ============================================================================
# 建图模式
# ============================================================================

async def run_test_mapping():
    """完整轨迹建图测试 (建图模式).

    流程:
      1. start_mapping 切换服务端到建图模式 (每 client 独立 session)
      2. 逐帧发 4 相机 base64 → 服务端 OnlineMapperCore.process_frame
      3. stop_mapping 触发 finalize, 返回 metrics + artifacts + visualizations
    """
    print_header("🗺️  在线建图 — 完整轨迹回放测试")
    print(f"  {C_BOLD}服务器:{C_RESET}  {WS_URL}")
    print(f"  {C_BOLD}数据:{C_RESET}    {DATA_DIR}")

    timestamps = get_timestamps()
    total_frames = len(timestamps)
    print(f"  {C_BOLD}总帧数:{C_RESET}  {total_frames}")

    if total_frames < 2:
        print(f"\n{C_RED}❌ 测试数据不足 ({total_frames} 帧), 至少需要 2 帧{C_RESET}")
        sys.exit(1)

    try:
        import websockets
    except ImportError:
        print(f"\n{C_RED}❌ 需要安装 websockets: pip install websockets{C_RESET}")
        sys.exit(1)

    print(f"\n{C_CYAN}🔗 连接 {WS_URL}...{C_RESET}")
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

    # --- 1. 启动建图会话 ---
    start_resp = await send_command(ws, 'start_mapping')
    if start_resp.get('status') != 'success':
        print(f"{C_RED}❌ start_mapping 失败: {start_resp.get('message')}{C_RESET}")
        await ws.close(); sys.exit(1)
    output_dir = start_resp.get('output_dir', '?')
    print(f"  {C_BOLD}会话输出:{C_RESET} {C_DIM}{output_dir}{C_RESET}")

    # --- 2. 帧流输入 ---
    print_header(f"▶️  建图: 喂入 {total_frames} 帧")
    print(f"\n{C_BOLD}{'seq':>4s} │ {'frame':>5s} │ {'ts':>10s} │ {'KF':^3s} │ "
          f"{'reason':^12s} │ {'VPR sim':>8s} │ {'info_gain':>10s} │ "
          f"{'category':^18s} │ {'name':<14s} │ {'nodes':>5s} │ "
          f"{'loops':>5s} │ {'dt':>7s}{C_RESET}")
    print_separator('─')

    start_time = time.time()
    stat_kf_triggered = 0
    stat_kf_by_cat = defaultdict(int)
    stat_reject = 0
    stat_reasons = defaultdict(int)
    stat_latencies = []

    last_status = {}
    for seq, ts in enumerate(timestamps):
        imgs = load_frame(ts)
        if 'camera_1' not in imgs or 'camera_2' not in imgs \
                or 'camera_3' not in imgs or 'camera_4' not in imgs:
            print(f"{seq:4d} │ skip ts={ts} (missing cameras: {list(imgs)})")
            continue

        t0 = time.time()
        # task 字段在建图模式下被忽略, 填 placeholder 即可
        resp = await send_frame(ws, "mapping", imgs, pts=int(ts))
        dt_ms = int((time.time() - t0) * 1000)
        stat_latencies.append(dt_ms)

        log = resp.get('log', {}) or {}
        mapping_status = resp.get('mapping', {}) or {}
        last_status = mapping_status

        fidx = log.get('frame_idx', seq)
        triggered = log.get('keyframe', False)
        reason = log.get('reason', '') or ''
        sim = log.get('vpr_sim_to_last')
        ig = log.get('info_gain') or 0.0
        cd = log.get('category_decision') or {}
        cat = cd.get('category', '') or ''
        name = cd.get('name', '') or ''

        # 统计
        if triggered:
            stat_kf_triggered += 1
            stat_reasons[reason] += 1
            if cat == 'reject':
                stat_reject += 1
            elif cat:
                stat_kf_by_cat[cat] += 1

        # 颜色标识
        kf_mark = f"{C_GREEN}KF{C_RESET}" if triggered else f"{C_DIM}──{C_RESET}"
        if cat == 'reject':
            cat_s = f"{C_RED}{cat:^18s}{C_RESET}"
        elif cat:
            cat_s = f"{C_GREEN}{cat:^18s}{C_RESET}"
        else:
            cat_s = f"{C_DIM}{'─':^18s}{C_RESET}"
        sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else "  ─  "
        nodes = mapping_status.get('n_nodes', 0)
        loops = mapping_status.get('n_loop_closures', 0)

        print(f"{seq:4d} │ {fidx:5d} │ {str(ts):>10s} │ {kf_mark:^3s} │ "
              f"{reason:^12s} │ {sim_s:>8s} │ {ig:>10.5f} │ "
              f"{cat_s} │ {name[:14]:<14s} │ {nodes:>5d} │ "
              f"{loops:>5d} │ {dt_ms:>5d}ms")

    feed_elapsed = time.time() - start_time

    # --- 3. finalize ---
    print_header("🏁 触发 stop_mapping → finalize")
    stop_resp = await send_command_with_timeout(ws, 'stop_mapping', timeout=180)
    summary = stop_resp.get('summary') or {}
    metrics = summary.get('metrics') or {}
    artifacts = summary.get('artifacts') or {}
    visuals = summary.get('visualizations') or {}
    finalize_s = summary.get('finalize_seconds', 0)

    # --- 4. 统计输出 ---
    print_header("📊 建图统计报告")
    print(f"\n  {C_BOLD}【基本】{C_RESET}")
    print(f"  {'总帧数':>14s}: {total_frames}")
    print(f"  {'发送帧数':>14s}: {len(stat_latencies)}")
    print(f"  {'喂帧总耗时':>14s}: {feed_elapsed:.1f}s  ({feed_elapsed/max(len(stat_latencies),1)*1000:.0f}ms/帧)")
    print(f"  {'finalize 耗时':>14s}: {finalize_s:.1f}s")
    if stat_latencies:
        lats = sorted(stat_latencies)
        p50 = lats[len(lats)//2]
        p90 = lats[int(len(lats) * 0.9)]
        print(f"  {'延迟 P50/P90':>14s}: {p50}ms / {p90}ms  (max={max(lats)}ms)")

    print(f"\n  {C_BOLD}【拓扑】{C_RESET}")
    print(f"  {'n_nodes':>14s}: {metrics.get('n_nodes', 0)}")
    print(f"  {'n_edges':>14s}: {metrics.get('n_edges', 0)}")
    print(f"  {'n_loops':>14s}: {metrics.get('n_loop_closures', 0)}")
    print(f"  {'semantic merges':>14s}: {metrics.get('n_semantic_merges', 0)}")

    print(f"\n  {C_BOLD}【关键帧】{C_RESET}")
    print(f"  {'triggered':>14s}: {metrics.get('n_keyframes_triggered', stat_kf_triggered)}")
    acc_by_cat = metrics.get('kf_accepted_by_category') or {}
    for cat, n in sorted(acc_by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:>14s}: {n}  {C_GREEN}{'█' * n}{C_RESET}")
    n_reject = metrics.get('kf_rejected_by_category', 0)
    if n_reject:
        print(f"  {'rejected':>14s}: {n_reject}  {C_RED}{'█' * n_reject}{C_RESET}")
    if stat_reasons:
        print(f"\n  {C_BOLD}【触发原因分布】{C_RESET}")
        for r, n in sorted(stat_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {r:>14s}: {n}")

    print(f"\n  {C_BOLD}【门牌 / 语义】{C_RESET}")
    print(f"  {'n_door_plates':>14s}: {metrics.get('n_door_plates', 0)}")
    pv = metrics.get('plate_voter') or {}
    if pv:
        print(f"  {'voter confirmed':>14s}: {pv.get('confirmed', 0)} / {pv.get('total_names_seen', 0)}")
        if pv.get('confirmed_names'):
            print(f"  {'confirmed':>14s}: {', '.join(pv['confirmed_names'])}")
        if pv.get('rejected_names'):
            print(f"  {'rejected':>14s}: {', '.join(pv['rejected_names'])}")
    print(f"  {'n_connections':>14s}: {metrics.get('n_connections', 0)}")
    print(f"  {'named_landmarks':>14s}: {metrics.get('n_named_landmarks', 0)}")

    runtimes = metrics.get('runtime_s') or {}
    if runtimes:
        print(f"\n  {C_BOLD}【runtime_s】{C_RESET}")
        total_runtime = runtimes.get('total', 0)
        for k in ('depth', 'vpr', 'detect', 'vo', 'plate_scan', 'name', 'total'):
            if k in runtimes:
                v = runtimes[k]
                pct = (v / max(total_runtime, 1e-6) * 100) if k != 'total' else 100.0
                print(f"  {k:>14s}: {v:>7.2f}s  ({pct:>5.1f}%)")

    print(f"\n  {C_BOLD}【产物】{C_RESET}")
    for k, v in artifacts.items():
        if v:
            print(f"  {k:>20s}: {C_DIM}{v}{C_RESET}")
    if visuals:
        print(f"\n  {C_BOLD}【可视化】{C_RESET}")
        for k, v in visuals.items():
            if v:
                print(f"  {k:>20s}: {C_DIM}{v}{C_RESET}")

    print_separator('═')
    ok = metrics.get('n_nodes', 0) > 0
    if ok:
        print(f"\n{C_GREEN}{C_BOLD}🎉 建图完成!{C_RESET}")
    else:
        print(f"\n{C_RED}{C_BOLD}💥 未产生任何节点, 建图可能失败{C_RESET}")
    print(f"{C_DIM}📋 服务端详细日志: deploy/logs/ws_proxy_with_memory.log{C_RESET}\n")

    if ws:
        await ws.close()

    if not ok:
        sys.exit(1)


async def send_command_with_timeout(ws, command, timeout=30):
    """stop_mapping 需要较长 timeout (finalize 可能几十秒)"""
    await ws.send(json.dumps({"command": command}))
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['nav', 'mapping'], default='nav',
                   help='测试模式: nav (记忆导航, 默认) | mapping (在线建图)')
    return p.parse_args()


async def main():
    args = parse_args()
    if args.mode == 'mapping':
        await run_test_mapping()
    else:
        await run_test_nav()


if __name__ == "__main__":
    asyncio.run(main())
