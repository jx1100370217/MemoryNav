#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternNav WebSocket代理服务（带记忆导航）

基于 ws_proxy.py，新增记忆导航能力：
1. 记忆引导: 每步首次请求返回记忆的 angle + pixel_goal
2. VPR持续验证: 每次请求用 camera_1~4 做 VPR 判断是否到达下一节点
3. 模型兜底: VPR丢失时用 Qwen3.5 打点继续推理

端口: 9528
"""

import asyncio
import websockets
import json
import logging
import logging.handlers
import base64
import io
import yaml
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# InternVLA 已移除，统一使用 Qwen3.5 打点兜底

# 记忆导航模块
from memory_nav import (
    MemoryNavigator, MemoryBuilder,
    MemoryGraph, MemoryVPR,
    NavigationPlan, NavigationStep, VPRResult
)

# 鱼眼去畸变 + 坐标变换
from memory_nav.fisheye_undistort import FisheyeUndistorter
from memory_nav.occlusion_detector import OcclusionDetector
from memory_nav.coord_transform import (
    pixel_target_to_action,
    pixel_norm_to_angle,
    estimate_distance_from_ynorm,
    get_camera_azimuth,
    DEFAULT_FOV, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    DEFAULT_CAMERA_HEIGHT, DEFAULT_PITCH_UP,
)


# ============================================================================
# 日志配置
# ============================================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = "ws_proxy_with_memory.log"


def setup_logging():
    """配置日志记录，同时输出到控制台和文件"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# 全局变量
# ============================================================================

connected_clients = {}

# 记忆导航全局实例
memory_navigator: Optional[MemoryNavigator] = None
occlusion_detector: Optional[OcclusionDetector] = None

# 鱼眼去畸变全局实例
fisheye_undistorter: Optional[FisheyeUndistorter] = None

# 记忆数据路径
MEMORY_DATA_DIR = "merged_labeled_data"
MEMORY_CACHE_PATH = "memory_nav/memory_cache"



# ============================================================================
# 记忆导航状态
# ============================================================================

@dataclass
class MemoryNavState:
    """记忆导航状态机"""
    plan: Optional[NavigationPlan] = None
    current_step_idx: int = 0
    phase: str = 'idle'          # 'idle', 'step_init', 'verifying', 'fallback', 'completed'
    consecutive_misses: int = 0
    MAX_MISSES: int = 8
    last_vpr_result: Optional[VPRResult] = None
    last_task: Optional[str] = None  # 记录发起记忆导航时的 task

    # ---- 遮挡检测 ----
    consecutive_occlusions: int = 0   # 连续遮挡次数
    last_occlusion_result: Optional[Dict] = None  # 最近一次遮挡检测结果
    last_query_features: Optional[Dict] = None  # 最近一次提取的特征
    last_good_sub_match: Optional[Dict] = None   # 上一帧成功的子图匹配结果 (confidence >= threshold)
    last_good_query_features: Optional[Dict] = None  # 上一帧匹配成功时的 VPR 特征（用于帧间相似度）
    # last_camera_image 已废弃，帧间相似度改用 last_query_features 中的 DINOv2 特征
    cache_miss_count: int = 0                     # 连续缓存未命中计数
    last_frame_similarity: Optional[float] = None     # 最近一次帧间DINOv2相似度
    last_cache_action: Optional[str] = None           # 最近一次缓存操作: accepted/reused/cleared/no_cache
    fallback_action: Optional[list] = None        # InternVLN 兜底动作
    fallback_pixel_target: Optional[list] = None  # InternVLN 兜底像素目标
    fallback_instruction: Optional[str] = None    # InternVLN 兜底指令
    fallback_camera_name: Optional[str] = None    # Qwen3.5 兜底打点相机
    next_step_sub_match: Optional[Dict] = None    # lookahead: 下一步的子图匹配结果

    def reset(self):
        """重置状态"""
        self.plan = None
        self.current_step_idx = 0
        self.phase = 'idle'
        self.consecutive_misses = 0
        self.last_vpr_result = None
        self.last_task = None
        self.consecutive_occlusions = 0
        self.last_occlusion_result = None
        self.last_query_features = None
        self.last_good_sub_match = None
        self.last_good_query_features = None
        self.cache_miss_count = 0
        self.fallback_action = None
        self.fallback_pixel_target = None
        self.fallback_instruction = None
        self.fallback_camera_name = None
        self.next_step_sub_match = None
        logger.info("[MemoryNavState] 状态已重置")

    def get_current_step(self) -> Optional[NavigationStep]:
        """获取当前步骤"""
        if self.plan is None or not self.plan.success:
            return None
        if self.current_step_idx >= len(self.plan.steps):
            return None
        return self.plan.steps[self.current_step_idx]

    def advance(self) -> bool:
        """前进到下一步，返回是否还有下一步"""
        if self.plan is None:
            return False
        self.current_step_idx += 1
        self.consecutive_misses = 0
        self.consecutive_occlusions = 0
        self.last_occlusion_result = None
        self.last_good_sub_match = None
        self.last_good_query_features = None
        self.cache_miss_count = 0
        self.fallback_action = None
        self.fallback_pixel_target = None
        self.fallback_instruction = None
        self.fallback_camera_name = None
        self.next_step_sub_match = None
        if self.current_step_idx >= len(self.plan.steps):
            self.phase = 'completed'
            logger.info(f"[MemoryNavState] 导航完成！已到达终点")
            return False
        self.phase = 'step_init'
        step = self.get_current_step()
        logger.info(f"[MemoryNavState] 前进到步骤 {self.current_step_idx}/{self.plan.total_steps}: "
                    f"{step.from_node_name} → {step.to_node_name}")
        return True

    def to_dict(self) -> Dict:
        """转换为字典"""
        step = self.get_current_step()
        return {
            'has_plan': self.plan is not None,
            'phase': self.phase,
            'current_step_idx': self.current_step_idx,
            'total_steps': self.plan.total_steps if self.plan else 0,
            'consecutive_misses': self.consecutive_misses,
            'current_step': step.to_dict() if step else None,
            'plan_path': self.plan.path if self.plan else [],
            'last_task': self.last_task,
        }

def init_memory_navigator(device: str = "cuda:0", vpr_method: str = "selavpr") -> Optional[MemoryNavigator]:
    """
    初始化记忆导航器

    Returns:
        MemoryNavigator 实例，初始化失败返回 None
    """
    try:
        logger.info("")
        logger.info("┌───────────────────────────────────────────────────────┐")
        logger.info("│            🧠 记忆导航模块初始化                      │")
        logger.info("└───────────────────────────────────────────────────────┘")

        # 创建 VPR 导航器 (支持: anyloc, megaloc, effovpr, selavpr)
        navigator = MemoryNavigator(
            vpr_method=vpr_method,
            device=device,
            qwen35_gpu="1",
            confidence_threshold=SUB_MATCH_CONFIDENCE_THRESHOLD,
        )
        logger.info(f"  ├─ VPR 模型:    {vpr_method.upper()} (dim={navigator.feature_dim}, device={device})")
        logger.info(f"  ├─ 子图匹配:    DINOv3 (device={device})")
        logger.info(f"  ├─ 缓存路径:    {MEMORY_CACHE_PATH}")
        logger.info(f"  ├─ 数据目录:    {MEMORY_DATA_DIR}")

        # 加载记忆数据
        navigator.load_memory(
            path=MEMORY_CACHE_PATH,
            data_dir=MEMORY_DATA_DIR
        )

        # 打印已加载的记忆信息
        if navigator.graph:
            stats = navigator.graph.get_stats()
            all_dests = navigator.get_all_destinations()
            logger.info(f"  ├─ 记忆图:      {stats['total_nodes']} 个节点, {stats['total_edges']} 条边")
            logger.info(f"  ├─ 可用目的地:  {len(all_dests)} 个")
            dest_names = [nname for _, nname in all_dests]
            for i in range(0, len(dest_names), 4):
                batch = dest_names[i:i+4]
                logger.info(f"  │                {', '.join(batch)}")
        else:
            logger.warning("  ├─ 记忆图:      ⚠️ 为空")

        logger.info("  └─ 状态:        ✅ 初始化完成")

        # 初始化鱼眼去畸变
        global fisheye_undistorter
        try:
            fisheye_undistorter = FisheyeUndistorter.from_yaml()
            if fisheye_undistorter.is_ready:
                logger.info(f"  📷 鱼眼去畸变:  ✅ 已加载 ({len(fisheye_undistorter.cameras)} 个相机)")
            else:
                logger.warning("  📷 鱼眼去畸变:  ⚠️ 参数文件缺失，跳过去畸变")
                fisheye_undistorter = None
        except Exception as e:
            logger.warning(f"  📷 鱼眼去畸变:  ⚠️ 初始化失败: {e}")
            fisheye_undistorter = None

        return navigator

    except Exception as e:
        logger.error(f"[Memory] 记忆导航模块初始化失败: {e}", exc_info=True)
        return None


# ============================================================================
# 图像编解码工具函数
# ============================================================================

def decode_base64_image(base64_data):
    """解码base64图像数据 → numpy array (H, W, 3) uint8"""
    try:
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"解码base64图像失败: {e}")
        return None


def decode_base64_depth(base64_data):
    """解码base64深度图数据 → numpy array (H, W) float32"""
    try:
        depth_bytes = base64.b64decode(base64_data)
        depth_image = Image.open(io.BytesIO(depth_bytes))
        depth_array = np.array(depth_image, dtype=np.float32)
        return depth_array
    except Exception as e:
        logger.error(f"解码base64深度图失败: {e}")
        return None


def encode_numpy_to_base64(array):
    """将numpy数组编码为base64"""
    try:
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"编码numpy数组失败: {e}")
        return None


# ============================================================================
# 动作转换函数
# ============================================================================

def convert_output_action_to_robot_action(output_action):
    """
    将离散动作序列转换为机器人控制命令 [x, y, yaw]

    动作编号:
        0: STOP, 1: ↑前进, 2: ←左转, 3: →右转, 5: ↓向下看(忽略)

    Returns:
        tuple: (action_list, task_status)
    """
    import math

    STEP_SIZE = 0.25
    TURN_ANGLE = math.pi / 24

    forward_count = 0
    left_turn_count = 0
    right_turn_count = 0
    has_stop = False

    for action in output_action:
        if action == 0:
            has_stop = True
        elif action == 1:
            forward_count += 1
        elif action == 2:
            left_turn_count += 1
        elif action == 3:
            right_turn_count += 1

    x = forward_count * STEP_SIZE
    y = 0.0
    yaw = (left_turn_count - right_turn_count) * TURN_ANGLE
    task_status = "end" if has_stop else "executing"

    return [[x, y, yaw]], task_status


def convert_trajectory_to_robot_action(output_trajectory):
    """
    将轨迹增量点列表转换为累积坐标的机器人控制命令格式

    输入: 33个点 [[0,0], [dx1, dy1], ...] - 增量
    输出: 33个点 [[0, 0, 0], [dx1, dy1, 0], [dx1+dx2, dy1+dy2, 0], ...] - 累积
    """
    if not output_trajectory or len(output_trajectory) == 0:
        return []

    traj_array = np.array(output_trajectory)
    delta_xy = traj_array[1:, :2] if traj_array.shape[0] > 1 else np.zeros((0, 2))

    if len(delta_xy) > 0:
        cumsum_xy = np.cumsum(delta_xy, axis=0)
    else:
        cumsum_xy = np.zeros((0, 2))

    action_list = [[0.0, 0.0, 0.0]]
    for i in range(len(cumsum_xy)):
        action_list.append([float(cumsum_xy[i, 0]), float(cumsum_xy[i, 1]), 0.0])

    return action_list


# ============================================================================
# 图像标注函数
# ============================================================================

def annotate_image(idx, image, instruction, output_action, trajectory, pixel_goal, output_dir):
    """在图像上标注推理结果"""
    try:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except:
            font = ImageFont.load_default()

        text_content = []
        text_content.append(f"Frame/PTS: {idx}")
        if output_action:
            action_map = {0: 'STOP', 1: '↑', 2: '←', 3: '→', 5: '↓'}
            action_str = ''.join([action_map.get(a, str(a)) for a in output_action[:10]])
            text_content.append(f"Actions: {action_str}")

        max_width = 0
        total_height = 0
        for line in text_content:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(line) * 8
            text_height = 20
            max_width = max(max_width, text_width)
            total_height += text_height

        padding = 10
        box_x, box_y = 10, 10
        box_width = max_width + 2 * padding
        box_height = total_height + 2 * padding
        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

        text_color = 'white'
        y_position = box_y + padding
        for line in text_content:
            draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
            y_position += 20

        image = np.array(image)

        if trajectory is not None and len(trajectory) > 0:
            img_height, img_width = image.shape[:2]
            window_size = 200
            window_margin = 0
            window_x = img_width - window_size - window_margin
            window_y = window_margin

            traj_points = []
            for point in trajectory:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    traj_points.append([float(point[0]), float(point[1])])

            if len(traj_points) > 0:
                traj_array = np.array(traj_points)
                x_coords = traj_array[:, 0]
                y_coords = traj_array[:, 1]

                fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
                fig.patch.set_alpha(0.6)
                fig.patch.set_facecolor('gray')
                ax.set_facecolor('lightgray')
                ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')
                ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
                ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')
                ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')
                ax.set_xlabel('Y (left +)', fontsize=8)
                ax.set_ylabel('X (up +)', fontsize=8)
                ax.invert_xaxis()
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(fontsize=6, loc='upper right')
                plt.tight_layout(pad=0.3)

                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                plot_img = cv2.resize(plot_img, (window_size, window_size))
                image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img

        if pixel_goal is not None and len(pixel_goal) >= 2:
            cv2.circle(image, (int(pixel_goal[1]), int(pixel_goal[0])), 5, (255, 0, 0), -1)

        image_pil = Image.fromarray(image).convert('RGB')
        output_path = os.path.join(output_dir, f'annotated_{idx}.jpg')
        image_pil.save(output_path)
        logger.info(f"已保存标注图像: {output_path}")

        return image

    except Exception as e:
        logger.error(f"图像标注失败: {e}", exc_info=True)
        return image if isinstance(image, np.ndarray) else np.array(image)


# ============================================================================
# 记忆导航辅助函数
# ============================================================================

def decode_camera_images(message_data: dict) -> Optional[Dict[str, np.ndarray]]:
    """
    从消息中解码 camera_1~4 图像

    Returns:
        {'camera_1': ndarray(BGR), ...} 或 None（如果缺少相机图）
    
    注意: 特征提取器假设输入为 BGR 格式（会内部做 BGR→RGB 转换），
    因此这里需要将 PIL 解码的 RGB 图像转换为 BGR，以保持与 memory_visualization_server.py
    中 cv2.imdecode（输出 BGR）的行为一致。
    """
    images = message_data.get('images', {})
    camera_images = {}
    for cam_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
        if cam_id in images and images[cam_id]:
            img = decode_base64_image(images[cam_id])  # PIL decode → RGB
            if img is not None:
                camera_images[cam_id] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB → BGR
    if len(camera_images) == 4:
        return camera_images
    logger.debug(f"[Memory] 相机图不完整: 收到 {list(camera_images.keys())}")
    return camera_images if camera_images else None





def _pixel_target_to_robot_action(x_norm, y_norm, camera_id,
                                  fov=DEFAULT_FOV, width=DEFAULT_WIDTH,
                                  height=DEFAULT_HEIGHT,
                                  camera_height=DEFAULT_CAMERA_HEIGHT,
                                  pitch_up=DEFAULT_PITCH_UP):
    """
    将归一化像素目标 (x_norm, y_norm) + camera_id 转换为机器人运动坐标 [x, y, 0.0]

    复用 coord_transform.pixel_target_to_action 的完整管线：
      1. x_norm → 柱面水平角 → 加相机方位角 → 全局 yaw
      2. y_norm → 柱面垂直角 → 俯仰角 → 距离估算
      3. yaw + distance → (x_forward, y_lateral)

    Returns:
        action: [x_forward, y_lateral, 0.0]
        debug: 调试信息字典
    """
    try:
        x_fwd, y_lat, debug = pixel_target_to_action(
            x_norm, y_norm, camera_id,
            camera_height=camera_height, pitch_up=pitch_up,
            fov=fov, width=width, height=height,
        )
        action = [round(x_fwd, 3), round(y_lat, 3), 0.0]
        dist = debug.get('distance', '?')
        dep = debug.get('depression_deg', '?')
        yaw_g = debug.get('yaw_global_deg', '?')
        t_ms = debug.get('elapsed_ms', '?')
        logger.info(f"🎯 [CoordTransform] camera={camera_id}, pixel=({x_norm:.3f}, {y_norm:.3f}) "
                    f"→ action={action}, dist={dist}m, dep={dep}°, yaw={yaw_g}°, t={t_ms}ms")
        return action, debug
    except Exception as e:
        logger.warning(f"[CoordTransform] 转换失败: {e}, 返回默认前进")
        return [1.0, 0.0, 0.0], {"error": str(e)}


def build_memory_response(
    robot_id, pts, nav_state: MemoryNavState,
    vpr_result: Optional[VPRResult],
    task_status: str = "executing",
    message: str = "",
    sub_image_match: dict = None,
) -> dict:
    """
    构建记忆导航响应（新方案）

    新方案返回：
    - camera_name: 目标所在相机
    - landmark_name: 注意力目标地标
    - crop_image_paths: 三级子图路径 (big/mid/small)
    - crop_image_path: crop 子图路径 (big 兼容引用)
    - sub_image_match: 子图匹配结果（实时匹配的区域百分比）

    机器人控制端根据 camera_name + 匹配区域 自行生成像素目标和避障轨迹。

    Args:
        robot_id: 机器人 ID
        pts: 时间戳
        nav_state: 导航状态
        vpr_result: VPR 结果
        task_status: 任务状态 ("executing" / "end")
        message: 消息
        sub_image_match: 子图匹配结果（可选）

    Returns:
        响应字典
    """
    step = nav_state.get_current_step()
    if step is None:
        logger.error("[Memory] build_memory_response: 当前步骤为 None")
        return {
            "status": "error",
            "id": robot_id,
            "pts": pts,
            "task_status": "end",
            "action": [[0.0, 0.0, 0.0]],
            "pixel_target": None,
            "camera_name": None,
            "message": "记忆导航内部错误: 当前步骤为空"
        }

    heading_offset = vpr_result.heading_offset if vpr_result else 0.0

    # 构建 memory_info
    memory_info = {
        "frame_similarity": nav_state.last_frame_similarity,
        "cache_action": nav_state.last_cache_action,
        "plan_path": nav_state.plan.path if nav_state.plan else [],
        "current_step": nav_state.current_step_idx,
        "total_steps": nav_state.plan.total_steps if nav_state.plan else 0,
        "from_node": step.from_node_name,
        "from_node_eng": getattr(step, 'from_node_name_eng', ''),
        "to_node": step.to_node_name,
        "to_node_eng": getattr(step, 'to_node_name_eng', ''),
        "from_node_id": step.from_node_id,
        "to_node_id": step.to_node_id,
        "heading_offset": heading_offset,
        "vpr_confidence": vpr_result.confidence if vpr_result else 0.0,
        "vpr_similarity": vpr_result.similarity if vpr_result else 0.0,
        "vpr_matched_node": vpr_result.matched_node_id if vpr_result else None,
        "phase": nav_state.phase,
        "consecutive_misses": nav_state.consecutive_misses,
        "consecutive_occlusions": nav_state.consecutive_occlusions,
        "occlusion": nav_state.last_occlusion_result,
        "lookahead_conf": (nav_state.next_step_sub_match.get('match', {}).get('confidence', 0)
                           if nav_state.next_step_sub_match else None),
        "lookahead_found": (nav_state.next_step_sub_match.get('match', {}).get('found', False)
                            if nav_state.next_step_sub_match else None),
    }

    if not message:
        message = (f"记忆导航: {step.from_node_name} → {step.to_node_name} "
                   f"(步骤{nav_state.current_step_idx + 1}/{nav_state.plan.total_steps})")

    # 当子图匹配失败且无缓存时，使用 Qwen3.5 兜底打点结果
    _use_fallback = (sub_image_match is None and task_status != "end"
                     and nav_state.fallback_action is not None)
    if _use_fallback:
        _pixel = nav_state.fallback_pixel_target
        _fallback_inst = nav_state.fallback_instruction
        # ---- 像素→机器人坐标转换 (Qwen3.5 兜底) ----
        _fb_cam = nav_state.fallback_camera_name or (step.camera_name if step else None)
        if _fb_cam in ('camera_3', 'camera_4'):
            # 侧面相机 → 原地旋转 45°
            _action = [[0.0, 0.0, 0.785]]
            logger.info(f"🔄 [Memory] Qwen3.5 兜底侧面相机 {_fb_cam}，输出旋转动作 [0,0,0.785]")
        elif _pixel and _fb_cam and len(_pixel) >= 2:
            _action_vec, _coord_debug = _pixel_target_to_robot_action(_pixel[0], _pixel[1], _fb_cam)
            _action = [_action_vec]
            memory_info["coord_transform"] = _coord_debug
        else:
            _action = nav_state.fallback_action or [[0.0, 0.0, 0.0]]
        logger.info(f"🤖 [Memory] 使用 Qwen3.5 兜底: landmark='{_fallback_inst}', "
                    f"action={_action[:3] if _action else None}..., pixel={_pixel}")
    else:
        _pixel = _extract_pixel_target(sub_image_match)
        _fallback_inst = None
        # ---- 像素→机器人坐标转换 (子图匹配) ----
        _match_cam = sub_image_match.get("camera_name") if sub_image_match else None
        if _match_cam in ('camera_3', 'camera_4'):
            # 侧面相机匹配成功 → 原地旋转 45° 让目标转到前方相机
            _action = [[0.0, 0.0, 0.785]]
            logger.info(f"🔄 [Memory] 侧面相机 {_match_cam} 匹配成功，输出旋转动作 [0,0,0.785]")
        elif _pixel and _match_cam and len(_pixel) >= 2:
            _action_vec, _coord_debug = _pixel_target_to_robot_action(_pixel[0], _pixel[1], _match_cam)
            _action = [_action_vec]
            memory_info["coord_transform"] = _coord_debug
        else:
            _action = [[0.0, 0.0, 0.0]]

    response = {
        "status": "success",
        "id": robot_id,
        "pts": pts,
        "task_status": task_status,
        "action": _action,
        "pixel_target": None,
        "camera_name": sub_image_match.get('camera_name') if sub_image_match else None,
        "landmark_name": step.landmark_name,
        "landmark_name_eng": getattr(step, 'landmark_name_eng', ''),
        "position_name_eng": getattr(step, 'to_node_name_eng', ''),
        "crop_image_paths": step.crop_image_paths,
        "crop_image_path": step.crop_image_path,
        "sub_image_match": sub_image_match,
        "fallback_instruction": _fallback_inst,
        "memory_active": True,
        "memory_info": memory_info,
        "message": message
    }

    fallback_tag = f", fallback='{_fallback_inst}'" if _use_fallback else ""
    logger.info(f"📍 [Memory] 记忆响应: camera={step.camera_name}, "
                f"landmark={step.landmark_name}, "
                f"match={'成功' if sub_image_match and sub_image_match.get('match', {}).get('found') else '未匹配'}, "
                f"phase={nav_state.phase}, step={nav_state.current_step_idx + 1}/{nav_state.plan.total_steps}"
                f"{fallback_tag}")

    return response


def do_sub_image_match(navigator, nav_state, camera_images):
    """执行子图匹配（辅助函数）
    
    使用 nav_state 中的当前步骤（而非 navigator 内部的 step index），
    确保子图匹配与 ws_proxy 的步骤状态同步。
    """
    if navigator is None or camera_images is None:
        return None
    try:
        current_step = nav_state.get_current_step() if nav_state else None
        return navigator.match_current_step(camera_images, step=current_step)
    except Exception as e:
        logger.warning(f"[Memory] 子图匹配异常: {e}")
        return None


def _extract_pixel_target(sub_image_match=None):
    """从子图匹配结果中提取 pixel_target [x, y] (归一化 0~1)，无结果返回 None"""
    if sub_image_match is None:
        return None
    match = sub_image_match.get('match')
    if match is None:
        return None
    # 必须 found=True 且坐标非零才返回
    if not match.get('found', False):
        return None
    center = match.get('center_pct')
    if center is None:
        return None
    if center.get('x', 0) == 0 and center.get('y', 0) == 0:
        return None
    return [center['x'], center['y']]


SUB_MATCH_CONFIDENCE_THRESHOLD = 0.60

FRAME_SIMILARITY_THRESHOLD = 0.70  # 帧间 DINOv2 特征相似度阈值，高于此值认为场景几乎没变


def _frame_similarity_dino(feat1, feat2, camera_name=None):
    """基于 DINOv2 VPR 特征计算帧间相似度（cosine similarity）。

    复用 VPR 流程已提取的特征，零额外推理成本。
    相比 SSIM：语义级比较，对光照变化鲁棒，对微小运动不过度敏感。

    Args:
        feat1, feat2: VPR 查询特征 {'camera_1': ndarray(4096,), ...}
        camera_name: 指定比较哪个相机，None 则取所有相机的平均相似度

    Returns:
        cosine similarity，范围 [-1, 1]，1.0 = 完全一样
    """
    if not feat1 or not feat2:
        return 0.0

    if camera_name and camera_name in feat1 and camera_name in feat2:
        a = feat1[camera_name]
        b = feat2[camera_name]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # 所有共有相机的平均相似度
    sims = []
    for cam in feat1:
        if cam in feat2:
            a, b = feat1[cam], feat2[cam]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a > 1e-8 and norm_b > 1e-8:
                sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
    return sum(sims) / len(sims) if sims else 0.0


def _cache_or_reuse_sub_match(nav_state: MemoryNavState, sub_match: dict,
                               query_features: Dict[str, np.ndarray] = None,
                               camera_name: str = None) -> dict:
    """
    缓存成功的子图匹配结果，失败时基于帧间相似度决定是否复用。

    核心逻辑：
    - 匹配成功 (conf >= SUB_MATCH_CONFIDENCE_THRESHOLD=0.65): 采纳并更新缓存（保存当前帧用于下次相似度比较）
    - 匹配失败 (conf < 0.35): 若前后帧相似度 >= FRAME_SIMILARITY_THRESHOLD=0.70，复用上一帧匹配框；否则清除缓存
    - 无缓存可用: 返回 None，让上层走 InternVLN 兜底
    """
    if sub_match is None:
        # 无匹配结果，尝试基于帧相似度复用
        if (nav_state.last_good_sub_match is not None
                and query_features is not None
                and nav_state.last_good_query_features is not None):
            sim = _frame_similarity_dino(query_features, nav_state.last_good_query_features, camera_name)
            if sim >= FRAME_SIMILARITY_THRESHOLD:
                logger.info(f"[SubMatch] 无匹配结果，帧相似度={sim:.4f} >= {FRAME_SIMILARITY_THRESHOLD}，复用缓存")
                nav_state.last_frame_similarity = sim
                nav_state.last_cache_action = 'reused'
                return nav_state.last_good_sub_match
            else:
                logger.info(f"[SubMatch] 无匹配结果，帧相似度={sim:.4f} < {FRAME_SIMILARITY_THRESHOLD}，场景已变，清除缓存")
                nav_state.last_frame_similarity = sim
                nav_state.last_cache_action = 'cleared'
                nav_state.last_good_sub_match = None
                nav_state.last_good_query_features = None
                return None
        nav_state.last_frame_similarity = None
        nav_state.last_cache_action = None
        return None

    confidence = sub_match.get('match', {}).get('confidence', 0)

    if confidence >= SUB_MATCH_CONFIDENCE_THRESHOLD:
        # 匹配成功，更新缓存 + 保存当前帧
        nav_state.last_frame_similarity = None
        nav_state.last_cache_action = 'accepted'
        nav_state.last_good_sub_match = sub_match
        nav_state.last_good_query_features = {k: v.copy() for k, v in query_features.items()} if query_features else None
        return sub_match
    else:
        # 匹配失败，基于帧相似度决定是否复用
        if (nav_state.last_good_sub_match is not None
                and query_features is not None
                and nav_state.last_good_query_features is not None):
            sim = _frame_similarity_dino(query_features, nav_state.last_good_query_features, camera_name)
            if sim >= FRAME_SIMILARITY_THRESHOLD:
                nav_state.last_frame_similarity = sim
                nav_state.last_cache_action = 'reused'
                logger.info(f"[SubMatch] confidence={confidence:.4f} < {SUB_MATCH_CONFIDENCE_THRESHOLD}，"
                            f"帧相似度={sim:.4f} >= {FRAME_SIMILARITY_THRESHOLD}，复用缓存 "
                            f"(cached_conf={nav_state.last_good_sub_match.get('match', {}).get('confidence', 0):.4f})")
                return nav_state.last_good_sub_match
            else:
                logger.info(f"[SubMatch] confidence={confidence:.4f} < {SUB_MATCH_CONFIDENCE_THRESHOLD}，"
                            f"帧相似度={sim:.4f} < {FRAME_SIMILARITY_THRESHOLD}，场景变化大，清除缓存")
                nav_state.last_frame_similarity = sim
                nav_state.last_cache_action = 'cleared'
                nav_state.last_good_sub_match = None
                nav_state.last_good_query_features = None
                return None
        else:
            nav_state.last_frame_similarity = None
            nav_state.last_cache_action = 'no_cache'
            logger.info(f"[SubMatch] confidence={confidence:.4f} < {SUB_MATCH_CONFIDENCE_THRESHOLD}，无缓存可用")
            return None  # 返回 None，让上层走 InternVLN 兜底


def visualize_sub_image_match(camera_images, sub_match_result, pts=None, cache_action=None):
    """
    在对应的 camera_x 图上标注子图匹配框和中心点，并保存到 deploy/logs/images/

    支持三种模式：
    1. 直接匹配成功 (found=True) → 绿色框 + 红色中心点
    2. 缓存复用 (cache_action='reused') → 黄色框 + 蓝色中心点
    3. 匹配失败但有 bbox → 灰色框 + 灰色中心点

    Args:
        camera_images: {'camera_1': ndarray(BGR), ...}
        sub_match_result: do_sub_image_match 返回的结果字典
        pts: 时间戳（用于文件名）
        cache_action: 缓存动作 ('accepted'/'reused'/'cleared'/'no_cache'/None)
    """
    if sub_match_result is None:
        return
    match_info = sub_match_result.get('match')
    if match_info is None:
        return

    camera_name = sub_match_result.get('camera_name')
    if not camera_name or camera_name not in camera_images:
        return

    # 需要有 bbox 数据（至少有非零坐标）
    bbox = match_info.get('bbox_pixel', {})
    has_bbox = (bbox.get('x_max', 0) > bbox.get('x_min', 0) and
                bbox.get('y_max', 0) > bbox.get('y_min', 0))
    is_found = match_info.get('found', False)
    is_reused = (cache_action == 'reused')

    if not has_bbox:
        return

    try:
        img = camera_images[camera_name].copy()
        img_h, img_w = img.shape[:2]

        images_dir = os.path.join(LOG_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)
        ts = f"{pts}" if pts is not None else f"{int(time.time() * 1000)}"

        conf = match_info.get('confidence', 0)

        if has_bbox:
            tl = match_info['top_left_pct']
            br = match_info['bottom_right_pct']
            ct = match_info['center_pct']
            x_min = int(tl['x'] * img_w)
            y_min = int(tl['y'] * img_h)
            x_max = int(br['x'] * img_w)
            y_max = int(br['y'] * img_h)
            cx = int(ct['x'] * img_w)
            cy = int(ct['y'] * img_h)
        else:
            x_min = y_min = x_max = y_max = cx = cy = 0

        if is_reused:
            # ---- 缓存复用: 黄色框 + 蓝色中心点 ----
            if has_bbox:
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            label = f"reused conf={conf:.3f}"
            cv2.putText(img, label, (max(x_min, 10), max(y_min - 8, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif is_found:
            # ---- 直接匹配成功: 绿色框 + 红色中心点 ----
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            label = f"conf={conf:.3f}"
            cv2.putText(img, label, (x_min, max(y_min - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # ---- 低置信度: 灰色框 + 灰色中心点 ----
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (128, 128, 128), 2)
            cv2.circle(img, (cx, cy), 5, (128, 128, 128), -1)
            label = f"low conf={conf:.3f}"
            cv2.putText(img, label, (x_min, max(y_min - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        mode_str = "reused" if is_reused else ("found" if is_found else "low")
        save_path = os.path.join(images_dir, f"{ts}_{camera_name}_match.jpg")
        cv2.imwrite(save_path, img)
        logger.info(f"\U0001f4be [SubImageMatch] 已保存匹配可视化: {save_path} "
                    f"(box=[{x_min},{y_min},{x_max},{y_max}], center=[{cx},{cy}], "
                    f"conf={conf:.3f}, mode={mode_str})")

    except Exception as e:
        logger.warning(f"[SubImageMatch] 可视化保存失败: {e}", exc_info=True)





def visualize_qwen35_grounding(camera_images, grounding_result, landmark_name, pts=None):
    """
    在对应的 camera 图上标注 Qwen3.5 兜底打点结果，保存到 deploy/logs/images/

    橙色十字准星 + 同心圆 + 地标名称标注（PIL 绘制中文）

    Args:
        camera_images: {'camera_1': ndarray(BGR), ...}
        grounding_result: fallback_point_grounding 返回的字典
        landmark_name: 地标名称
        pts: 时间戳（用于文件名）
    """
    if not grounding_result or not grounding_result.get("success"):
        return
    
    camera_name = grounding_result.get("camera_name", "")
    point = grounding_result.get("point")  # [x_norm, y_norm] in [0,1]
    if not camera_name or camera_name not in camera_images or not point:
        return

    try:
        img = camera_images[camera_name].copy()
        img_h, img_w = img.shape[:2]

        images_dir = os.path.join(LOG_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)
        ts = f"{pts}" if pts is not None else f"{int(time.time() * 1000)}"

        px = int(point[0] * img_w)
        py = int(point[1] * img_h)

        # 橙色十字准星 (OpenCV 画几何图形)
        color_bgr = (0, 140, 255)  # BGR orange
        cv2.line(img, (px - 30, py), (px + 30, py), color_bgr, 3)
        cv2.line(img, (px, py - 30), (px, py + 30), color_bgr, 3)
        cv2.circle(img, (px, py), 8, color_bgr, -1)
        cv2.circle(img, (px, py), 10, (255, 255, 255), 2)
        cv2.circle(img, (px, py), 24, color_bgr, 2)
        cv2.circle(img, (px, py), 40, color_bgr, 1)

        # 用 PIL 绘制中文文字
        from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        draw = PILDraw.Draw(pil_img)

        # 加载中文字体
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font_large = None
        font_small = None
        for fp in font_paths:
            if os.path.exists(fp):
                font_large = PILFont.truetype(fp, 22)
                font_small = PILFont.truetype(fp, 16)
                break
        if font_large is None:
            font_large = PILFont.load_default()
            font_small = PILFont.load_default()

        color_rgb = (255, 140, 0)  # RGB orange

        # 坐标标注
        coord_label = f"[{point[0]:.3f}, {point[1]:.3f}]"
        draw.text((px + 15, py - 20), coord_label, fill=color_rgb, font=font_small)

        # 顶部标注: Qwen3.5 + landmark (黑底)
        label = f"Qwen3.5: {landmark_name}"
        bbox = draw.textbbox((0, 0), label, font=font_large)
        text_w = bbox[2] - bbox[0] + 20
        draw.rectangle([(0, 0), (text_w, 32)], fill=(0, 0, 0))
        draw.text((10, 4), label, fill=color_rgb, font=font_large)

        # 右上角耗时
        latency = grounding_result.get("latency", 0)
        if latency:
            lat_label = f"{latency:.2f}s"
            lat_bbox = draw.textbbox((0, 0), lat_label, font=font_small)
            lat_w = lat_bbox[2] - lat_bbox[0]
            draw.text((img_w - lat_w - 10, 6), lat_label, fill=(200, 200, 200), font=font_small)

        # 转回 OpenCV BGR
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        save_path = os.path.join(images_dir, f"{ts}_{camera_name}_qwen35.jpg")
        cv2.imwrite(save_path, img)
        logger.debug(f"[Qwen35Vis] 打点可视化已保存: {save_path}")

    except Exception as e:
        logger.warning(f"[Qwen35Vis] 可视化保存失败: {e}", exc_info=True)

# ============================================================================
# 核心推理函数 (带记忆导航)
# ============================================================================

async def process_inference_with_memory(message_data, session_state,
                                         navigator: Optional[MemoryNavigator],
                                         nav_state: MemoryNavState,
                                         memory_enabled: bool):
    """
    处理推理请求（带记忆导航能力）

    三层导航策略:
    1. 记忆引导: 每步首次请求返回记忆的 angle + pixel_goal
    2. VPR持续验证: 每次请求用 camera_1~4 做 VPR 判断是否到达下一节点
    3. 模型兜底: VPR丢失时用 Qwen3.5 打点继续推理

    Args:
        message_data: 消息数据
        session_state: 会话状态
        navigator: MemoryNavigator实例
        nav_state: MemoryNavState 状态机
        memory_enabled: 是否启用记忆导航

    Returns:
        dict: 推理结果
    """
    try:
        logger.info(f"[MemoryProxy] 开始处理推理请求 (memory_enabled={memory_enabled})")
        _sub_match = None  # 子图匹配结果

        # ================================================================
        # 1. 基础解析 (同 ws_proxy.py)
        # ================================================================

        # 打印请求JSON（不含base64图像数据）
        request_log = {k: v for k, v in message_data.items() if k != 'images'}
        if 'images' in message_data:
            images_log = {}
            for img_key, img_val in message_data['images'].items():
                images_log[img_key] = f"<base64 data, length={len(img_val) if img_val else 0}>"
            request_log['images'] = images_log
        logger.info(f"📥 请求JSON: {json.dumps(request_log, ensure_ascii=False, indent=2)}")

        robot_id = message_data.get('id', None)
        pts = int(message_data['pts']) if 'pts' in message_data else None

        # 验证必要字段
        if 'task' not in message_data:
            return {
                "status": "error", "id": robot_id, "pts": pts,
                "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "缺少必要字段: task"
            }

        if 'images' not in message_data or 'front_1' not in message_data.get('images', {}):
            return {
                "status": "error", "id": robot_id, "pts": pts,
                "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "缺少必要字段: images.front_1"
            }

        instruction = message_data['task']

        # ================================================================
        # 图像处理：解码、调整尺寸、保存
        # ================================================================
        rgb_base64 = message_data['images']['front_1']
        rgb = decode_base64_image(rgb_base64)
        if rgb is None:
            return {
                "status": "error", "id": robot_id, "pts": pts,
                "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "RGB图像(images.front_1)解码失败"
            }

        logger.info(f"📸 输入RGB图像: 原始尺寸={rgb.shape}, base64长度={len(rgb_base64)} bytes")

        target_width, target_height = 640, 480
        if rgb.shape[1] != target_width or rgb.shape[0] != target_height:
            logger.info(f"📐 调整图像尺寸 {rgb.shape[1]}x{rgb.shape[0]} → {target_width}x{target_height}")
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        images_dir = os.path.join(LOG_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)

        timestamp_str = f"{pts}" if pts is not None else f"{int(time.time() * 1000)}"
        input_image_path = os.path.join(images_dir, f"{timestamp_str}_input.jpg")
        try:
            Image.fromarray(rgb).save(input_image_path)
            logger.info(f"💾 保存输入图像: {input_image_path}")
        except Exception as e:
            logger.warning(f"保存输入图像失败: {e}")

        # 保存环视相机图片
        for camera_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            if camera_id in message_data.get('images', {}):
                camera_base64 = message_data['images'][camera_id]
                if camera_base64:
                    camera_image = decode_base64_image(camera_base64)
                    if camera_image is not None:
                        camera_image_path = os.path.join(images_dir, f"{timestamp_str}_{camera_id}.jpg")
                        try:
                            Image.fromarray(camera_image).save(camera_image_path)
                        except Exception as e:
                            logger.warning(f"保存 {camera_id} 图片失败: {e}")

        # ================================================================
        # 处理 task 为 None/"None"/"none"
        # ================================================================
        if instruction is None or instruction in ["None", "none"]:
            if session_state.get('last_task') is not None:
                instruction = session_state['last_task']
                logger.info(f"📌 task 为空，延用上一次: '{instruction}'")
            else:
                return {
                    "status": "error", "id": robot_id, "pts": pts,
                    "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                    "pixel_target": None,
                    "message": "首次请求时task不能为空"
                }

        # ================================================================
        # 检测 task 变化 → 清空历史 + 重置记忆状态
        # ================================================================
        current_task = instruction
        previous_task = session_state.get('last_task')

        if previous_task is not None and current_task != previous_task:
            logger.info(f"🔄 task 变化: '{previous_task}' → '{current_task}'")
            logger.info(f"🧹 重置记忆导航状态")
            nav_state.reset()

        # ================================================================
        # 处理 STOP 指令
        # ================================================================
        if instruction in ["STOP", "stop"]:
            logger.info(f"🛑 STOP 指令")
            session_state['request_count'] += 1
            session_state['last_instruction'] = instruction
            session_state['last_task'] = instruction
            nav_state.reset()
            response = {
                "status": "success", "id": robot_id, "pts": pts,
                "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "收到STOP指令，任务结束"
            }
            logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")
            return response

        # ================================================================
        # 处理直接控制指令
        # ================================================================
        if instruction in ["turn left", "turn right", "go straight"]:
            import math
            direct_commands = {
                "turn left": [0.0, 0.0, math.pi / 12],
                "turn right": [0.0, 0.0, -math.pi / 12],
                "go straight": [1.0, 0.0, 0.0]
            }
            action = direct_commands[instruction]
            logger.info(f"⚡ 直接控制: '{instruction}' → {action}")
            session_state['request_count'] += 1
            session_state['last_instruction'] = instruction
            response = {
                "status": "success", "id": robot_id, "pts": pts,
                "task_status": "end", "action": [action],
                "pixel_target": None,
                "message": f"执行直接控制指令: {instruction}"
            }
            logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")
            return response

        # ================================================================
        # 2. 记忆导航: VPR 定位
        # ================================================================
        vpr_result: Optional[VPRResult] = None
        camera_images: Optional[Dict[str, np.ndarray]] = None

        if memory_enabled and navigator is not None:
            camera_images = decode_camera_images(message_data)
            if camera_images and len(camera_images) == 4:
                # 鱼眼去畸变（如果可用）
                if fisheye_undistorter is not None:
                    camera_images = fisheye_undistorter.undistort_batch(camera_images)
                    logger.info("📷 [Undistort] 鱼眼去畸变完成")
                logger.info(f"🧠 [Memory] 开始 VPR 定位 (4相机图就绪)")
                try:
                    vpr_result, query_features = await asyncio.to_thread(
                        navigator.locate_by_images, camera_images, True
                    )
                    nav_state.last_query_features = query_features
                    if vpr_result:
                        logger.info(f"🧠 [Memory] VPR 定位成功: "
                                    f"node={vpr_result.matched_node_id} "
                                    f"({vpr_result.matched_node_name}), "
                                    f"sim={vpr_result.similarity:.4f}, "
                                    f"conf={vpr_result.confidence:.4f}, "
                                    f"heading_offset={vpr_result.heading_offset:.1f}°, "
                                    f"best_shift={vpr_result.best_shift}")
                        nav_state.last_vpr_result = vpr_result
                    else:
                        logger.info(f"🧠 [Memory] VPR 定位失败: 无匹配节点")
                except Exception as e:
                    logger.error(f"🧠 [Memory] VPR 定位异常: {e}", exc_info=True)
            else:
                logger.debug(f"🧠 [Memory] 相机图不完整，跳过 VPR")

        # ================================================================
        # 3. 有活跃记忆计划时: VPR 验证 + 响应
        # ================================================================
        if memory_enabled and nav_state.plan is not None and nav_state.phase != 'completed':
            # ---- 打印当前位置与 from_node / to_node 的 VPR 相似度 ----
            logger.info(f"─── 📋 当前步 [{nav_state.current_step_idx + 1}/{nav_state.plan.total_steps}] ───")
            _cur_step = nav_state.get_current_step()
            if _cur_step and nav_state.last_query_features and navigator and navigator.vpr:
                _from_id = _cur_step.from_node_id
                _to_id = _cur_step.to_node_id
                _sim_from = navigator.vpr.get_node_similarity(nav_state.last_query_features, _from_id)
                _sim_to = navigator.vpr.get_node_similarity(nav_state.last_query_features, _to_id)
                logger.info(f"📊 [VPR Similarity] 当前位置 vs from_node({_cur_step.from_node_name}, {_from_id}): "
                            f"sim={_sim_from:.4f} | 当前位置 vs to_node({_cur_step.to_node_name}, {_to_id}): "
                            f"sim={_sim_to:.4f}")

            # 执行子图匹配（供后续响应使用）
            logger.info(f"── 🔍 当前步子图匹配: {_cur_step.from_node_name} → {_cur_step.to_node_name} ──" if _cur_step else "── 🔍 当前步子图匹配 ──")
            _sub_match = do_sub_image_match(memory_navigator, nav_state, camera_images) if camera_images else None
            # 帧间相似度使用 VPR 已提取的 DINOv2 特征（零额外开销）
            _cache_step = nav_state.get_current_step()
            _cache_cam_name = _cache_step.camera_name if _cache_step else None
            _sub_match = _cache_or_reuse_sub_match(nav_state, _sub_match, nav_state.last_query_features, _cache_cam_name)
            visualize_sub_image_match(camera_images, _sub_match, pts, cache_action=nav_state.last_cache_action)

            # ---- lookahead: 对下一步也做子图匹配 ----
            _next_step_idx = nav_state.current_step_idx + 1
            if _next_step_idx < len(nav_state.plan.steps) and camera_images:
                _next_step = nav_state.plan.steps[_next_step_idx]
                logger.info(f"── 🔭 下一步子图匹配: {_next_step.from_node_name} → {_next_step.to_node_name} ──")
                try:
                    _next_sub_match = navigator.match_current_step(camera_images, step=_next_step)
                    nav_state.next_step_sub_match = _next_sub_match
                    _next_found = (_next_sub_match is not None
                                   and _next_sub_match.get('match', {}).get('found', False)
                                   and _next_sub_match.get('match', {}).get('confidence', 0) >= SUB_MATCH_CONFIDENCE_THRESHOLD)
                    logger.info(f"🔭 [Lookahead] 下一步 {_next_step.from_node_name} → {_next_step.to_node_name} "
                                f"子图匹配: {'✅ 成功' if _next_found else '❌ 未匹配'} "
                                f"(conf={_next_sub_match.get('match', {}).get('confidence', 0):.4f})" if _next_sub_match else
                                f"🔭 [Lookahead] 下一步 {_next_step.from_node_name} → {_next_step.to_node_name} "
                                f"子图匹配: ❌ 无结果")
                except Exception as e:
                    nav_state.next_step_sub_match = None
                    logger.warning(f"🔭 [Lookahead] 下一步子图匹配异常: {e}")
            else:
                nav_state.next_step_sub_match = None

            # ---- Qwen3.5 兜底打点: 子图匹配失败且无缓存时 ----
            nav_state.fallback_action = None
            nav_state.fallback_pixel_target = None
            nav_state.fallback_instruction = None
            nav_state.fallback_camera_name = None
            if _sub_match is None:
                _fb_step = nav_state.get_current_step()
                _fb_landmark = getattr(_fb_step, 'landmark_name', '') if _fb_step else ''
                if _fb_landmark and navigator:
                    nav_state.fallback_instruction = _fb_landmark
                    logger.info(f"🤖 [Memory] 子图匹配无结果，启动 Qwen3.5 兜底打点: '{_fb_landmark}'")
                    try:
                        _fb_target_camera = getattr(_fb_step, 'camera_name', None)
                        _fb_start = time.time()
                        _fb_result = await asyncio.to_thread(
                            navigator.fallback_point_grounding,
                            camera_images, _fb_landmark, _fb_target_camera
                        )
                        _fb_time = time.time() - _fb_start
                        logger.info(f"🤖 [Memory] Qwen3.5 兜底打点完成: {_fb_time:.2f}s")

                        if _fb_result.get("success") and _fb_result.get("point"):
                            nav_state.fallback_pixel_target = _fb_result["point"]
                            nav_state.fallback_camera_name = _fb_result.get("camera_name")
                            nav_state.fallback_action = [[0.0, 0.0, 0.0]]  # 打点模式不输出 action
                            logger.info(f"🤖 [Memory] Qwen3.5 兜底像素: {_fb_result['point']}, "
                                       f"camera={_fb_result.get('camera_name')}")
                            # 保存打点可视化
                            visualize_qwen35_grounding(camera_images, _fb_result, _fb_landmark, pts)
                        else:
                            logger.info(f"🤖 [Memory] Qwen3.5 兜底打点失败: {_fb_result.get('error', 'unknown')}")

                    except Exception as e:
                        logger.warning(f"🤖 [Memory] Qwen3.5 兜底推理异常: {e}", exc_info=True)

            step = nav_state.get_current_step()
            if step is None:
                # 已经到最后一步之后了
                logger.info(f"🧠 [Memory] 计划已完成，切换到 completed")
                nav_state.phase = 'completed'
                session_state['request_count'] += 1
                session_state['last_instruction'] = instruction
                session_state['last_task'] = current_task
                return build_memory_response(
                    robot_id, pts, nav_state, vpr_result,
                    task_status="end",
                    message=f"记忆导航完成！已到达 {nav_state.plan.goal_node_name}"
                )

            target_node_id = step.to_node_id
            source_node_id = step.from_node_id

            logger.info(f"─── ⚡ 决策 ───")
            logger.info(f"🧠 [Memory] 活跃计划: 步骤 {nav_state.current_step_idx + 1}/{nav_state.plan.total_steps}, "
                        f"{step.from_node_name}({source_node_id}) → {step.to_node_name}({target_node_id}), "
                        f"phase={nav_state.phase}, misses={nav_state.consecutive_misses}")

            if vpr_result is not None:
                matched_id = vpr_result.matched_node_id

                # ---- Case A: VPR 匹配到目标节点 → 相似度阈值 + lookahead 双重确认后 advance ----
                VPR_ARRIVE_THRESHOLD = 0.70  # 到达 to_node 的 VPR 相似度阈值
                _sim_to_node = navigator.vpr.get_node_similarity(nav_state.last_query_features, target_node_id) if nav_state.last_query_features else 0.0
                if matched_id == target_node_id and _sim_to_node >= VPR_ARRIVE_THRESHOLD:
                    is_last_step = (nav_state.current_step_idx + 1 >= len(nav_state.plan.steps))

                    # lookahead: 检查下一步子图匹配是否成功
                    _next_match_ok = False
                    if nav_state.next_step_sub_match is not None:
                        _nm = nav_state.next_step_sub_match.get('match', {})
                        _next_match_ok = (_nm.get('found', False)
                                          and _nm.get('confidence', 0) >= SUB_MATCH_CONFIDENCE_THRESHOLD)

                    if is_last_step or _next_match_ok:
                        # 双重确认通过（最后一步无需 lookahead / 下一步子图匹配成功）
                        _reason = "最后一步" if is_last_step else f"下一步子图匹配成功(conf={nav_state.next_step_sub_match.get('match', {}).get('confidence', 0):.4f})"
                        logger.info(f"✅ [Memory] VPR 匹配到目标节点 {target_node_id} + {_reason}! 前进到下一步")
                        has_next = nav_state.advance()

                        if has_next:
                            nav_state.phase = 'step_init'
                            session_state['request_count'] += 1
                            session_state['last_instruction'] = instruction
                            session_state['last_task'] = current_task
                            _adv_msg = f"记忆导航: 步骤前进 ({_reason})"
                            resp = build_memory_response(robot_id, pts, nav_state, vpr_result, sub_image_match=_sub_match, message=_adv_msg)
                            logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                            return resp
                        else:
                            # 导航完成 — 不调用 build_memory_response (无当前步骤)
                            nav_state.phase = 'completed'
                            session_state['request_count'] += 1
                            session_state['last_instruction'] = instruction
                            session_state['last_task'] = current_task
                            resp = {
                                "status": "success",
                                "id": robot_id,
                                "pts": pts,
                                "task_status": "end",
                                "action": [[0.0, 0.0, 0.0]],
                                "pixel_target": None,
                                "camera_name": None,
                                "landmark_name": None,
                                "sub_image_match": None,
                                "memory_active": True,
                                "memory_info": {
                                    "plan_path": nav_state.plan.path if nav_state.plan else [],
                                    "current_step": nav_state.current_step_idx,
                                    "total_steps": nav_state.plan.total_steps if nav_state.plan else 0,
                                    "from_node": vpr_result.matched_node_name if vpr_result else "",
                                    "from_node_eng": getattr(vpr_result, 'matched_node_name_eng', '') if vpr_result else "",
                                    "to_node": nav_state.plan.goal_node_name if nav_state.plan else "",
                                    "to_node_eng": getattr(nav_state.plan, 'goal_node_name_eng', '') if nav_state.plan else "",
                                    "phase": "completed",
                                    "vpr_confidence": vpr_result.confidence if vpr_result else 0.0,
                                },
                                "message": f"🎉 记忆导航完成！已到达 {nav_state.plan.goal_node_name}"
                            }
                            logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                            return resp
                    else:
                        # VPR 到了目标节点，但下一步子图匹配未成功，暂不 advance
                        logger.info(f"⏳ [Memory] VPR 匹配到目标节点 {target_node_id} ({step.to_node_name}), "
                                    f"sim_to={_sim_to_node:.4f}，但下一步子图匹配未成功，暂不切换")
                        nav_state.consecutive_misses = 0
                        nav_state.phase = 'verifying'
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        _held_msg = f"记忆导航: VPR匹配到{step.to_node_name}，但下一步子图匹配未成功，暂不切换"
                        resp = build_memory_response(robot_id, pts, nav_state, vpr_result, sub_image_match=_sub_match, message=_held_msg)
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp

                # ---- Case B/C: VPR 匹配到非目标节点 / 相似度不足 → 继续当前步骤 ----
                else:
                    if matched_id == target_node_id:
                        logger.info(f"🔄 [Memory] VPR 匹配到目标节点 {matched_id} ({vpr_result.matched_node_name}), "
                                    f"但 sim_to={_sim_to_node:.4f} < {VPR_ARRIVE_THRESHOLD}, 继续当前步骤")
                    else:
                        logger.info(f"🔄 [Memory] VPR 匹配到非目标节点 {matched_id} "
                                    f"({vpr_result.matched_node_name}), 继续当前步骤")
                    nav_state.consecutive_misses = 0
                    nav_state.phase = 'verifying'
                    session_state['request_count'] += 1
                    session_state['last_instruction'] = instruction
                    session_state['last_task'] = current_task
                    resp = build_memory_response(robot_id, pts, nav_state, vpr_result, sub_image_match=_sub_match)
                    logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                    return resp

            else:
                # ---- Case D: VPR 丢失 ----
                nav_state.consecutive_misses += 1
                logger.info(f"❓ [Memory] VPR 丢失 ({nav_state.consecutive_misses})")

                # VPR 丢失但子图匹配成功 → 直接用子图匹配结果继续导航，跳过遮挡检测
                if _sub_match is not None:
                    logger.info(f"🔄 [Memory] VPR 丢失但子图匹配成功，继续当前步骤")
                    nav_state.phase = 'verifying'
                    nav_state.consecutive_occlusions = 0
                    session_state['request_count'] += 1
                    session_state['last_instruction'] = instruction
                    session_state['last_task'] = current_task
                    resp = build_memory_response(robot_id, pts, nav_state, nav_state.last_vpr_result, sub_image_match=_sub_match)
                    logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                    return resp

                # VPR 丢失 + 子图匹配也失败 → 遮挡检测
                _occ_camera = step.camera_name if step else ''
                _occ_occluded = False
                _occ_result = None

                if occlusion_detector is not None and _occ_camera and camera_images.get(_occ_camera) is not None:
                    try:
                        _occ_result = await asyncio.to_thread(
                            occlusion_detector.detect,
                            camera_images[_occ_camera],
                            _occ_camera
                        )
                        _occ_occluded = _occ_result.occluded
                        nav_state.last_occlusion_result = _occ_result.to_dict()
                        logger.info(f"🔍 [Memory] 遮挡检测: camera={_occ_camera}, "
                                    f"occluded={_occ_occluded}, "
                                    f"max_area={_occ_result.max_area_ratio:.4f}, "
                                    f"reason={_occ_result.reason}")
                    except Exception as e:
                        logger.warning(f"[Memory] 遮挡检测异常: {e}")

                if _occ_occluded:
                    # ---- 判定为遮挡 → 停止等待 ----
                    nav_state.consecutive_occlusions += 1
                    logger.info(f"🚧 [Memory] 遮挡! 原地等待 "
                                f"(连续遮挡 {nav_state.consecutive_occlusions} 次)")
                    session_state['request_count'] += 1
                    session_state['last_instruction'] = instruction
                    session_state['last_task'] = current_task
                    resp = {
                        "status": "success",
                        "id": robot_id,
                        "pts": pts,
                        "task_status": "executing",
                        "action": [[0.0, 0.0, 0.0]],
                        "pixel_target": None,
                        "camera_name": _occ_camera,
                        "landmark_name": step.landmark_name if step else None,
                        "sub_image_match": _sub_match,
                        "memory_active": True,
                        "memory_info": {
                            "plan_path": nav_state.plan.path if nav_state.plan else [],
                            "current_step": nav_state.current_step_idx,
                            "total_steps": nav_state.plan.total_steps if nav_state.plan else 0,
                            "from_node": step.from_node_name if step else "",
                            "to_node": step.to_node_name if step else "",
                            "phase": "occluded",
                            "consecutive_misses": nav_state.consecutive_misses,
                            "consecutive_occlusions": nav_state.consecutive_occlusions,
                            "occlusion": nav_state.last_occlusion_result,
                        },
                        "message": f"记忆导航: 检测到遮挡 ({_occ_result.reason if _occ_result else ''}), 原地等待"
                    }
                    logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                    return resp

                else:
                    # ---- 未遮挡 → Qwen3.5 打点继续导航 ----
                    nav_state.consecutive_occlusions = 0  # 重置连续遮挡计数
                    _fb_landmark = step.landmark_name if step else ''

                    if _fb_landmark and nav_state.fallback_pixel_target:
                        # 已有 Qwen3.5 打点结果（在上方子图匹配失败时已调用）
                        _fb_pixel = nav_state.fallback_pixel_target
                        _fb_cam = step.camera_name if step else None
                        if _fb_pixel and _fb_cam and len(_fb_pixel) >= 2:
                            _fb_vec, _fb_debug = _pixel_target_to_robot_action(_fb_pixel[0], _fb_pixel[1], _fb_cam)
                            _fb_action = [_fb_vec]
                        else:
                            _fb_action = [[1.0, 0.0, 0.0]]  # 直行 1m
                        logger.info(f"🤖 [Memory] 未遮挡，使用 Qwen3.5 打点结果导航: "
                                    f"landmark='{_fb_landmark}', pixel={_fb_pixel}")
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        resp = {
                            "status": "success",
                            "id": robot_id,
                            "pts": pts,
                            "task_status": "executing",
                            "action": _fb_action,
                            "pixel_target": _fb_pixel,
                            "camera_name": _fb_cam,
                            "landmark_name": _fb_landmark,
                            "sub_image_match": _sub_match,
                            "fallback_instruction": _fb_landmark,
                            "memory_active": True,
                            "memory_info": {
                                "plan_path": nav_state.plan.path if nav_state.plan else [],
                                "current_step": nav_state.current_step_idx,
                                "total_steps": nav_state.plan.total_steps if nav_state.plan else 0,
                                "from_node": step.from_node_name if step else "",
                                "to_node": step.to_node_name if step else "",
                                "phase": "qwen35_fallback",
                                "consecutive_misses": nav_state.consecutive_misses,
                            },
                            "message": f"记忆导航: VPR丢失+未遮挡，Qwen3.5打点导航 (landmark={_fb_landmark})"
                        }
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp
                    else:
                        # Qwen3.5 打点也失败了 → 重发记忆引导
                        logger.info(f"🔄 [Memory] VPR丢失 + 未遮挡 + Qwen3.5无结果，重发记忆引导")
                        nav_state.phase = 'fallback'
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        resp = build_memory_response(
                            robot_id, pts, nav_state, nav_state.last_vpr_result,
                            sub_image_match=_sub_match,
                            message=f"记忆导航: VPR丢失+未遮挡+打点无结果，重发引导"
                        )
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp

        # ================================================================
        # 4. 无活跃计划但 VPR 定位成功 → 尝试建立记忆导航
        # ================================================================
        if (memory_enabled and navigator is not None
                and nav_state.plan is None
                and nav_state.phase == 'idle'
                and vpr_result is not None):

            logger.info(f"🧠 [Memory] 无活跃计划, VPR 定位到 {vpr_result.matched_node_id} "
                        f"({vpr_result.matched_node_name}), 尝试匹配 task → 目的地")

            # 语义匹配目的地
            try:
                dest_node = await asyncio.to_thread(navigator.find_destination, current_task)
                if dest_node:
                    logger.info(f"🧠 [Memory] 找到目的地: {dest_node.node_id} ({dest_node.node_name})")

                    # 检查是否已在目的地
                    if dest_node.node_id == vpr_result.matched_node_id:
                        logger.info(f"✅ [Memory] 已在目的地!")
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        return {
                            "status": "success", "id": robot_id, "pts": pts,
                            "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                            "pixel_target": None,
                            "memory_active": True,
                            "memory_info": {
                                "plan_path": [vpr_result.matched_node_id],
                                "current_step": 0, "total_steps": 0,
                                "from_node": vpr_result.matched_node_name,
                                "from_node_eng": getattr(vpr_result, 'matched_node_name_eng', ''),
                                "to_node": dest_node.node_name,
                                "to_node_eng": getattr(dest_node, 'node_name_eng', ''),
                                "heading_offset": vpr_result.heading_offset,
                                "vpr_confidence": vpr_result.confidence,
                                "phase": "completed"
                            },
                            "message": f"已到达目的地: {dest_node.node_name}"
                        }

                    # 规划路径
                    plan = await asyncio.to_thread(
                        navigator.plan_navigation,
                        dest_node,
                        vpr_result.matched_node_id
                    )

                    if plan.success and plan.total_steps > 0:
                        logger.info(f"✅ [Memory] 路径规划成功: {' → '.join(plan.path)} "
                                    f"({plan.total_steps} 步)")
                        nav_state.plan = plan
                        nav_state.current_step_idx = 0
                        nav_state.phase = 'step_init'
                        nav_state.consecutive_misses = 0
                        nav_state.last_task = current_task

                        # 首次规划成功，执行子图匹配
                        _sub_match = do_sub_image_match(memory_navigator, nav_state, camera_images) if camera_images else None
                        _sub_match = _cache_or_reuse_sub_match(nav_state, _sub_match, nav_state.last_query_features)
                        visualize_sub_image_match(camera_images, _sub_match, pts, cache_action=nav_state.last_cache_action)

                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task

                        resp = build_memory_response(
                            robot_id, pts, nav_state, vpr_result,
                            sub_image_match=_sub_match, message=f"记忆导航启动: {plan.start_node_name} → {plan.goal_node_name} "
                                    f"({plan.total_steps}步)"
                        )
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp
                    else:
                        logger.warning(f"🧠 [Memory] 路径规划失败: {plan.message}")
                else:
                    logger.info(f"🧠 [Memory] 未找到匹配目的地: '{current_task}'")
            except Exception as e:
                logger.error(f"🧠 [Memory] 目的地匹配/规划异常: {e}", exc_info=True)

        # ================================================================
        # 5. Qwen3.5 打点兜底 (记忆导航未匹配到路径时)
        # ================================================================
        logger.info(f"🤖 [Qwen3.5] 记忆导航未匹配，走 Qwen3.5 打点兜底 (phase={nav_state.phase})")

        # 尝试用 Qwen3.5 对 task 进行打点
        qwen_response = {
            "status": "success",
            "id": robot_id,
            "pts": pts,
            "task_status": "executing",
            "action": [[0.0, 0.0, 0.0]],
            "pixel_target": None,
            "message": ""
        }

        if navigator and camera_images:
            try:
                _fb_landmark = current_task  # 直接用 task 作为 landmark
                logger.info(f"🤖 [Qwen3.5] 启动打点: '{_fb_landmark}'")
                _fb_start = time.time()
                _fb_result = await asyncio.to_thread(
                    navigator.fallback_point_grounding,
                    camera_images, _fb_landmark, None
                )
                _fb_time = time.time() - _fb_start
                logger.info(f"🤖 [Qwen3.5] 打点完成: {_fb_time:.2f}s")

                if _fb_result.get("success") and _fb_result.get("point"):
                    _fb_cam = _fb_result.get("camera_name", "")
                    _fb_point = _fb_result["point"]
                    logger.info(f"🤖 [Qwen3.5] 打点成功: pixel={_fb_point}, camera={_fb_cam}")

                    # 保存打点可视化
                    visualize_qwen35_grounding(camera_images, _fb_result, _fb_landmark, pts)

                    # 侧面相机输出旋转
                    if _fb_cam in ('camera_2', 'camera_4'):
                        _rot = 0.785 if _fb_cam == 'camera_2' else -0.785
                        logger.info(f"🔄 [Qwen3.5] 侧面相机 {_fb_cam}，输出旋转动作 [0,0,{_rot}]")
                        qwen_response["action"] = [[0.0, 0.0, _rot]]
                    else:
                        qwen_response["action"] = [[0.0, 0.0, 0.0]]

                    qwen_response["pixel_target"] = _fb_point
                    qwen_response["message"] = f"Qwen3.5打点兜底: landmark='{_fb_landmark}', camera={_fb_cam}"
                    qwen_response["qwen35_fallback"] = True
                else:
                    _err = _fb_result.get('error', 'unknown')
                    logger.info(f"🤖 [Qwen3.5] 打点失败: {_err}")
                    qwen_response["action"] = [[0.0, 0.0, 0.0]]
                    qwen_response["task_status"] = "executing"
                    qwen_response["message"] = f"Qwen3.5打点无结果，原地等待 (error={_err})"
            except Exception as e:
                logger.warning(f"🤖 [Qwen3.5] 打点异常: {e}", exc_info=True)
                qwen_response["action"] = [[0.0, 0.0, 0.0]]
                qwen_response["message"] = f"Qwen3.5打点异常: {e}"
        else:
            logger.warning("🤖 [Qwen3.5] navigator 或 camera_images 不可用，原地等待")
            qwen_response["message"] = "记忆导航未匹配且Qwen3.5不可用，原地等待"

        logger.info(f"📤 响应JSON: {json.dumps(qwen_response, ensure_ascii=False, indent=2)}")

        session_state['request_count'] += 1
        session_state['last_instruction'] = instruction
        session_state['last_task'] = current_task

        return qwen_response

    except Exception as e:
        logger.error(f"推理处理异常: {e}", exc_info=True)
        return {
            "status": "error",
            "id": message_data.get('id', None),
            "pts": message_data.get('pts', None),
            "task_status": "end",
            "action": [[0.0, 0.0, 0.0]],
            "pixel_target": None,
            "message": f"推理处理异常: {e}"
        }


# ============================================================================
# WebSocket 客户端处理
# ============================================================================

async def handle_client(websocket):
    """处理单个客户端连接"""
    client_id = id(websocket)
    session_state = {
        'last_instruction': None,
        'request_count': 0,
        'last_task': None
    }

    # 每个客户端独立的记忆导航状态
    nav_state = MemoryNavState()
    memory_enabled = True  # 默认启用记忆导航

    global memory_navigator

    try:
        connected_clients[client_id] = {
            'websocket': websocket,
            'session_state': session_state,
            'nav_state': nav_state,
            'memory_enabled': memory_enabled,
        }
        logger.info(f"新客户端连接 [{client_id}]。连接数: {len(connected_clients)}")


        async for message in websocket:
            try:
                data = json.loads(message)

                # 日志（不打印大块数据）
                log_data = {}
                for k, v in data.items():
                    if k in ['rgb', 'depth'] and isinstance(v, str):
                        log_data[f"{k}_length"] = len(v)
                    elif k == 'images' and isinstance(v, dict):
                        log_images = {}
                        for img_k, img_v in v.items():
                            if isinstance(img_v, str):
                                log_images[img_k] = f"<base64, len={len(img_v)}>"
                            else:
                                log_images[img_k] = img_v
                        log_data['images'] = log_images
                    else:
                        log_data[k] = v

                logger.info("=" * 150)
                logger.info(f"收到消息 [{client_id}]: {json.dumps(log_data, ensure_ascii=False)}")
                logger.info("=" * 150)

                # ---- 处理命令 ----

                command = data.get('command')

                if command == 'reset':
                    session_state['last_instruction'] = None
                    session_state['request_count'] = 0
                    session_state['last_task'] = None
                    nav_state.reset()
                    response = {"status": "success", "message": "记忆导航状态已清空"}
                    logger.info(f"记忆导航已重置 [{client_id}]")

                elif command == 'session_status':
                    response = {
                        "status": "success",
                        "message": "会话状态信息",
                        "session_info": {
                            "request_count": session_state['request_count'],
                            "last_instruction": session_state.get('last_instruction', None),
                            "last_task": session_state.get('last_task', None),
                        },
                        "memory_enabled": memory_enabled,
                        "memory_nav_state": nav_state.to_dict(),
                        "memory_navigator_status": (
                            memory_navigator.get_status() if memory_navigator else None
                        )
                    }

                elif command == 'toggle_memory':
                    memory_enabled = not memory_enabled
                    connected_clients[client_id]['memory_enabled'] = memory_enabled
                    if not memory_enabled:
                        nav_state.reset()
                    response = {
                        "status": "success",
                        "memory_enabled": memory_enabled,
                        "message": f"记忆导航已{'启用' if memory_enabled else '禁用'}"
                    }
                    logger.info(f"🧠 记忆导航: {'启用' if memory_enabled else '禁用'} [{client_id}]")

                elif command == 'memory_status':
                    response = {
                        "status": "success",
                        "memory_enabled": memory_enabled,
                        "memory_nav_state": nav_state.to_dict(),
                        "memory_navigator_status": (
                            memory_navigator.get_status() if memory_navigator else None
                        ),
                        "memory_navigator_graph_stats": (
                            memory_navigator.graph.get_stats() if memory_navigator and memory_navigator.graph else None
                        ),
                        "all_destinations": (
                            memory_navigator.get_all_destinations() if memory_navigator else []
                        ),
                        "message": "记忆导航状态"
                    }

                elif command == 'reset_memory':
                    nav_state.reset()
                    response = {
                        "status": "success",
                        "message": "记忆导航状态已重置（Agent历史保留）"
                    }
                    logger.info(f"🧠 记忆状态已重置 [{client_id}]")

                # ---- 处理推理请求 ----
                else:
                    response = await process_inference_with_memory(
                        data, session_state,
                        memory_navigator, nav_state, memory_enabled
                    )

                await websocket.send(json.dumps(response, ensure_ascii=False))
                logger.info(f"已发送响应 [{client_id}]")

            except json.JSONDecodeError:
                logger.error("无效的JSON格式", exc_info=True)
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "无效的JSON格式"
                }, ensure_ascii=False))
            except Exception as e:
                logger.error(f"处理消息时发生错误: {e}", exc_info=True)
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"处理消息时发生错误: {e}"
                }, ensure_ascii=False))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"客户端连接已关闭 [{client_id}]")
    finally:
        if client_id in connected_clients:
            del connected_clients[client_id]
        logger.info(f"客户端断开 [{client_id}]。连接数: {len(connected_clients)}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """启动WebSocket服务器（带记忆导航）"""
    global memory_navigator, occlusion_detector

    # 切换工作目录到项目根目录
    os.chdir(project_root)
    logger.info("")
    logger.info("╔═══════════════════════════════════════════════════════════╗")
    logger.info("║         🚀 MemoryNav WebSocket 服务器启动中...           ║")
    logger.info("╚═══════════════════════════════════════════════════════════╝")
    logger.info(f"  📂 工作目录: {os.getcwd()}")
    logger.info("")

    # ── 1. 加载 VPR 配置 ──
    from memory_nav.vpr_config_loader import load_vpr_config
    _vpr_cfg = load_vpr_config()
    vpr_method = _vpr_cfg['vpr_method']
    vpr_device = _vpr_cfg['device']

    # ── 2. 初始化记忆导航模块 ──
    memory_navigator = init_memory_navigator(device=vpr_device, vpr_method=vpr_method)

    # ── 3. 加载推理模型 ──
    logger.info("")
    logger.info("┌───────────────────────────────────────────────────────┐")
    logger.info("│            📦 推理模型加载                            │")
    logger.info("└───────────────────────────────────────────────────────┘")

    logger.info("  ├─ 推理模型: Qwen3.5 打点 (按需加载)")

    # ── 遮挡检测器 (YOLOv8n) ──
    try:
        occlusion_detector = OcclusionDetector(device=vpr_device)
        occlusion_detector.preload()
        logger.info(f"  ├─ 遮挡检测:    ✅ YOLOv8n (device={vpr_device})")
    except Exception as e:
        occlusion_detector = None
        logger.warning(f"  ├─ 遮挡检测:    ⚠️ 加载失败 ({e})")

    qwen35_status = "❌ 未加载"
    if memory_navigator is not None:
        try:
            memory_navigator.qwen35_grounder.start()
            qwen35_gpu = getattr(memory_navigator.qwen35_grounder, 'gpu_id', '?')
            qwen35_status = f"✅ 已加载 (GPU={qwen35_gpu})"
        except Exception as e:
            qwen35_status = f"⚠️ 加载失败，首次使用时重试 ({e})"
    logger.info(f"  └─ Qwen3.5:      {qwen35_status}")

    # ── 4. 启动 WebSocket 服务 ──
    WS_PORT = 9528
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        WS_PORT,
        open_timeout=60,
        ping_interval=30,
        ping_timeout=10,
        max_size=50*1024*1024
    )

    # ── 启动完成汇总 ──
    undist_ok = "✅ 已启用" if fisheye_undistorter else "❌ 未加载"
    memory_ok = "✅ 已启用" if memory_navigator else "❌ 初始化失败"
    logger.info("")
    logger.info("╔═══════════════════════════════════════════════════════════╗")
    logger.info("║         ✅ MemoryNav 服务器启动完成                      ║")
    logger.info("╠═══════════════════════════════════════════════════════════╣")
    logger.info(f"║  🌐 监听端口:     ws://0.0.0.0:{WS_PORT}")
    logger.info(f"║  🧠 记忆导航:     {memory_ok}")
    logger.info(f"║  🤖 兜底打点:     Qwen3.5-9B  |  {qwen35_status}")
    logger.info(f"║  🔍 子图匹配:     DINOv3 密集特征匹配")
    logger.info(f"║  📷 鱼眼去畸变:   {undist_ok}")
    logger.info(f"║  🎯 坐标转换:     pixel→robot_xy (coord_transform)")
    logger.info(f"║  📊 VPR 方法:     {vpr_method.upper()} (device={vpr_device})")
    logger.info("╠═══════════════════════════════════════════════════════════╣")
    logger.info("║  📚 API 协议                                             ║")
    logger.info("║  ┌─ 输入字段 ────────────────────────────────────────┐   ║")
    logger.info("║  │  id        机器人ID                                │   ║")
    logger.info("║  │  pts       时间戳 (ms)                             │   ║")
    logger.info("║  │  task      导航指令 (如 '去前台')                   │   ║")
    logger.info("║  │  images    front_1(必需) + camera_1~4(记忆导航)     │   ║")
    logger.info("║  └────────────────────────────────────────────────────┘   ║")
    logger.info("║  ┌─ 输出字段 ────────────────────────────────────────┐   ║")
    logger.info("║  │  status / task_status / action / memory_active     │   ║")
    logger.info("║  │  camera_name / landmark_name / sub_image_match     │   ║")
    logger.info("║  │  memory_info                                       │   ║")
    logger.info("║  └────────────────────────────────────────────────────┘   ║")
    logger.info("║  ┌─ 控制命令 ────────────────────────────────────────┐   ║")
    logger.info("║  │  reset           重置 Agent + 记忆状态             │   ║")
    logger.info("║  │  session_status  查看会话状态                       │   ║")
    logger.info("║  │  toggle_memory   切换记忆导航开关                   │   ║")
    logger.info("║  │  memory_status   查看记忆导航详情                   │   ║")
    logger.info("║  │  reset_memory    仅重置记忆状态                     │   ║")
    logger.info("║  └────────────────────────────────────────────────────┘   ║")
    logger.info("╚═══════════════════════════════════════════════════════════╝")
    logger.info("")

    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ 服务器正在关闭...")
    except Exception as e:
        logger.error(f"❌ 服务器发生错误: {e}", exc_info=True)
