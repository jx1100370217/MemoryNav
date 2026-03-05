#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternNav WebSocket代理服务（带记忆导航）

基于 ws_proxy.py，新增记忆导航能力：
1. 记忆引导: 每步首次请求返回记忆的 angle + pixel_goal
2. VPR持续验证: 每次请求用 camera_1~4 做 VPR 判断是否到达下一节点
3. 模型兜底: VPR丢失时用 InternVLA 继续推理

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
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

# 记忆导航模块
from deploy.memory_nav import (
    MemoryNavigator, MemoryBuilder,
    MemoryGraph, MemoryVPR,
    NavigationPlan, NavigationStep, VPRResult
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
global_agent = None
agent_lock = asyncio.Lock()

# 记忆导航全局实例
memory_navigator: Optional[MemoryNavigator] = None

# 记忆数据路径
MEMORY_DATA_DIR = "merged_labeled_data"
MEMORY_CACHE_PATH = "deploy/memory_nav/memory_cache"

# 默认 stitch 图像尺寸（用于 pixel_position 归一化）
DEFAULT_STITCH_WIDTH = 1024
DEFAULT_STITCH_HEIGHT = 1024


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

    # ---- 相似度趋势检测 ----
    source_sim_history: List[float] = field(default_factory=list)
    target_sim_history: List[float] = field(default_factory=list)
    deviation_count: int = 0          # 连续偏离次数
    TREND_WINDOW: int = 2             # 趋势检测滑动窗口
    MAX_DEVIATIONS: int = 5           # 最大偏离次数 → 强制 advance
    last_query_features: Optional[Dict] = None  # 最近一次提取的特征

    def reset(self):
        """重置状态"""
        self.plan = None
        self.current_step_idx = 0
        self.phase = 'idle'
        self.consecutive_misses = 0
        self.last_vpr_result = None
        self.last_task = None
        self.source_sim_history = []
        self.target_sim_history = []
        self.deviation_count = 0
        self.last_query_features = None
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
        self.source_sim_history = []
        self.target_sim_history = []
        self.deviation_count = 0
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


# ============================================================================
# InternVLA Agent 参数与初始化
# ============================================================================

class Args:
    """InternVLAN1AsyncAgent初始化参数"""
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = str(project_root / "checkpoints/InternRobotics/InternVLA-N1-DualVLN")
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 8
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.plan_step_gap = 8


def init_agent(model_path=None, device=None):
    """初始化InternVLAN1AsyncAgent"""
    args = Args()
    if model_path:
        args.model_path = model_path
    if device:
        args.device = device

    logger.info(f"正在加载模型: {args.model_path}")
    logger.info(f"使用设备: {args.device}")
    logger.info(f"图像尺寸: {args.resize_w}x{args.resize_h}")
    logger.info(f"历史帧数: {args.num_history}")

    agent = InternVLAN1AsyncAgent(args)

    logger.info("正在预热模型...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4)
    agent.reset()
    agent.step(dummy_rgb, dummy_depth, dummy_pose, "test", intrinsic=args.camera_intrinsic)
    logger.info("模型加载完成！")

    return agent


def init_memory_navigator(device: str = "cuda:0", vpr_method: str = "selavpr") -> Optional[MemoryNavigator]:
    """
    初始化记忆导航器

    Returns:
        MemoryNavigator 实例，初始化失败返回 None
    """
    try:
        logger.info("="*80)
        logger.info("[Memory] 开始初始化记忆导航模块...")

        # 创建 VPR 导航器 (支持: anyloc, megaloc, effovpr, selavpr)
        navigator = MemoryNavigator(
            vpr_method=vpr_method,
            device=device
        )
        logger.info(f"[Memory] {vpr_method.upper()} VPR 导航器已创建 (dim={navigator.feature_dim}, device={device})")

        # 尝试加载记忆数据
        logger.info(f"[Memory] 记忆缓存路径: {MEMORY_CACHE_PATH}")
        logger.info(f"[Memory] 记忆数据目录: {MEMORY_DATA_DIR}")

        navigator.load_memory(
            path=MEMORY_CACHE_PATH,
            data_dir=MEMORY_DATA_DIR
        )

        # 打印已加载的记忆信息
        if navigator.graph:
            stats = navigator.graph.get_stats()
            logger.info(f"[Memory] 记忆图加载完成: {stats['total_nodes']} 节点, {stats['total_edges']} 边")
            all_dests = navigator.get_all_destinations()
            logger.info(f"[Memory] 可用目的地 ({len(all_dests)}):")
            for nid, nname in all_dests:
                logger.info(f"  - {nid}: {nname}")
        else:
            logger.warning("[Memory] 记忆图为空")

        logger.info("[Memory] 记忆导航模块初始化完成！")
        logger.info("="*80)
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


def get_stitch_image_size(stitch_image_path: str) -> Tuple[int, int]:
    """
    获取 stitch 图像尺寸，用于 pixel_position 归一化

    Returns:
        (width, height)
    """
    try:
        if stitch_image_path and os.path.exists(stitch_image_path):
            with Image.open(stitch_image_path) as img:
                w, h = img.size
                logger.debug(f"[Memory] stitch 图像尺寸: {w}x{h} ({stitch_image_path})")
                return w, h
    except Exception as e:
        logger.warning(f"[Memory] 读取 stitch 图像尺寸失败: {e}")
    return DEFAULT_STITCH_WIDTH, DEFAULT_STITCH_HEIGHT


def build_memory_response(
    robot_id, pts, nav_state: MemoryNavState,
    vpr_result: Optional[VPRResult],
    task_status: str = "executing",
    message: str = ""
) -> dict:
    """
    构建记忆导航响应

    Args:
        robot_id: 机器人 ID
        pts: 时间戳
        nav_state: 导航状态
        vpr_result: VPR 结果
        task_status: 任务状态 ("executing" / "end")
        message: 消息

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
            "memory_active": False,
            "message": "记忆导航内部错误: 当前步骤为空"
        }

    # 像素目标 (pixel_position 在 node_position_info.json 中已归一化为 [0,1])
    px, py = step.pixel_position
    norm_x = max(0.0, min(1.0, float(px)))
    norm_y = max(0.0, min(1.0, float(py)))

    # 角度: 严格使用 edge 的 angle (绝对朝向)
    # heading_offset 仅作为参考信息返回，不参与角度计算
    heading_offset = vpr_result.heading_offset if vpr_result else 0.0
    angle = step.angle

    # 构建 memory_info
    memory_info = {
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
    }

    if not message:
        message = (f"记忆导航: {step.from_node_name} → {step.to_node_name} "
                   f"(步骤{nav_state.current_step_idx + 1}/{nav_state.plan.total_steps})")

    response = {
        "status": "success",
        "id": robot_id,
        "pts": pts,
        "task_status": task_status,
        "action": [[0.0, 0.0, 0.0]],  # 记忆模式下 action 不使用
        "pixel_target": [norm_x, norm_y],
        "angle": angle,
        "memory_active": True,
        "memory_info": memory_info,
        "message": message
    }

    logger.info(f"📍 [Memory] 记忆响应: pixel_target=[{norm_x:.4f}, {norm_y:.4f}], "
                f"angle={angle:.2f}° (edge={step.angle:.2f}°, offset={heading_offset:.2f}° 仅参考), "
                f"phase={nav_state.phase}, step={nav_state.current_step_idx + 1}/{nav_state.plan.total_steps}")

    return response


# ============================================================================
# 核心推理函数 (带记忆导航)
# ============================================================================

async def process_inference_with_memory(message_data, session_state, agent,
                                         navigator: Optional[MemoryNavigator],
                                         nav_state: MemoryNavState,
                                         memory_enabled: bool):
    """
    处理推理请求（带记忆导航能力）

    三层导航策略:
    1. 记忆引导: 每步首次请求返回记忆的 angle + pixel_goal
    2. VPR持续验证: 每次请求用 camera_1~4 做 VPR 判断是否到达下一节点
    3. 模型兜底: VPR丢失时用 InternVLA 继续推理

    Args:
        message_data: 消息数据
        session_state: 会话状态
        agent: InternVLAN1AsyncAgent实例
        navigator: MemoryNavigator实例
        nav_state: MemoryNavState 状态机
        memory_enabled: 是否启用记忆导航

    Returns:
        dict: 推理结果
    """
    try:
        logger.info(f"[MemoryProxy] 开始处理推理请求 (memory_enabled={memory_enabled})")

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
                "pixel_target": None, "memory_active": False,
                "message": "缺少必要字段: task"
            }

        if 'images' not in message_data or 'front_1' not in message_data.get('images', {}):
            return {
                "status": "error", "id": robot_id, "pts": pts,
                "task_status": "end", "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None, "memory_active": False,
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
                "pixel_target": None, "memory_active": False,
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
                    "pixel_target": None, "memory_active": False,
                    "message": "首次请求时task不能为空"
                }

        # ================================================================
        # 检测 task 变化 → 清空历史 + 重置记忆状态
        # ================================================================
        current_task = instruction
        previous_task = session_state.get('last_task')

        if previous_task is not None and current_task != previous_task:
            logger.info(f"🔄 task 变化: '{previous_task}' → '{current_task}'")
            logger.info(f"🧹 清空 agent 历史 + 重置记忆导航状态")
            async with agent_lock:
                agent.reset()
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
                "pixel_target": None, "memory_active": False,
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
                "pixel_target": None, "memory_active": False,
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

            logger.info(f"🧠 [Memory] 活跃计划: 步骤 {nav_state.current_step_idx + 1}/{nav_state.plan.total_steps}, "
                        f"{step.from_node_name}({source_node_id}) → {step.to_node_name}({target_node_id}), "
                        f"phase={nav_state.phase}, misses={nav_state.consecutive_misses}")

            # ---- 无论 VPR 成功与否，都记录 source/target 相似度（为趋势检测积累数据）----
            if nav_state.last_query_features and navigator.vpr:
                try:
                    _src_sim = navigator.vpr.get_node_similarity(
                        nav_state.last_query_features, source_node_id)
                    _tgt_sim = navigator.vpr.get_node_similarity(
                        nav_state.last_query_features, target_node_id)
                    nav_state.source_sim_history.append(_src_sim)
                    nav_state.target_sim_history.append(_tgt_sim)
                    logger.info(f"📊 [Memory] 相似度记录: source({step.from_node_name})={_src_sim:.4f}, "
                                f"target({step.to_node_name})={_tgt_sim:.4f}, "
                                f"history_len={len(nav_state.source_sim_history)}")
                except Exception as e:
                    logger.warning(f"[Memory] 记录相似度失败: {e}")

            if vpr_result is not None:
                matched_id = vpr_result.matched_node_id

                # ---- Case A: VPR 匹配到目标节点 → advance ----
                if matched_id == target_node_id:
                    logger.info(f"✅ [Memory] VPR 匹配到目标节点 {target_node_id}! 前进到下一步")
                    has_next = nav_state.advance()

                    if has_next:
                        nav_state.phase = 'step_init'
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        resp = build_memory_response(robot_id, pts, nav_state, vpr_result)
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
                            "angle": None,
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

                # ---- Case B: VPR 匹配到当前路径中更后面的节点 → 跳步 ----
                elif matched_id in nav_state.plan.path:
                    matched_path_idx = nav_state.plan.path.index(matched_id)
                    # 当前步骤的 from_node 在 path 中的位置
                    current_from_path_idx = nav_state.plan.path.index(source_node_id) if source_node_id in nav_state.plan.path else -1

                    if matched_path_idx > current_from_path_idx + 1:
                        # 跳过中间节点
                        # 找到对应的 step_idx
                        new_step_idx = matched_path_idx - 1  # path[i] → path[i+1] 对应 step[i]
                        if new_step_idx < len(nav_state.plan.steps):
                            logger.info(f"⏩ [Memory] VPR 匹配到路径中的后续节点 {matched_id} "
                                        f"(path_idx={matched_path_idx}), 跳到步骤 {new_step_idx}")
                            nav_state.current_step_idx = new_step_idx
                            nav_state.consecutive_misses = 0
                            nav_state.phase = 'step_init'

                            # 检查是否已是最后一个节点（终点）
                            if matched_id == nav_state.plan.goal_node_id:
                                nav_state.phase = 'completed'
                                session_state['request_count'] += 1
                                session_state['last_instruction'] = instruction
                                session_state['last_task'] = current_task
                                resp = build_memory_response(
                                    robot_id, pts, nav_state, vpr_result,
                                    task_status="end",
                                    message=f"记忆导航完成！已到达 {nav_state.plan.goal_node_name}"
                                )
                                logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                                return resp

                            session_state['request_count'] += 1
                            session_state['last_instruction'] = instruction
                            session_state['last_task'] = current_task
                            resp = build_memory_response(robot_id, pts, nav_state, vpr_result)
                            logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                            return resp

                    # VPR 匹配到源节点或之前的节点 → 继续当前步骤
                    logger.info(f"🔄 [Memory] VPR 匹配到路径中的节点 {matched_id} "
                                f"(path_idx={matched_path_idx}), 继续当前步骤")
                    nav_state.consecutive_misses = 0
                    nav_state.phase = 'verifying'
                    session_state['request_count'] += 1
                    session_state['last_instruction'] = instruction
                    session_state['last_task'] = current_task
                    resp = build_memory_response(robot_id, pts, nav_state, vpr_result)
                    logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                    return resp

                # ---- Case C: VPR 匹配到不在路径中的节点 → 也当作"已知位置" ----
                else:
                    logger.info(f"🔀 [Memory] VPR 匹配到路径外节点 {matched_id} "
                                f"({vpr_result.matched_node_name}), "
                                f"sim={vpr_result.similarity:.4f}")
                    # 尝试从当前位置重新规划
                    if vpr_result.confidence >= 0.8:
                        logger.info(f"🔄 [Memory] 高置信度匹配到路径外节点，尝试重新规划...")
                        try:
                            new_plan = await asyncio.to_thread(
                                navigator.plan_navigation,
                                nav_state.plan.goal_node_id,
                                matched_id
                            )
                            if new_plan.success and new_plan.total_steps > 0:
                                logger.info(f"✅ [Memory] 重新规划成功: {' → '.join(new_plan.path)}")
                                nav_state.plan = new_plan
                                nav_state.current_step_idx = 0
                                nav_state.consecutive_misses = 0
                                nav_state.phase = 'step_init'
                                session_state['request_count'] += 1
                                session_state['last_instruction'] = instruction
                                session_state['last_task'] = current_task
                                resp = build_memory_response(robot_id, pts, nav_state, vpr_result,
                                    message=f"记忆导航重规划: {new_plan.start_node_name} → {new_plan.goal_node_name}")
                                logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                                return resp
                        except Exception as e:
                            logger.warning(f"[Memory] 重新规划失败: {e}")

                    # 否则继续返回当前步骤的记忆响应
                    nav_state.phase = 'verifying'
                    session_state['request_count'] += 1
                    session_state['last_instruction'] = instruction
                    session_state['last_task'] = current_task
                    resp = build_memory_response(robot_id, pts, nav_state, vpr_result)
                    logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                    return resp

            else:
                # ---- Case D: VPR 丢失 → 相似度趋势检测 ----
                nav_state.consecutive_misses += 1
                logger.info(f"❓ [Memory] VPR 丢失 ({nav_state.consecutive_misses})")

                # 相似度已在上方统一记录，直接读取最新值
                source_sim = nav_state.source_sim_history[-1] if nav_state.source_sim_history else 0.0
                target_sim = nav_state.target_sim_history[-1] if nav_state.target_sim_history else 0.0

                # 趋势检测（需要至少 TREND_WINDOW 个样本）
                tw = nav_state.TREND_WINDOW
                if len(nav_state.source_sim_history) >= tw:
                    src_hist = nav_state.source_sim_history[-tw:]
                    tgt_hist = nav_state.target_sim_history[-tw:]
                    src_trend = src_hist[-1] - src_hist[0]  # 负=下降, 正=上升
                    tgt_trend = tgt_hist[-1] - tgt_hist[0]

                    logger.info(f"📈 [Memory] 趋势: source_trend={src_trend:+.4f}, "
                                f"target_trend={tgt_trend:+.4f} "
                                f"(window={tw}, deviation_count={nav_state.deviation_count})")

                    if src_trend <= 0 and tgt_trend > 0:
                        # ---- 情况1: 远离源节点 + 接近目标节点 → 方向正确，直行 ----
                        logger.info(f"✅ [Memory] 趋势正确 (source↓ target↑)，发送 go straight")
                        nav_state.deviation_count = 0  # 重置偏离计数
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        resp = {
                            "status": "success",
                            "id": robot_id,
                            "pts": pts,
                            "task_status": "executing",
                            "action": [[1.0, 0.0, 0.0]],  # 前进 1 米
                            "pixel_target": None,
                            "angle": None,
                            "memory_active": True,
                            "memory_info": {
                                "plan_path": nav_state.plan.path if nav_state.plan else [],
                                "current_step": nav_state.current_step_idx,
                                "total_steps": nav_state.plan.total_steps if nav_state.plan else 0,
                                "from_node": step.from_node_name,
                                "to_node": step.to_node_name,
                                "phase": "trend_go_straight",
                                "consecutive_misses": nav_state.consecutive_misses,
                                "source_sim": source_sim,
                                "target_sim": target_sim,
                                "source_trend": src_trend,
                                "target_trend": tgt_trend,
                            },
                            "message": f"记忆导航: 趋势正确 (source↓ target↑)，直行前进"
                        }
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp

                    elif src_trend < 0 and tgt_trend < 0:
                        # ---- 情况2: 远离源节点 + 远离目标节点 → 偏离路线 ----
                        nav_state.deviation_count += 1
                        logger.warning(f"⚠️ [Memory] 偏离检测! source↓ target↓ "
                                       f"(deviation {nav_state.deviation_count}/{nav_state.MAX_DEVIATIONS})")

                        if nav_state.deviation_count >= nav_state.MAX_DEVIATIONS:
                            # 偏离超限 → 强制 advance
                            logger.warning(f"🚨 [Memory] 偏离超限! 强制 advance")
                            has_next = nav_state.advance()
                            session_state['request_count'] += 1
                            session_state['last_instruction'] = instruction
                            session_state['last_task'] = current_task
                            if has_next:
                                nav_state.phase = 'step_init'
                                resp = build_memory_response(
                                    robot_id, pts, nav_state, nav_state.last_vpr_result,
                                    message=f"记忆导航: 偏离超限，强制前进到下一步"
                                )
                            else:
                                nav_state.phase = 'completed'
                                resp = {
                                    "status": "success",
                                    "id": robot_id,
                                    "pts": pts,
                                    "task_status": "end",
                                    "action": [[0.0, 0.0, 0.0]],
                                    "pixel_target": None,
                                    "angle": None,
                                    "memory_active": True,
                                    "memory_info": {
                                        "plan_path": nav_state.plan.path if nav_state.plan else [],
                                        "current_step": nav_state.current_step_idx,
                                        "total_steps": nav_state.plan.total_steps if nav_state.plan else 0,
                                        "phase": "completed",
                                        "deviation_count": nav_state.deviation_count,
                                    },
                                    "message": f"🎉 记忆导航完成（偏离超限强制结束）！目标: {nav_state.plan.goal_node_name if nav_state.plan else ''}"
                                }
                            logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                            return resp

                        # 偏离但未超限 → 重新发送记忆引导 (angle + pixel_goal)，让机器人重新调整朝向
                        logger.info(f"🔄 [Memory] 偏离未超限，重发记忆引导帮助纠偏")
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        resp = build_memory_response(
                            robot_id, pts, nav_state, nav_state.last_vpr_result,
                            message=f"记忆导航: 检测到偏离 ({nav_state.deviation_count}/{nav_state.MAX_DEVIATIONS})，重发引导纠偏"
                        )
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp

                    else:
                        # ---- 其他情况 (source↑ 或 趋势不明) → 可能还在源节点附近 ----
                        # 重发记忆引导
                        logger.info(f"🔄 [Memory] 趋势不明确 (src={src_trend:+.4f}, tgt={tgt_trend:+.4f})，重发记忆引导")
                        nav_state.deviation_count = 0  # 没有偏离，重置
                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task
                        resp = build_memory_response(
                            robot_id, pts, nav_state, nav_state.last_vpr_result,
                            message=f"记忆导航: 趋势不明确，重发引导"
                        )
                        logger.info(f"📤 响应JSON: {json.dumps(resp, ensure_ascii=False, indent=2)}")
                        return resp

                # 趋势样本不足 → 走 InternVLA 兜底推理
                nav_state.phase = 'fallback'
                logger.info(f"🔄 [Memory] 趋势样本不足 ({len(nav_state.source_sim_history)}/{tw})，走 InternVLA 兜底推理")
                # 继续往下走到 InternVLA 推理

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

                        session_state['request_count'] += 1
                        session_state['last_instruction'] = instruction
                        session_state['last_task'] = current_task

                        resp = build_memory_response(
                            robot_id, pts, nav_state, vpr_result,
                            message=f"记忆导航启动: {plan.start_node_name} → {plan.goal_node_name} "
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
        # 5. 走普通 InternVLA 推理 (兜底路径)
        # ================================================================
        logger.info(f"🤖 [InternVLA] 走模型推理 (phase={nav_state.phase})")

        # 解码深度图
        if 'depth' in message_data and message_data['depth']:
            depth = decode_base64_depth(message_data['depth'])
            if depth is None:
                logger.warning("深度图解码失败，使用全零深度图")
                depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            else:
                logger.info(f"📏 深度图: shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]")
        else:
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            logger.info("未提供深度图，使用全零")

        # 解析 pose
        if 'pose' in message_data and message_data['pose']:
            pose = np.array(message_data['pose'], dtype=np.float32)
        else:
            pose = np.eye(4, dtype=np.float32)

        # 解析 intrinsic
        if 'intrinsic' in message_data and message_data['intrinsic']:
            intrinsic = np.array(message_data['intrinsic'], dtype=np.float32)
        else:
            intrinsic = np.array([
                [386.5, 0.0, 328.9, 0.0],
                [0.0, 386.5, 244.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

        look_down = message_data.get('look_down', False)

        # Agent 状态
        max_history_frames = agent.num_history if hasattr(agent, 'num_history') else 8
        current_history_count = len(agent.rgb_list) if hasattr(agent, 'rgb_list') else 0
        current_episode_idx = agent.episode_idx if hasattr(agent, 'episode_idx') else 0

        if current_episode_idx == 0 or not look_down:
            if current_episode_idx == 0:
                sampled_history_ids = []
            else:
                sampled_history_ids = np.unique(
                    np.linspace(0, current_episode_idx - 1, max_history_frames, dtype=np.int32)
                ).tolist()
        else:
            sampled_history_ids = "使用上次采样"

        resize_h = agent.resize_h if hasattr(agent, 'resize_h') else 384
        resize_w = agent.resize_w if hasattr(agent, 'resize_w') else 384

        logger.info(f"🎯 推理参数:")
        logger.info(f"  ├─ 指令: '{instruction}'")
        logger.info(f"  ├─ RGB={rgb.shape}, Depth={depth.shape}")
        logger.info(f"  ├─ 目标尺寸={resize_h}x{resize_w}, 最大历史帧={max_history_frames}")
        logger.info(f"  ├─ 已累积={current_history_count}帧, 采样={sampled_history_ids}")
        logger.info(f"  └─ look_down={look_down}, episode_idx={current_episode_idx}")

        start_time = time.time()

        async with agent_lock:
            dual_sys_output = await asyncio.to_thread(
                agent.step,
                rgb, depth, pose, instruction, intrinsic, look_down
            )

        # 检测动作5 (向下看)
        if (dual_sys_output.output_action is not None and
            len(dual_sys_output.output_action) > 0 and
            dual_sys_output.output_action[0] == 5):
            logger.info(f"🔍 检测到动作5（向下看），重新推理 look_down=True")
            async with agent_lock:
                dual_sys_output = await asyncio.to_thread(
                    agent.step,
                    rgb, depth, pose, instruction, intrinsic, look_down=True
                )

        inference_time = time.time() - start_time
        history_count_after = len(agent.rgb_list) if hasattr(agent, 'rgb_list') else 0
        episode_idx_after = agent.episode_idx if hasattr(agent, 'episode_idx') else 0
        logger.info(f"✅ 推理完成: {inference_time:.2f}秒, 历史帧={history_count_after}, "
                    f"episode_idx={episode_idx_after}")

        # 构建响应
        response = {
            "status": "success",
            "id": robot_id,
            "pts": pts,
            "task_status": "executing",
            "action": [[0.0, 0.0, 0.0]],
            "pixel_target": None,
            "memory_active": False,
            "message": ""
        }

        # 如果有记忆 fallback 状态，标记
        if nav_state.phase == 'fallback' and nav_state.plan is not None:
            response["memory_active"] = True
            step = nav_state.get_current_step()
            if step:
                response["memory_info"] = {
                    "plan_path": nav_state.plan.path,
                    "current_step": nav_state.current_step_idx,
                    "total_steps": nav_state.plan.total_steps,
                    "from_node": step.from_node_name,
                    "from_node_eng": getattr(step, 'from_node_name_eng', ''),
                    "to_node": step.to_node_name,
                    "to_node_eng": getattr(step, 'to_node_name_eng', ''),
                    "phase": "fallback",
                    "consecutive_misses": nav_state.consecutive_misses,
                }

        logger.info(f"📊 推理结果:")

        if dual_sys_output.output_action is not None:
            action_map = {0: 'STOP', 1: '↑前进', 2: '←左转', 3: '→右转', 5: '↓向下看'}
            action_str = ', '.join([f"{action_map.get(a, str(a))}" for a in dual_sys_output.output_action[:5]])
            if len(dual_sys_output.output_action) > 5:
                action_str += f", ... (共{len(dual_sys_output.output_action)}个)"
            logger.info(f"  ├─ 动作: {action_str}")
            logger.info(f"  │  原始: {dual_sys_output.output_action}")

            robot_action, task_status = convert_output_action_to_robot_action(dual_sys_output.output_action)
            response["action"] = robot_action
            response["task_status"] = task_status
            logger.info(f"  ├─ 机器人动作: {robot_action}")
            logger.info(f"  ├─ 任务状态: {task_status}")

        elif dual_sys_output.output_trajectory is not None:
            traj_shape = dual_sys_output.output_trajectory.shape
            logger.info(f"  ├─ 轨迹: shape={traj_shape}")

            robot_action = convert_trajectory_to_robot_action(dual_sys_output.output_trajectory.tolist())
            response["action"] = robot_action
            response["task_status"] = "executing"

            if len(robot_action) > 0:
                cumsum_trajectory = np.array([[pt[0], pt[1]] for pt in robot_action])
                start_point = cumsum_trajectory[0]
                end_point = cumsum_trajectory[-1]
                logger.info(f"  │  起点: [{start_point[0]:.3f}, {start_point[1]:.3f}]")
                logger.info(f"  │  终点: [{end_point[0]:.3f}, {end_point[1]:.3f}]")
                dual_sys_output.output_trajectory = cumsum_trajectory

            logger.info(f"  ├─ 轨迹点数: {len(robot_action)}")

        if dual_sys_output.output_pixel is not None:
            pixel_y_normalized = dual_sys_output.output_pixel[0] / 480.0
            pixel_x_normalized = dual_sys_output.output_pixel[1] / 640.0
            response["pixel_target"] = [pixel_x_normalized, pixel_y_normalized]
            logger.info(f"  └─ 像素目标: [y={dual_sys_output.output_pixel[0]}, x={dual_sys_output.output_pixel[1]}]")
            logger.info(f"     归一化: [x={pixel_x_normalized:.4f}, y={pixel_y_normalized:.4f}]")

        # 检测小动作自动停止 (33 个三元组)
        action_list = response["action"]
        if len(action_list) == 33:
            all_small = True
            for triplet in action_list:
                if len(triplet) >= 3:
                    if abs(triplet[0]) >= 0.5 or abs(triplet[1]) >= 0.5 or abs(triplet[2]) >= 0.5:
                        all_small = False
                        break
            if all_small:
                logger.info(f"🎯 33个小动作 → 自动停止")
                response["action"] = [[0.0, 0.0, 0.0]]
                response["task_status"] = "end"

        # 可视化
        try:
            annotated_image = annotate_image(
                idx=timestamp_str,
                image=rgb,
                instruction=instruction,
                output_action=dual_sys_output.output_action,
                trajectory=dual_sys_output.output_trajectory,
                pixel_goal=dual_sys_output.output_pixel,
                output_dir=images_dir
            )
        except Exception as e:
            logger.warning(f"生成可视化失败: {e}", exc_info=True)

        logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")

        session_state['request_count'] += 1
        session_state['last_instruction'] = instruction
        session_state['last_task'] = current_task

        return response

    except Exception as e:
        logger.error(f"推理处理异常: {e}", exc_info=True)
        return {
            "status": "error",
            "id": message_data.get('id', None),
            "pts": message_data.get('pts', None),
            "task_status": "end",
            "action": [[0.0, 0.0, 0.0]],
            "pixel_target": None,
            "memory_active": False,
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

    global global_agent, memory_navigator

    try:
        connected_clients[client_id] = {
            'websocket': websocket,
            'session_state': session_state,
            'nav_state': nav_state,
            'memory_enabled': memory_enabled,
        }
        logger.info(f"新客户端连接 [{client_id}]。连接数: {len(connected_clients)}")

        # global_agent 已在 main() 启动时初始化

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
                    async with agent_lock:
                        global_agent.reset()
                    session_state['last_instruction'] = None
                    session_state['request_count'] = 0
                    session_state['last_task'] = None
                    nav_state.reset()
                    response = {"status": "success", "message": "Agent已重置，记忆导航状态已清空"}
                    logger.info(f"Agent已重置 [{client_id}]")

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
                        data, session_state, global_agent,
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
    global memory_navigator, global_agent

    # 切换工作目录到项目根目录
    os.chdir(project_root)
    logger.info("🚀 启动 InternNav WebSocket服务器 (带记忆导航)...")
    logger.info(f"📂 工作目录: {os.getcwd()}")

    # 初始化记忆导航模块 (从 deploy/vpr_config.yaml 统一配置)
    from deploy.memory_nav.vpr_config_loader import load_vpr_config
    _vpr_cfg = load_vpr_config()
    vpr_method = _vpr_cfg['vpr_method']
    vpr_device = _vpr_cfg['device']
    logger.info(f"📊 VPR 方法: {vpr_method}, 设备: {vpr_device}")
    memory_navigator = init_memory_navigator(device=vpr_device, vpr_method=vpr_method)

    # 启动时加载 InternVLA 模型
    logger.info("正在加载 InternVLA 模型...")
    global_agent = init_agent()
    logger.info("InternVLA 模型加载完成！")
    if memory_navigator is not None:
        logger.info("✅ 记忆导航模块已就绪")
    else:
        logger.warning("⚠️ 记忆导航模块初始化失败，将仅使用 InternVLA 推理")

    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        9528,  # 端口 9528 (区别于 ws_proxy.py 的 9527)
        ping_interval=30,
        ping_timeout=10,
        max_size=50*1024*1024
    )

    logger.info("=" * 80)
    logger.info("✅ InternNav WebSocket服务器已启动 (带记忆导航)")
    logger.info(f"   端口: 9528")
    logger.info(f"   记忆导航: {'已启用' if memory_navigator else '未启用（初始化失败）'}")
    logger.info("=" * 80)
    logger.info("📚 输入格式:")
    logger.info("    - id: 机器人ID")
    logger.info("    - pts: 时间戳 (毫秒)")
    logger.info("    - task: 导航指令 (如 '去前台')")
    logger.info("    - images:")
    logger.info("        - front_1: base64编码的前置图像 (必需)")
    logger.info("        - camera_1~4: 环视相机图像 (记忆导航需要)")
    logger.info("📚 输出格式:")
    logger.info("    - status: 'success' / 'error'")
    logger.info("    - id: 机器人ID")
    logger.info("    - pts: 时间戳")
    logger.info("    - task_status: 'executing' / 'end'")
    logger.info("    - action: [[x, y, yaw], ...]")
    logger.info("    - pixel_target: [norm_x, norm_y]")
    logger.info("    - angle: 角度 (仅记忆导航模式)")
    logger.info("    - memory_active: bool")
    logger.info("    - memory_info: {...} (仅记忆导航模式)")
    logger.info("🔧 命令:")
    logger.info("    - command: 'reset' (重置Agent + 记忆状态)")
    logger.info("    - command: 'session_status' (查看会话状态)")
    logger.info("    - command: 'toggle_memory' (切换记忆导航开关)")
    logger.info("    - command: 'memory_status' (查看记忆导航详情)")
    logger.info("    - command: 'reset_memory' (仅重置记忆状态)")

    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ 服务器正在关闭...")
    except Exception as e:
        logger.error(f"❌ 服务器发生错误: {e}", exc_info=True)
