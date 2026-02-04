#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternNav InternVLA-N1 WebSocket代理服务
基于InternVLAN1AsyncAgent提供实时导航推理服务
"""

import asyncio
import websockets
import json
import logging
import logging.handlers
import base64
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent


# 日志配置
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = "ws_proxy.log"


def setup_logging():
    """配置日志记录，同时输出到控制台和文件"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件处理器（滚动日志）
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logging.getLogger(__name__)


# 初始化日志
logger = setup_logging()

# 存储所有连接的客户端及其会话状态
connected_clients = {}

# 全局agent实例（单例模式）
global_agent = None
agent_lock = asyncio.Lock()


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


def annotate_image(idx, image, instruction, output_action, trajectory, pixel_goal, output_dir):
    """
    在图像上标注推理结果，包括指令、动作、轨迹和像素目标

    Args:
        idx: 帧ID或时间戳
        image: 输入图像 (H, W, 3) numpy array
        instruction: 导航指令
        output_action: 动作序列列表
        trajectory: 轨迹数组
        pixel_goal: 像素目标 [y, x]
        output_dir: 输出目录

    Returns:
        标注后的图像 numpy array
    """
    try:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        # 使用默认字体（避免字体文件不存在的问题）
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except:
            font = ImageFont.load_default()

        # 构建文本内容
        text_content = []
        text_content.append(f"Frame/PTS: {idx}")
        if output_action:
            action_map = {0: 'STOP', 1: '↑', 2: '←', 3: '→', 5: '↓'}
            action_str = ''.join([action_map.get(a, str(a)) for a in output_action[:10]])
            text_content.append(f"Actions: {action_str}")

        # 计算文本框大小
        max_width = 0
        total_height = 0
        for line in text_content:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(line) * 8  # 估算宽度
            text_height = 20
            max_width = max(max_width, text_width)
            total_height += text_height

        # 绘制文本框背景
        padding = 10
        box_x, box_y = 10, 10
        box_width = max_width + 2 * padding
        box_height = total_height + 2 * padding

        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

        # 绘制文本
        text_color = 'white'
        y_position = box_y + padding

        for line in text_content:
            draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
            y_position += 20

        image = np.array(image)

        # 绘制轨迹可视化（右上角）
        if trajectory is not None and len(trajectory) > 0:
            img_height, img_width = image.shape[:2]

            # 窗口参数
            window_size = 200
            window_margin = 0
            window_x = img_width - window_size - window_margin
            window_y = window_margin

            # 提取轨迹点
            traj_points = []
            for point in trajectory:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    traj_points.append([float(point[0]), float(point[1])])

            if len(traj_points) > 0:
                traj_array = np.array(traj_points)
                x_coords = traj_array[:, 0]
                y_coords = traj_array[:, 1]

                # 创建matplotlib图形
                fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
                fig.patch.set_alpha(0.6)
                fig.patch.set_facecolor('gray')
                ax.set_facecolor('lightgray')

                # 绘制轨迹
                ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')

                # 标记起点（绿色）和终点（红色）
                ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
                ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')

                # 标记原点
                ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')

                # 设置坐标轴
                ax.set_xlabel('Y (left +)', fontsize=8)
                ax.set_ylabel('X (up +)', fontsize=8)
                ax.invert_xaxis()
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(fontsize=6, loc='upper right')

                plt.tight_layout(pad=0.3)

                # 转换为numpy数组
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                # 调整大小并叠加到图像上
                plot_img = cv2.resize(plot_img, (window_size, window_size))
                image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img

        # 绘制像素目标（蓝色圆圈）
        if pixel_goal is not None and len(pixel_goal) >= 2:
            # pixel_goal是[y, x]格式，cv2.circle需要(x, y)格式
            cv2.circle(image, (int(pixel_goal[1]), int(pixel_goal[0])), 5, (255, 0, 0), -1)

        # 保存标注后的图像
        image_pil = Image.fromarray(image).convert('RGB')
        output_path = os.path.join(output_dir, f'annotated_{idx}.jpg')
        image_pil.save(output_path)
        logger.info(f"已保存标注图像: {output_path}")

        return image

    except Exception as e:
        logger.error(f"图像标注失败: {e}", exc_info=True)
        return image if isinstance(image, np.ndarray) else np.array(image)


def init_agent(model_path=None, device=None):
    """
    初始化InternVLAN1AsyncAgent

    Args:
        model_path: 模型路径，如果为None则使用默认路径
        device: 设备，如果为None则使用cuda:0

    Returns:
        InternVLAN1AsyncAgent实例
    """
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

    # 模型预热
    logger.info("正在预热模型...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4)
    agent.reset()
    agent.step(dummy_rgb, dummy_depth, dummy_pose, "test", intrinsic=args.camera_intrinsic)
    logger.info("模型加载完成！")

    return agent


def decode_base64_image(base64_data):
    """
    解码base64图像数据

    Args:
        base64_data: base64编码的图像数据

    Returns:
        numpy array (H, W, 3) uint8
    """
    try:
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"解码base64图像失败: {e}")
        return None


def decode_base64_depth(base64_data):
    """
    解码base64深度图数据

    Args:
        base64_data: base64编码的深度图数据

    Returns:
        numpy array (H, W) float32
    """
    try:
        depth_bytes = base64.b64decode(base64_data)
        depth_image = Image.open(io.BytesIO(depth_bytes))
        # 深度图通常是16位或32位浮点
        depth_array = np.array(depth_image, dtype=np.float32)
        return depth_array
    except Exception as e:
        logger.error(f"解码base64深度图失败: {e}")
        return None


def convert_output_action_to_robot_action(output_action):
    """
    将离散动作序列转换为机器人控制命令 [x, y, yaw]

    动作编号：
        0: STOP（停止）
        1: ↑（前进）
        2: ←（左转）
        3: →（右转）
        5: ↓（向下看，机器人不支持，忽略）

    Args:
        output_action: 离散动作序列列表，如 [3, 3, 3, 3]

    Returns:
        tuple: (action_list, task_status)
            - action_list: [[x, y, yaw]] 格式的控制命令
            - task_status: "end" 如果包含STOP动作，否则 "executing"
    """
    import math

    # 定义常量
    STEP_SIZE = 0.25  # 前进步长（米）
    TURN_ANGLE = math.pi / 24  # 每次转弯角度（弧度），约7.5度，4次=30度

    # 统计动作
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
        # action == 5 (向下看) 被忽略，不影响输出

    # 计算合并后的控制命令
    x = forward_count * STEP_SIZE
    y = 0.0
    yaw = (left_turn_count - right_turn_count) * TURN_ANGLE

    # 确定任务状态
    task_status = "end" if has_stop else "executing"

    return [[x, y, yaw]], task_status


def convert_trajectory_to_robot_action(output_trajectory):
    """
    将轨迹增量点列表转换为累积坐标的机器人控制命令格式

    参照 internnav/model/utils/vln_utils.py 中的 reconstruct_xy_from_delta 函数

    输入: 33个点 [[0,0], [dx1, dy1], [dx2, dy2], ...] - 第一个点是起点(0,0)，后续是增量
    输出: 33个点 [[0, 0, 0], [dx1, dy1, 0], [dx1+dx2, dy1+dy2, 0], ...] - 累积坐标

    转换规则:
    - 第 1 个点：(0, 0)
    - 第 2 个点：(dx1, dy1)
    - 第 3 个点：(dx1+dx2, dy1+dy2)
    - 第 n 个点：从第 1 步到第 n-1 步的所有 dx 之和及所有 dy 之和

    Args:
        output_trajectory: 轨迹增量点列表，如 [[0, 0], [dx1, dy1], [dx2, dy2], ...]

    Returns:
        list: [[0, 0, 0], [dx1, dy1, 0], [dx1+dx2, dy1+dy2, 0], ...] 格式的累积坐标
    """
    if not output_trajectory or len(output_trajectory) == 0:
        return []

    # 转换为 numpy 数组
    traj_array = np.array(output_trajectory)

    # 跳过第一个点(起点 0,0)，取后续的增量值
    delta_xy = traj_array[1:, :2] if traj_array.shape[0] > 1 else np.zeros((0, 2))

    # 计算累积和 (cumsum)
    if len(delta_xy) > 0:
        cumsum_xy = np.cumsum(delta_xy, axis=0)
    else:
        cumsum_xy = np.zeros((0, 2))

    # 构建输出：第一个点是 (0, 0)，后续点是累积值
    action_list = [[0.0, 0.0, 0.0]]  # 起点
    for i in range(len(cumsum_xy)):
        action_list.append([float(cumsum_xy[i, 0]), float(cumsum_xy[i, 1]), 0.0])

    return action_list


def encode_numpy_to_base64(array):
    """
    将numpy数组编码为base64

    Args:
        array: numpy array

    Returns:
        base64编码的字符串
    """
    try:
        # 将numpy数组转换为bytes
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"编码numpy数组失败: {e}")
        return None


async def process_inference(message_data, session_state, agent):
    """
    处理推理请求

    Args:
        message_data: 消息数据
        session_state: 会话状态
        agent: InternVLAN1AsyncAgent实例

    Returns:
        dict: 推理结果
    """
    try:
        logger.info(f"开始处理推理请求")

        # 打印请求JSON（不包含base64图像数据）
        request_log = {k: v for k, v in message_data.items() if k != 'images'}
        if 'images' in message_data:
            images_log = {}
            for img_key, img_val in message_data['images'].items():
                images_log[img_key] = f"<base64 data, length={len(img_val) if img_val else 0}>"
            request_log['images'] = images_log
        logger.info(f"📥 请求JSON: {json.dumps(request_log, ensure_ascii=False, indent=2)}")

        # 提取基本字段
        robot_id = message_data.get('id', None)
        pts = int(message_data['pts']) if 'pts' in message_data else None

        # 验证必要字段 - 从 task 获取指令（允许task为None以延用上次task）
        if 'task' not in message_data:
            return {
                "status": "error",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "缺少必要字段: task"
            }

        # 验证 images.front_1 字段
        if 'images' not in message_data or 'front_1' not in message_data.get('images', {}):
            return {
                "status": "error",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "缺少必要字段: images.front_1"
            }

        instruction = message_data['task']

        # ===== 图像处理：解码、调整尺寸、保存（所有请求都需要） =====
        # 解码RGB图像 - 从 images.front_1 获取
        rgb_base64 = message_data['images']['front_1']
        rgb = decode_base64_image(rgb_base64)
        if rgb is None:
            return {
                "status": "error",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "RGB图像(images.front_1)解码失败"
            }

        # 打印原始图像信息
        logger.info(f"📸 输入RGB图像: 原始尺寸={rgb.shape}, 数据类型={rgb.dtype}, base64长度={len(rgb_base64)} bytes")

        # 检查并调整图像尺寸为 640x480
        target_width, target_height = 640, 480
        if rgb.shape[1] != target_width or rgb.shape[0] != target_height:
            logger.info(f"📐 输入图像尺寸 {rgb.shape[1]}x{rgb.shape[0]} != {target_width}x{target_height}，进行调整")
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"✅ 图像已调整为 {target_width}x{target_height}")
        else:
            logger.info(f"✅ 图像尺寸已符合要求: {target_width}x{target_height}")

        # 创建图像保存目录
        images_dir = os.path.join(LOG_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # 保存输入RGB图像（调整后的 640x480）
        timestamp_str = f"{pts}" if pts is not None else f"{int(time.time() * 1000)}"
        input_image_path = os.path.join(images_dir, f"{timestamp_str}_input.jpg")
        try:
            Image.fromarray(rgb).save(input_image_path)
            logger.info(f"💾 保存输入图像: {input_image_path} (尺寸: {rgb.shape[1]}x{rgb.shape[0]})")
        except Exception as e:
            logger.warning(f"保存输入图像失败: {e}")

        # 保存环视相机图片 (camera_1, camera_2, camera_3, camera_4)
        for camera_id in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
            if camera_id in message_data.get('images', {}):
                camera_base64 = message_data['images'][camera_id]
                if camera_base64:  # 确保不是空字符串或None
                    camera_image = decode_base64_image(camera_base64)
                    if camera_image is not None:
                        camera_image_path = os.path.join(images_dir, f"{timestamp_str}_{camera_id}.jpg")
                        try:
                            Image.fromarray(camera_image).save(camera_image_path)
                            logger.info(f"💾 保存环视相机图片: {camera_image_path}")
                        except Exception as e:
                            logger.warning(f"保存 {camera_id} 图片失败: {e}")
                    else:
                        logger.warning(f"{camera_id} 图片解码失败，跳过保存")

        # ===== 需求1：处理task为None/"None"/"none" =====
        if instruction is None or instruction in ["None", "none"]:
            # 使用上一次的task
            if session_state.get('last_task') is not None:
                instruction = session_state['last_task']
                logger.info(f"📌 检测到task为空/None，延用上一次的task: '{instruction}'")
            else:
                # 首次请求且task为空，返回错误
                return {
                    "status": "error",
                    "id": robot_id,
                    "pts": pts,
                    "task_status": "end",
                    "action": [[0.0, 0.0, 0.0]],
                    "pixel_target": None,
                    "message": "首次请求时task不能为空"
                }

        # ===== 需求2：检测task变化，清空历史 =====
        current_task = instruction
        previous_task = session_state.get('last_task')

        if previous_task is not None and current_task != previous_task:
            logger.info(f"🔄 检测到task变化: '{previous_task}' → '{current_task}'")
            logger.info(f"🧹 清空agent历史记录，准备从头推理新任务")
            async with agent_lock:
                agent.reset()
            logger.info(f"✅ Agent历史已清空 (历史帧数={len(agent.rgb_list)}, episode_idx={agent.episode_idx})")

        # ===== 需求3：处理STOP指令 =====
        if instruction in ["STOP", "stop"]:
            logger.info(f"🛑 检测到STOP指令，直接返回停止动作")

            # 更新session_state
            session_state['request_count'] += 1
            session_state['last_instruction'] = instruction
            session_state['last_task'] = instruction

            response = {
                "status": "success",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [[0.0, 0.0, 0.0]],
                "pixel_target": None,
                "message": "收到STOP指令，任务结束"
            }

            # 打印响应JSON，保持与正常推理一致的日志格式
            logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")

            return response

        # ===== 新增需求：处理直接控制指令 =====
        if instruction in ["turn left", "turn right", "go straight"]:
            import math

            # 定义直接控制指令的映射
            direct_commands = {
                "turn left": [0.0, 0.0, math.pi / 12],      # 左转15度
                "turn right": [0.0, 0.0, -math.pi / 12],    # 右转15度
                "go straight": [1.0, 0.0, 0.0]              # 前进1米
            }

            action = direct_commands[instruction]
            logger.info(f"⚡ 检测到直接控制指令: '{instruction}'")
            logger.info(f"   控制命令: x={action[0]:.3f}, y={action[1]:.3f}, yaw={action[2]:.4f} rad ({action[2] * 180 / math.pi:.1f}°)")

            # 更新session_state（不更新last_task，保持导航任务不变）
            session_state['request_count'] += 1
            session_state['last_instruction'] = instruction

            response = {
                "status": "success",
                "id": robot_id,
                "pts": pts,
                "task_status": "end",
                "action": [action],
                "pixel_target": None,
                "message": f"执行直接控制指令: {instruction}"
            }

            # 打印响应JSON
            logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")

            return response

        # 解码深度图（如果提供）
        if 'depth' in message_data and message_data['depth']:
            depth = decode_base64_depth(message_data['depth'])
            if depth is None:
                logger.warning("深度图解码失败，使用全零深度图")
                depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            else:
                logger.info(f"📏 输入深度图: 尺寸={depth.shape}, 数据类型={depth.dtype}, 深度范围=[{depth.min():.2f}, {depth.max():.2f}]")
        else:
            # 如果没有提供深度图，使用全零深度图
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            logger.info("未提供深度图，使用全零深度图")

        # 解析pose（如果提供）
        if 'pose' in message_data and message_data['pose']:
            pose = np.array(message_data['pose'], dtype=np.float32)
        else:
            pose = np.eye(4, dtype=np.float32)

        # 解析intrinsic（如果提供）
        if 'intrinsic' in message_data and message_data['intrinsic']:
            intrinsic = np.array(message_data['intrinsic'], dtype=np.float32)
        else:
            intrinsic = np.array([
                [386.5, 0.0, 328.9, 0.0],
                [0.0, 386.5, 244.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

        # 解析look_down标志
        look_down = message_data.get('look_down', False)

        # 获取agent的历史帧配置和当前状态
        max_history_frames = agent.num_history if hasattr(agent, 'num_history') else 8
        current_history_count = len(agent.rgb_list) if hasattr(agent, 'rgb_list') else 0
        current_episode_idx = agent.episode_idx if hasattr(agent, 'episode_idx') else 0
        resize_h = agent.resize_h if hasattr(agent, 'resize_h') else 384
        resize_w = agent.resize_w if hasattr(agent, 'resize_w') else 384

        # 计算本次推理将要采样的历史帧序号（模拟agent内部的采样逻辑）
        if current_episode_idx == 0 or not look_down:
            # 第一帧或非look_down模式时，根据episode_idx采样
            if current_episode_idx == 0:
                sampled_history_ids = []
            else:
                sampled_history_ids = np.unique(np.linspace(0, current_episode_idx - 1, max_history_frames, dtype=np.int32)).tolist()
        else:
            # look_down模式时使用之前的历史帧
            sampled_history_ids = "使用上次采样"

        # if current_episode_idx == 0:
        #     sampled_history_ids = []
        # else:
        #     sampled_history_ids = np.unique(np.linspace(0, current_episode_idx - 1, max_history_frames, dtype=np.int32)).tolist()

        logger.info(f"🎯 推理参数详情:")
        logger.info(f"  ├─ 导航指令: '{instruction}'")
        logger.info(f"  ├─ 输入尺寸: RGB={rgb.shape}, Depth={depth.shape}")
        logger.info(f"  ├─ 模型配置: 目标尺寸={resize_h}x{resize_w}, 最大历史帧数={max_history_frames}")
        logger.info(f"  ├─ 历史帧状态: 已累积={current_history_count}帧, 本次采样使用={sampled_history_ids}")
        logger.info(f"  └─ 其他参数: look_down={look_down}, episode_idx={current_episode_idx}")

        # 执行推理
        start_time = time.time()

        async with agent_lock:
            # 使用asyncio.to_thread在线程池中运行阻塞的推理
            dual_sys_output = await asyncio.to_thread(
                agent.step,
                rgb, depth, pose, instruction, intrinsic, look_down
            )

        # 【新增】检测动作5并处理"向下看"
        if (dual_sys_output.output_action is not None and
            len(dual_sys_output.output_action) > 0 and
            dual_sys_output.output_action[0] == 5):

            logger.info(f"🔍 检测到动作5（向下看），准备执行look_down推理...")
            logger.info(f"   原始输出动作: {dual_sys_output.output_action}")

            # 使用相同的图像，设置look_down=True重新推理
            async with agent_lock:
                dual_sys_output = await asyncio.to_thread(
                    agent.step,
                    rgb, depth, pose, instruction, intrinsic, look_down=True
                )

            logger.info(f"✅ look_down推理完成")
            logger.info(f"   新的输出动作: {dual_sys_output.output_action}")
            logger.info(f"   新的输出像素: {dual_sys_output.output_pixel}")
            logger.info(f"   新的输出轨迹: {dual_sys_output.output_trajectory is not None}")

        inference_time = time.time() - start_time

        # 推理完成后再次获取历史帧数量和episode索引
        history_count_after = len(agent.rgb_list) if hasattr(agent, 'rgb_list') else 0
        episode_idx_after = agent.episode_idx if hasattr(agent, 'episode_idx') else 0
        logger.info(f"✅ 推理完成: 耗时={inference_time:.2f}秒, 累积历史帧={history_count_after}帧 (episode_idx={episode_idx_after})")

        # 构建响应 - 新格式，适配机器人控制接口
        response = {
            "status": "success",
            "id": robot_id,
            "pts": pts,
            "task_status": "executing",  # 默认值，后续根据输出调整
            "action": [[0.0, 0.0, 0.0]],  # 默认值
            "pixel_target": None,  # 归一化像素目标，默认为None
            "message": ""
        }

        # 添加输出字段并转换为机器人控制格式
        logger.info(f"📊 推理结果详情:")

        if dual_sys_output.output_action is not None:
            # 情况1/2/4：离散动作序列，转换为合并的[x, y, yaw]格式
            action_map = {0: 'STOP', 1: '↑前进', 2: '←左转', 3: '→右转', 5: '↓向下看'}
            action_str = ', '.join([f"{action_map.get(a, str(a))}" for a in dual_sys_output.output_action[:5]])
            if len(dual_sys_output.output_action) > 5:
                action_str += f", ... (共{len(dual_sys_output.output_action)}个动作)"
            logger.info(f"  ├─ 输出动作序列: {action_str}")
            logger.info(f"  │  └─ 原始序列: {dual_sys_output.output_action}")

            # 【新增】如果包含动作5，添加说明
            if 5 in dual_sys_output.output_action:
                logger.info(f"  │  ⚠️  注意: 输出包含动作5（向下看），已在推理阶段处理")

            # 转换为机器人控制格式
            robot_action, task_status = convert_output_action_to_robot_action(dual_sys_output.output_action)
            response["action"] = robot_action
            response["task_status"] = task_status
            logger.info(f"  ├─ 转换后机器人动作: {robot_action}")
            logger.info(f"  ├─ 任务状态: {task_status}")

        elif dual_sys_output.output_trajectory is not None:
            # 情况3：轨迹点列表，转换为累积坐标
            traj_shape = dual_sys_output.output_trajectory.shape
            logger.info(f"  ├─ 输出轨迹: shape={traj_shape}")

            # 转换为机器人控制格式（累积坐标）
            robot_action = convert_trajectory_to_robot_action(dual_sys_output.output_trajectory.tolist())
            response["action"] = robot_action
            response["task_status"] = "executing"

            # 计算累积坐标用于日志和可视化
            if len(robot_action) > 0:
                # robot_action 已经是累积坐标格式 [[x, y, yaw], ...]
                cumsum_trajectory = np.array([[pt[0], pt[1]] for pt in robot_action])
                start_point = cumsum_trajectory[0]
                end_point = cumsum_trajectory[-1]
                logger.info(f"  │  ├─ 起点(累积): [{start_point[0]:.3f}, {start_point[1]:.3f}]")
                logger.info(f"  │  └─ 终点(累积): [{end_point[0]:.3f}, {end_point[1]:.3f}]")
                # 保存累积轨迹供可视化使用
                dual_sys_output.output_trajectory = cumsum_trajectory

            logger.info(f"  ├─ 转换后轨迹点数: {len(robot_action)}")

        if dual_sys_output.output_pixel is not None:
            # 图像尺寸为 640x480
            pixel_y_normalized = dual_sys_output.output_pixel[0] / 480.0
            pixel_x_normalized = dual_sys_output.output_pixel[1] / 640.0
            response["pixel_target"] = [ pixel_x_normalized,pixel_y_normalized]
            logger.info(f"  └─ 输出像素目标: [y={dual_sys_output.output_pixel[0]}, x={dual_sys_output.output_pixel[1]}]")
            logger.info(f"     归一化像素目标: [y={pixel_y_normalized:.4f}, x={pixel_x_normalized:.4f}]")

        # ===== 需求4：检测小动作并自动停止 =====
        action_list = response["action"]
        if len(action_list) == 33:
            # 检查每个三元组的所有值是否都小于0.5（绝对值）
            all_small_movements = True
            for action_triplet in action_list:
                # action_triplet 格式: [x, y, yaw]
                if len(action_triplet) >= 3:
                    x, y, yaw = action_triplet[0], action_triplet[1], action_triplet[2]
                    if abs(x) >= 0.5 or abs(y) >= 0.5 or abs(yaw) >= 0.5:
                        all_small_movements = False
                        break

            if all_small_movements:
                logger.info(f"🎯 检测到33个小动作（所有值绝对值<0.5），自动转换为停止")
                logger.info(f"   原始action前3个: {action_list[:3]}")
                response["action"] = [[0.0, 0.0, 0.0]]
                response["task_status"] = "end"
                logger.info(f"   修改后: action={response['action']}, task_status={response['task_status']}")

        # 可视化推理结果并保存
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
            logger.info(f"🎨 生成可视化结果: {os.path.join(images_dir, f'annotated_{timestamp_str}.jpg')}")
        except Exception as e:
            logger.warning(f"生成可视化结果失败: {e}", exc_info=True)

        # 打印响应JSON
        logger.info(f"📤 响应JSON: {json.dumps(response, ensure_ascii=False, indent=2)}")

        # 更新会话状态
        session_state['request_count'] += 1
        session_state['last_instruction'] = instruction
        session_state['last_task'] = instruction  # 新增：保存当前task

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
            "message": f"推理处理异常: {e}"
        }


async def handle_client(websocket):
    """处理单个客户端连接"""
    client_id = id(websocket)
    session_state = {
        'last_instruction': None,
        'request_count': 0,
        'last_task': None  # 新增：记录上一次的task
    }

    global global_agent

    try:
        # 将新客户端添加到连接集合
        connected_clients[client_id] = {
            'websocket': websocket,
            'session_state': session_state
        }
        logger.info(f"新客户端连接 [{client_id}]。当前连接数: {len(connected_clients)}")

        # 如果全局agent未初始化，则初始化
        if global_agent is None:
            async with agent_lock:
                if global_agent is None:  # 双重检查
                    global_agent = init_agent()

        # 保持连接并处理消息
        async for message in websocket:
            try:
                # 解析接收到的JSON消息
                data = json.loads(message)

                # 日志记录（不打印大块数据：图像、深度图等）
                log_data = {}
                for k, v in data.items():
                    if k in ['rgb', 'depth'] and isinstance(v, str):
                        log_data[f"{k}_length"] = len(v)
                    elif k == 'images' and isinstance(v, dict):
                        log_images = {}
                        for img_k, img_v in v.items():
                            if isinstance(img_v, str):
                                log_images[img_k] = f"<base64 data, length={len(img_v)}>"
                            else:
                                log_images[img_k] = img_v
                        log_data['images'] = log_images
                    else:
                        log_data[k] = v

                logger.info("="*150)
                logger.info(f"收到消息 [{client_id}]: {json.dumps(log_data, ensure_ascii=False)}")
                logger.info("="*150)

                # 处理重置命令
                if data.get('command') == 'reset':
                    async with agent_lock:
                        global_agent.reset()
                    session_state['last_instruction'] = None
                    session_state['request_count'] = 0
                    response = {
                        "status": "success",
                        "message": "Agent已重置"
                    }
                    logger.info(f"Agent已重置 [{client_id}]")

                # 处理会话状态查询
                elif data.get('command') == 'session_status':
                    response = {
                        "status": "success",
                        "message": "会话状态信息",
                        "session_info": {
                            "request_count": session_state['request_count'],
                            "last_instruction": session_state.get('last_instruction', None)
                        }
                    }

                # 处理推理请求
                else:
                    response = await process_inference(data, session_state, global_agent)

                # 发送响应
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
        # 清理断开的连接
        if client_id in connected_clients:
            del connected_clients[client_id]
        logger.info(f"客户端断开连接 [{client_id}]。当前连接数: {len(connected_clients)}")


async def main():
    """启动WebSocket服务器"""
    # 切换工作目录到项目根目录，以便相对路径正确解析
    os.chdir(project_root)
    logger.info("🚀 启动InternNav WebSocket服务器...")
    logger.info(f"📂 工作目录: {os.getcwd()}")

    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        9527,  # WebSocket服务端口
        ping_interval=30,  # 心跳间隔(秒)
        ping_timeout=10,    # 心跳超时(秒)
        max_size=50*1024*1024  # 传输文件大小最大值（50M）
    )

    logger.info("✅ InternNav WebSocket服务器已启动，监听端口 9527")
    logger.info("📚 支持的消息格式:")
    logger.info("  输入格式:")
    logger.info("    - id: 机器人ID (必需)")
    logger.info("    - pts: 时间戳 (毫秒，必需)")
    logger.info("    - task: 导航指令 (必需，如 '穿过马路后左转')")
    logger.info("    - images: 图像字典 (必需)")
    logger.info("        - front_1: base64编码的前置摄像头图像 (必需)")
    logger.info("        - camera_1~4: 其他摄像头图像 (可选，暂不使用)")
    logger.info("  输出格式:")
    logger.info("    - status: 'success' 或 'error'")
    logger.info("    - id: 机器人ID")
    logger.info("    - pts: 时间戳")
    logger.info("    - task_status: 'executing' 或 'end'")
    logger.info("    - action: [[x, y, yaw], ...] 机器人控制命令")
    logger.info("    - message: 错误描述信息")
    logger.info("🔧 会话管理命令:")
    logger.info("  - command: 'reset' (重置Agent)")
    logger.info("  - command: 'session_status' (查看会话状态)")

    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ 服务器正在关闭...")
    except Exception as e:
        logger.error(f"❌ 服务器发生错误: {e}", exc_info=True)
