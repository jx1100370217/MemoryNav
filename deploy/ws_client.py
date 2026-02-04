#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternNav WebSocket客户端
用于测试ws_proxy.py的InternVLA-N1推理服务
"""

import os

# 禁用localhost的代理，避免WebSocket连接被代理拦截
# 这必须在导入websocket之前设置
no_proxy = os.environ.get('no_proxy', os.environ.get('NO_PROXY', ''))
if 'localhost' not in no_proxy:
    localhost_list = 'localhost,127.0.0.1,::1'
    if no_proxy:
        os.environ['no_proxy'] = f"{no_proxy},{localhost_list}"
    else:
        os.environ['no_proxy'] = localhost_list

import base64
import json
import websocket
import glob
import time
from typing import Dict, List, Optional
import numpy as np
from PIL import Image


# 数据集配置字典
DATASET_CONFIGS = {
    'realworld_sample_data1': {
        'path': 'assets/realworld_sample_data1',
        'pattern': 'debug_raw_*.jpg',
        'instruction_file': 'instruction.txt',
        'instruction': None,  # 从文件读取
        'has_look_down': True,
        'look_down_suffix': '_look_down.jpg',
        'supports_single': True,
        'supports_continuous': True,
    },
    'realworld_sample_data2': {
        'path': 'assets/realworld_sample_data2',
        'pattern': 'debug_raw_*.jpg',
        'instruction_file': 'instruction.txt',
        'instruction': None,
        'has_look_down': True,
        'look_down_suffix': '_look_down.jpg',
        'supports_single': True,
        'supports_continuous': True,
    },
    'test_data': {
        'path': 'test_data',
        'pattern': '*_input.jpg',  # 新格式
        'instruction_file': None,
        'instruction': 'Stop at the black chair ahead.',  # 固定指令
        'has_look_down': False,
        'look_down_suffix': None,
        'supports_single': False,  # 只支持 continuous
        'supports_continuous': True,
    }
}


class WsClient:
    """WebSocket客户端，支持InternNav推理"""

    def __init__(self, host='localhost', port=9527):
        """
        初始化WebSocket客户端

        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.ws_url = f"ws://{host}:{port}"
        self.ws = None
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=120)
            print(f"✅ 成功连接到 {self.ws_url}")
        except Exception as e:
            print(f"❌ 连接失败 {self.ws_url}: {e}")

    def call_inference(self, instruction: str, rgb_path: str,
                      depth_path: Optional[str] = None,
                      robot_id: str = "TEST_ROBOT_001",
                      pts: Optional[int] = None,
                      pose: Optional[np.ndarray] = None,
                      intrinsic: Optional[np.ndarray] = None,
                      look_down: bool = False) -> Optional[dict]:
        """
        调用推理接口

        Args:
            instruction: 导航指令
            rgb_path: RGB图像路径
            depth_path: 深度图路径（可选）
            robot_id: 机器人ID
            pts: 时间戳（毫秒），如果为None则自动生成
            pose: 位姿矩阵 (4, 4)
            intrinsic: 相机内参矩阵 (4, 4)
            look_down: 是否向下看

        Returns:
            dict: 服务器响应
        """
        if not self.ws:
            print("❌ WebSocket 连接不可用。")
            return None

        try:
            # 读取RGB图像并编码
            if not os.path.exists(rgb_path):
                print(f"❌ RGB图像不存在: {rgb_path}")
                return None

            with open(rgb_path, 'rb') as f:
                rgb_base64 = base64.b64encode(f.read()).decode('utf-8')

            # 读取深度图并编码（如果提供）
            depth_base64 = None
            if depth_path and os.path.exists(depth_path):
                with open(depth_path, 'rb') as f:
                    depth_base64 = base64.b64encode(f.read()).decode('utf-8')

            # 如果没有提供pts，自动生成
            if pts is None:
                pts = int(time.time() * 1000)

            # 构建请求数据
            data = {
                'id': robot_id,
                'task': instruction,
                'pts': pts,
                'images': {
                    'front_1': rgb_base64
                },
                'look_down': look_down
            }

            if depth_base64:
                data['depth'] = depth_base64

            if pose is not None:
                data['pose'] = pose.tolist()

            if intrinsic is not None:
                data['intrinsic'] = intrinsic.tolist()

            # 发送数据
            json_data = json.dumps(data)
            print(f"📤 正在发送请求")
            print(f"   ID: {robot_id}")
            print(f"   指令: {instruction}")
            print(f"   PTS: {pts}")
            print(f"   RGB图像: {os.path.basename(rgb_path)}")
            if depth_path:
                print(f"   深度图: {os.path.basename(depth_path)}")
            print(f"   向下看: {look_down}")

            start_time = time.time()
            self.ws.send(json_data)

            # 接收响应
            print("⏳ 等待服务器响应...")
            result = self.ws.recv()
            elapsed_time = time.time() - start_time

            recv_json = json.loads(result)
            print(f"✅ 收到响应，总耗时: {elapsed_time:.2f}秒")

            return recv_json

        except Exception as e:
            print(f"❌ call_inference 时发生错误: {e}")
            # 尝试重新连接
            self.reconnect()
            return None

    def reset_agent(self) -> Optional[dict]:
        """
        重置Agent状态

        Returns:
            dict: 服务器响应
        """
        if not self.ws:
            print("❌ WebSocket 连接不可用。")
            return None

        try:
            data = {
                'command': 'reset'
            }

            json_data = json.dumps(data)
            self.ws.send(json_data)

            result = self.ws.recv()

            if not result or not result.strip():
                print("⚠️  服务器返回空响应")
                return {"status": "error", "message": "服务器返回空响应"}

            recv_json = json.loads(result)

            print("✅ Agent已重置")
            return recv_json

        except json.JSONDecodeError as e:
            print(f"❌ reset_agent JSON解析错误: {e}")
            print(f"   收到的内容: {result[:200] if result else 'None'}")
            return None
        except Exception as e:
            print(f"❌ reset_agent 时发生错误: {e}")
            return None

    def get_session_status(self) -> Optional[dict]:
        """
        获取当前会话状态

        Returns:
            dict: 会话状态信息
        """
        if not self.ws:
            print("❌ WebSocket 连接不可用。")
            return None

        try:
            data = {
                'command': 'session_status'
            }

            json_data = json.dumps(data)
            self.ws.send(json_data)

            result = self.ws.recv()
            recv_json = json.loads(result)

            return recv_json

        except Exception as e:
            print(f"❌ get_session_status 时发生错误: {e}")
            return None

    def reconnect(self):
        """尝试重新连接到服务器"""
        print("🔄 尝试重新连接...")
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=120)
            print(f"✅ 成功重新连接到 {self.ws_url}")
        except Exception as e:
            print(f"❌ 重新连接失败: {e}")
            self.ws = None

    def close(self):
        """关闭WebSocket连接"""
        if self.ws:
            self.ws.close()
            print("🔌 WebSocket 连接已关闭。")


def get_dataset_config(dataset_name: str) -> dict:
    """
    获取数据集配置

    Args:
        dataset_name: 数据集名称

    Returns:
        dict: 数据集配置字典

    Raises:
        ValueError: 数据集不存在
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(f"未知的数据集: {dataset_name}。可用: {available}")

    return DATASET_CONFIGS[dataset_name].copy()


def load_instruction(config: dict, scene_dir: str) -> str:
    """
    从配置或文件加载导航指令

    Args:
        config: 数据集配置
        scene_dir: 场景数据目录

    Returns:
        str: 导航指令
    """
    # 如果配置中有固定的指令，直接返回
    if config['instruction'] is not None:
        return config['instruction']

    # 否则从 instruction.txt 文件读取
    if config['instruction_file'] is not None:
        instruction_path = os.path.join(scene_dir, config['instruction_file'])
        if os.path.exists(instruction_path):
            with open(instruction_path, 'r') as f:
                return f.read().strip()
        else:
            print(f"⚠️  未找到 {config['instruction_file']}")

    # 默认指令
    return "请向前直行"


def get_rgb_files(scene_dir: str, config: dict) -> List[str]:
    """
    获取RGB图像文件列表

    Args:
        scene_dir: 场景数据目录
        config: 数据集配置

    Returns:
        List[str]: 排序后的RGB图像路径列表(不包含look_down图像)
    """
    pattern = config['pattern']
    rgb_files = sorted(glob.glob(os.path.join(scene_dir, pattern)))

    # 过滤掉 look_down 图像
    if config['has_look_down'] and config['look_down_suffix']:
        rgb_files = [f for f in rgb_files if not f.endswith(config['look_down_suffix'])]

    return rgb_files


def get_look_down_path(rgb_path: str, config: dict) -> Optional[str]:
    """
    获取对应的 look_down 图像路径(如果存在)

    Args:
        rgb_path: RGB图像路径
        config: 数据集配置

    Returns:
        Optional[str]: look_down图像路径，如果不存在或配置不支持则返回None
    """
    if not config['has_look_down'] or not config['look_down_suffix']:
        return None

    # 对于 debug_raw_*.jpg 格式
    if config['pattern'].startswith('debug_raw_'):
        look_down_path = rgb_path.replace('.jpg', config['look_down_suffix'])
        if os.path.exists(look_down_path):
            return look_down_path

    return None


def print_response(response: dict, verbose: bool = True):
    """
    美化打印响应结果

    Args:
        response: 服务器响应
        verbose: 是否打印详细信息
    """
    if not response:
        print("\n❌ 未从服务器收到有效响应。")
        return

    print("\n" + "="*80)
    print("📊 推理结果")
    print("="*80)

    if response.get('status') == 'success':
        print(f"✅ 状态: 成功")
        print(f"\n🤖 机器人ID: {response.get('id', 'N/A')}")
        print(f"⏰ 时间戳(PTS): {response.get('pts', 'N/A')}")
        inference_time = response.get('inference_time')
        if inference_time is not None:
            print(f"⏱️  推理时间: {inference_time:.2f}秒")
        else:
            print(f"⏱️  推理时间: N/A")

        # 打印输出动作序列
        if response.get('output_action'):
            action_seq = response['output_action']
            print(f"\n🎯 输出动作序列:")
            print(f"   {action_seq}")
            action_map = {0: 'STOP', 1: '前进↑', 2: '左转←', 3: '右转→', 5: '向下看↓'}
            action_names = [action_map.get(a, f'未知({a})') for a in action_seq]
            print(f"   解析: {' -> '.join(action_names)}")

        # 打印输出轨迹
        if response.get('output_trajectory'):
            trajectory = response['output_trajectory']
            print(f"\n📈 输出轨迹 (连续):")
            print(f"   轨迹点数: {len(trajectory)}")
            if verbose and len(trajectory) > 0:
                print(f"   前3个点: {trajectory[:3]}")
                print(f"   后3个点: {trajectory[-3:]}")

        # 打印输出像素目标
        if response.get('output_pixel'):
            pixel = response['output_pixel']
            print(f"\n🎯 输出像素目标:")
            print(f"   坐标 [y, x]: {pixel}")

        # 打印归一化像素目标
        if response.get('pixel_target') is not None:
            pixel_target = response['pixel_target']
            print(f"\n🎯 归一化像素目标 (pixel_target):")
            print(f"   坐标 [y, x]: [{pixel_target[0]:.4f}, {pixel_target[1]:.4f}]")

    else:
        print(f"❌ 状态: 失败")
        print(f"🤖 机器人ID: {response.get('id', 'N/A')}")
        print(f"⏰ 时间戳(PTS): {response.get('pts', 'N/A')}")
        print(f"💬 错误消息: {response.get('message', 'N/A')}")

    print("="*80 + "\n")


def test_single_inference(client: WsClient, scene_dir: str = None, dataset: str = None):
    """
    单次推理测试

    Args:
        client: WsClient实例
        scene_dir: 场景数据目录(优先级高于dataset)
        dataset: 数据集名称
    """
    print("\n📌 测试: 单次推理测试")
    print("-"*80)

    # 确定数据集配置
    if scene_dir is not None:
        # 如果直接指定了scene_dir，使用默认配置
        config = DATASET_CONFIGS['realworld_sample_data1'].copy()
        config['path'] = scene_dir
    elif dataset is not None:
        # 使用指定的数据集配置
        try:
            config = get_dataset_config(dataset)
            scene_dir = config['path']

            # 检查是否支持single模式
            if not config['supports_single']:
                print(f"❌ 数据集 '{dataset}' 不支持单次推理模式，请使用 continuous 模式")
                return None
        except ValueError as e:
            print(f"❌ {e}")
            return None
    else:
        # 默认使用 realworld_sample_data1
        dataset = 'realworld_sample_data1'
        config = get_dataset_config(dataset)
        scene_dir = config['path']

    if not os.path.exists(scene_dir):
        print(f"❌ 场景目录不存在: {scene_dir}")
        return None

    # 加载导航指令
    instruction = load_instruction(config, scene_dir)
    print(f"📝 导航指令: {instruction}")

    # 获取RGB图像列表
    rgb_files = get_rgb_files(scene_dir, config)
    if not rgb_files:
        print(f"❌ 未找到RGB图像文件 (pattern: {config['pattern']})")
        return None

    rgb_path = rgb_files[0]
    print(f"📸 使用RGB图像: {os.path.basename(rgb_path)}")

    # 执行推理
    response = client.call_inference(
        instruction=instruction,
        rgb_path=rgb_path,
        depth_path=None,
        robot_id="TEST_ROBOT_001"
    )

    print_response(response, verbose=True)
    return response


def test_continuous_inference(client: WsClient, scene_dir: str = None,
                             max_frames: int = None, dataset: str = None):
    """
    连续推理测试

    Args:
        client: WsClient实例
        scene_dir: 场景数据目录(优先级高于dataset)
        max_frames: 最大测试帧数
        dataset: 数据集名称
    """
    print("\n📌 测试: 连续推理模式")
    print("="*80)

    # 确定数据集配置
    if scene_dir is not None:
        # 如果直接指定了scene_dir，使用默认配置
        config = DATASET_CONFIGS['realworld_sample_data1'].copy()
        config['path'] = scene_dir
    elif dataset is not None:
        # 使用指定的数据集配置
        try:
            config = get_dataset_config(dataset)
            scene_dir = config['path']

            # 检查是否支持continuous模式
            if not config['supports_continuous']:
                print(f"❌ 数据集 '{dataset}' 不支持连续推理模式")
                return None
        except ValueError as e:
            print(f"❌ {e}")
            return None
    else:
        # 默认使用 realworld_sample_data1
        dataset = 'realworld_sample_data1'
        config = get_dataset_config(dataset)
        scene_dir = config['path']

    if not os.path.exists(scene_dir):
        print(f"❌ 场景目录不存在: {scene_dir}")
        return None

    print(f"🎯 测试集: {dataset if dataset else 'custom'}")
    print(f"📂 路径: {scene_dir}")
    print("="*80)

    # 加载导航指令
    instruction = load_instruction(config, scene_dir)
    print(f"📝 导航指令: {instruction}")

    # 获取所有RGB图像
    rgb_files = get_rgb_files(scene_dir, config)
    if not rgb_files:
        print(f"❌ 未找到RGB图像文件 (pattern: {config['pattern']})")
        return None

    if max_frames and max_frames > 0:
        rgb_files = rgb_files[:max_frames]

    print(f"📊 找到 {len(rgb_files)} 帧数据")

    # 重置Agent
    print("\n🔄 重置Agent状态...")
    client.reset_agent()

    # 连续推理
    total_frames = len(rgb_files)
    success_count = 0
    failed_count = 0
    total_inference_time = 0

    for frame_idx, rgb_path in enumerate(rgb_files, 1):
        print("\n" + "="*80)
        print(f"🎬 处理第 {frame_idx}/{total_frames} 帧")
        if frame_idx == 1:
            print(f"[第一帧] 将发送完整指令: '{instruction}'")
        else:
            print(f"[后续帧] 将发送'None'，测试ws_proxy中的task复用机制")
        print("="*80)

        # 检查是否有对应的look_down图像
        look_down_path = get_look_down_path(rgb_path, config)
        has_look_down = look_down_path is not None

        print(f"📸 RGB图像: {os.path.basename(rgb_path)}")
        if has_look_down:
            print(f"📸 Look-down图像: {os.path.basename(look_down_path)}")

        # 执行推理
        # 第一帧使用真实指令，后续帧传"None"以测试ws_proxy的task复用逻辑
        infer_instruction = instruction if frame_idx == 1 else "None"
        print(f"📝 指令: {instruction} (发送: {infer_instruction})")
        print("⏳ 开始推理...")

        response = client.call_inference(
            instruction=infer_instruction,
            rgb_path=rgb_path,
            depth_path=None,
            robot_id="TEST_ROBOT_001",
            pts=int(time.time() * 1000),
            look_down=False
        )

        if response and response.get('status') == 'success':
            success_count += 1
            inference_time = response.get('inference_time', 0)
            total_inference_time += inference_time

            print(f"\n✅ 推理成功 (耗时: {inference_time:.2f}秒)")

            # 打印结果摘要
            if response.get('output_action'):
                print(f"   动作序列: {response['output_action']}")
            if response.get('output_trajectory'):
                print(f"   轨迹点数: {len(response['output_trajectory'])}")
            if response.get('output_pixel'):
                print(f"   像素目标: {response['output_pixel']}")
            if response.get('pixel_target') is not None:
                pt = response['pixel_target']
                print(f"   归一化像素目标: [{pt[0]:.4f}, {pt[1]:.4f}]")

            # 如果有look_down图像，执行额外的look_down推理
            if has_look_down:
                print("\n   执行look_down推理...")
                # look_down推理也采用相同的task复用逻辑
                look_down_infer_instruction = instruction if frame_idx == 1 else "None"
                look_down_response = client.call_inference(
                    instruction=look_down_infer_instruction,
                    rgb_path=look_down_path,
                    depth_path=None,
                    robot_id="TEST_ROBOT_001",
                    pts=int(time.time() * 1000),
                    look_down=True
                )
                if look_down_response and look_down_response.get('status') == 'success':
                    print(f"   ✅ Look-down推理成功")

        else:
            failed_count += 1
            print(f"\n❌ 推理失败")
            if response:
                print(f"   错误信息: {response.get('message', 'N/A')}")

        # 短暂延迟
        if frame_idx < total_frames:
            time.sleep(0.3)

    # 统计报告
    print("\n" + "="*80)
    print("📈 统计报告")
    print("="*80)
    print(f"\n📊 总体统计:")
    print(f"   总帧数: {total_frames}")
    print(f"   成功推理: {success_count} ({success_count/total_frames*100:.1f}%)")
    print(f"   失败推理: {failed_count} ({failed_count/total_frames*100:.1f}%)")

    if success_count > 0:
        avg_inference_time = total_inference_time / success_count
        print(f"\n⏱️  平均推理时间: {avg_inference_time:.2f}秒")
        print(f"   总推理时间: {total_inference_time:.2f}秒")

    print("\n" + "="*80)
    print("📊 查看最终会话状态:")
    print("="*80)
    status = client.get_session_status()
    if status:
        print(json.dumps(status, indent=2, ensure_ascii=False))

    return {
        'total_frames': total_frames,
        'success_count': success_count,
        'failed_count': failed_count,
        'avg_inference_time': avg_inference_time if success_count > 0 else 0
    }


def main():
    """主测试函数"""
    import argparse

    print("🚀 启动InternNav WebSocket客户端测试")
    print("="*80)

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='InternNav WebSocket客户端测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认数据集 (realworld_sample_data1) 进行连续推理
  python ws_client.py

  # 使用 test_data 数据集进行连续推理
  python ws_client.py --dataset test_data

  # 使用 test_data 数据集，限制最多测试5帧
  python ws_client.py --dataset test_data --max-frames 5

  # 使用自定义目录路径 (向后兼容)
  python ws_client.py --scene-dir assets/realworld_sample_data2

  # 单次推理模式 (仅支持 realworld_sample_data1/2)
  python ws_client.py --mode single --dataset realworld_sample_data1

  # 连续推理模式，指定服务器地址
  python ws_client.py --dataset test_data --host 192.168.1.100 --port 9527

可用数据集:
  - realworld_sample_data1 (默认)
  - realworld_sample_data2
  - test_data (仅支持continuous模式)
        """
    )

    # 添加参数
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'continuous'],
        default='continuous',
        help='测试模式: single(单次推理) 或 continuous(连续推理)，默认: continuous'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help=f'数据集名称，可选: {", ".join(DATASET_CONFIGS.keys())}。默认: realworld_sample_data1'
    )

    parser.add_argument(
        '--scene-dir',
        type=str,
        default=None,
        help='自定义场景数据目录路径 (优先级高于--dataset，用于向后兼容)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='最大测试帧数，默认: 测试所有帧'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='WebSocket服务器地址，默认: localhost'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=9527,
        help='WebSocket服务器端口，默认: 9527'
    )

    # 解析参数
    args = parser.parse_args()

    # 创建客户端
    client = WsClient(host=args.host, port=args.port)

    if not client.ws:
        print("❌ 无法建立 WebSocket 连接，正在退出。")
        return

    # 执行测试
    try:
        if args.mode == 'single':
            test_single_inference(
                client,
                scene_dir=args.scene_dir,
                dataset=args.dataset
            )
        elif args.mode == 'continuous':
            test_continuous_inference(
                client,
                scene_dir=args.scene_dir,
                max_frames=args.max_frames,
                dataset=args.dataset
            )
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    # 关闭连接
    client.close()
    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()
