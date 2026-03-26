"""子图匹配坐标 → 机器人 (x, y) 运动坐标转换

柱面图像素位置 → 物理方向角 → 机器人坐标系 (x=前, y=左)
"""

import math
import time
from typing import Dict, Optional, Tuple

import yaml

# 默认参数
DEFAULT_FOV = 180        # 柱面图 FOV（度）
DEFAULT_WIDTH = 1920     # 柱面图宽度
DEFAULT_HEIGHT = 1536    # 柱面图高度
DEFAULT_DISTANCE = 1.0   # 可配置移动距离（米）

# 距离估算参数
DEFAULT_CAMERA_HEIGHT = 1.0    # 相机距地面高度（米）
DEFAULT_PITCH_UP = 15.0        # 柱面去畸变 pitch_up（度）
MIN_DEPRESSION_DEG = 5.0       # 最小俯角阈值（度）
DEFAULT_FIXED_DISTANCE = 20.0  # 固定距离（米），目标在水平面以上时使用
MIN_DISTANCE = 0.3             # 最小距离（米）
MAX_DISTANCE = 30.0            # 最大距离（米）

# 硬编码方位角（从 params.yaml T_ic 计算，单位：度，逆时针为正）
_DEFAULT_AZIMUTHS = {
    'camera_1': 39.42,
    'camera_2': -35.84,
    'camera_3': -142.04,
    'camera_4': 143.52,
}


def camera_azimuths_from_yaml(yaml_path: str) -> Dict[str, float]:
    """从 params.yaml 的 T_ic 计算各相机方位角（度，逆时针正）。

    光轴 = R_ic 第三列 (z_cam → IMU)
    azimuth = atan2(y_imu, x_imu)
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    azimuths = {}
    for cam in cfg['cameras']:
        name = cam.get('name', '')
        t_ic = cam.get('T_ic')
        if t_ic is None:
            continue
        # T_ic 是 4x4 行主序展平，旋转矩阵 R 的第三列索引: [2, 6, 10]
        z_x = t_ic[2]   # R[0][2]
        z_y = t_ic[6]   # R[1][2]
        azimuth_rad = math.atan2(z_y, z_x)
        azimuths[name] = math.degrees(azimuth_rad)
    return azimuths


def get_camera_azimuth(camera_id: str,
                       camera_azimuths: Optional[Dict[str, float]] = None) -> float:
    """返回相机方位角（度，逆时针正）。"""
    azimuths = camera_azimuths if camera_azimuths is not None else _DEFAULT_AZIMUTHS
    if camera_id not in azimuths:
        raise KeyError(f"未知相机: {camera_id}, 可用: {list(azimuths.keys())}")
    return azimuths[camera_id]


def pixel_to_angle(x_pixel: float,
                   fov: float = DEFAULT_FOV,
                   width: int = DEFAULT_WIDTH) -> float:
    """柱面图 x 像素 → 相机内角度（弧度）。

    柱面投影是等角距的：focal_gen = width / (fov_rad)
    画面右边(x_pixel > center) → 顺时针 → 负角度
    """
    fov_rad = fov * math.pi / 180.0
    focal_gen = width / fov_rad
    return (width / 2.0 - x_pixel) / focal_gen


def pixel_norm_to_angle(x_norm: float, fov: float = DEFAULT_FOV) -> float:
    """归一化 x_norm ∈ [0,1] → 相机内角度（弧度）。

    画面右边(x_norm>0.5) → 顺时针 → 负角度
    画面左边(x_norm<0.5) → 逆时针 → 正角度
    """
    return (0.5 - x_norm) * fov * math.pi / 180.0


def pixel_to_robot_xy(camera_id: str,
                      x_pixel: float,
                      y_pixel: float = None,
                      distance: float = DEFAULT_DISTANCE,
                      fov: float = DEFAULT_FOV,
                      width: int = DEFAULT_WIDTH,
                      camera_azimuths: Optional[Dict[str, float]] = None
                      ) -> Tuple[float, float]:
    """一步到位：(camera_id, x_pixel) → (x_forward, y_left)。

    1. x_pixel → 相机内角度
    2. + camera_azimuth → 全局角度
    3. * distance → (x, y)
    """
    theta_h = pixel_to_angle(x_pixel, fov, width)
    azimuth_deg = get_camera_azimuth(camera_id, camera_azimuths)
    global_angle = math.radians(azimuth_deg) + theta_h
    x = distance * math.cos(global_angle)
    y = distance * math.sin(global_angle)
    return x, y


def pixel_norm_to_robot_xy(camera_id: str,
                           x_norm: float,
                           y_norm: float = None,
                           distance: float = DEFAULT_DISTANCE,
                           fov: float = DEFAULT_FOV,
                           camera_azimuths: Optional[Dict[str, float]] = None
                           ) -> Tuple[float, float]:
    """归一化版本：x_norm ∈ [0,1] → (x_forward, y_left)。"""
    theta_h = pixel_norm_to_angle(x_norm, fov)
    azimuth_deg = get_camera_azimuth(camera_id, camera_azimuths)
    global_angle = math.radians(azimuth_deg) + theta_h
    x = distance * math.cos(global_angle)
    y = distance * math.sin(global_angle)
    return x, y


def estimate_distance_from_ynorm(
    y_norm: float,
    camera_height: float = DEFAULT_CAMERA_HEIGHT,
    pitch_up: float = DEFAULT_PITCH_UP,
    fov: float = DEFAULT_FOV,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fixed_distance: float = DEFAULT_FIXED_DISTANCE,
    min_distance: float = MIN_DISTANCE,
    max_distance: float = MAX_DISTANCE,
) -> Tuple[float, float]:
    """y_norm → (distance, depression_deg)

    利用柱面投影几何关系，从图像垂直位置估算目标的水平距离。

    Args:
        y_norm: 归一化垂直坐标 [0,1]，0=图像顶部，1=图像底部
        camera_height: 相机距地面高度（米）
        pitch_up: 柱面去畸变 pitch_up（度）
        fov: 柱面图水平 FOV（度）
        width: 柱面图宽度（像素）
        height: 柱面图高度（像素）
        fixed_distance: 目标在水平面以上时使用的固定距离（米）
        min_distance: 最小距离（米）
        max_distance: 最大距离（米）

    Returns:
        distance: 估算的水平距离（米）
        depression_deg: 俯仰角（度，正=向下看地面）
    """
    fov_rad = fov * math.pi / 180.0
    focal_gen = width / fov_rad
    ny = y_norm * height - height / 2.0

    pitch_rad = pitch_up * math.pi / 180.0
    obj_y = ny * math.cos(pitch_rad) - focal_gen * math.sin(pitch_rad)
    obj_z = ny * math.sin(pitch_rad) + focal_gen * math.cos(pitch_rad)

    depression = math.atan2(-obj_y, obj_z)  # 正值=向下
    depression_deg = math.degrees(depression)

    if depression_deg > MIN_DEPRESSION_DEG:
        distance = camera_height / math.tan(depression)
        distance = max(min_distance, min(distance, max_distance))
    else:
        distance = fixed_distance

    return distance, depression_deg


def pixel_target_to_action(
    x_norm: float,
    y_norm: float,
    camera_id: str,
    camera_height: float = DEFAULT_CAMERA_HEIGHT,
    pitch_up: float = DEFAULT_PITCH_UP,
    fov: float = DEFAULT_FOV,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    camera_azimuths: Optional[Dict[str, float]] = None,
    fixed_distance: float = DEFAULT_FIXED_DISTANCE,
) -> Tuple[float, float, dict]:
    """完整转换：pixel_target + camera_id → (x_forward, y_lateral) + 调试信息

    1. x_norm → 水平角 yaw_global
    2. y_norm → 垂直俯仰角 depression → 距离估算
    3. yaw_global + distance → (x_forward, y_lateral)

    注意：不返回 yaw，yaw 由调用方保持原值。

    Returns:
        x_forward: 前进方向分量（米）
        y_lateral: 横向分量（米，正=左）
        debug: 调试信息字典
    """
    t0 = time.perf_counter()

    # 1. 水平角度
    theta_h = pixel_norm_to_angle(x_norm, fov)
    azimuth_deg = get_camera_azimuth(camera_id, camera_azimuths)
    yaw_global = math.radians(azimuth_deg) + theta_h

    # 2. 距离估算
    distance, depression_deg = estimate_distance_from_ynorm(
        y_norm, camera_height, pitch_up, fov, width, height, fixed_distance
    )

    # 3. 分解为 (x, y)
    x_forward = distance * math.cos(yaw_global)
    y_lateral = distance * math.sin(yaw_global)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    debug = {
        'distance': round(distance, 3),
        'depression_deg': round(depression_deg, 1),
        'yaw_local_deg': round(math.degrees(theta_h), 1),
        'yaw_global_deg': round(math.degrees(yaw_global), 1),
        'elapsed_ms': round(elapsed_ms, 3),
    }

    return x_forward, y_lateral, debug


if __name__ == '__main__':
    print("=== coord_transform 单元测试 ===\n")

    # 测试 1: y_norm=0.5 应对应 depression=pitch_up=15°
    dist, dep = estimate_distance_from_ynorm(0.5)
    print(f"y_norm=0.5 → depression={dep:.1f}° (期望≈15.0°), distance={dist:.3f}m")

    # 测试 2: x_norm=0.5 应对应 yaw_local=0°
    angle = pixel_norm_to_angle(0.5)
    print(f"x_norm=0.5 → yaw_local={math.degrees(angle):.1f}° (期望=0.0°)")

    # 测试 3: 各种 y_norm 值
    print("\n--- y_norm vs distance (camera_height=1.0m, pitch_up=15°) ---")
    for yn in [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        d, dep = estimate_distance_from_ynorm(yn)
        print(f"  y_norm={yn:.1f} → depression={dep:+6.1f}°, distance={d:6.2f}m")

    # 测试 4: pixel_target_to_action 各相机
    print("\n--- pixel_target_to_action (x_norm=0.5, y_norm=0.6) ---")
    for cam in ['camera_1', 'camera_2', 'camera_3', 'camera_4']:
        xf, yl, dbg = pixel_target_to_action(0.5, 0.6, cam)
        print(f"  {cam}: x_fwd={xf:+.3f}, y_lat={yl:+.3f}, "
              f"dist={dbg['distance']:.3f}m, dep={dbg['depression_deg']:.1f}°, "
              f"yaw_global={dbg['yaw_global_deg']:.1f}°, "
              f"elapsed={dbg['elapsed_ms']:.3f}ms")
