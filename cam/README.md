# Cam - 多目鱼眼相机 ROS2 节点

基于 NVIDIA Jetson 平台的多目鱼眼相机采集与处理 ROS2 节点，支持鱼眼去畸变、虚拟视角生成、双目拼接等功能。

## 功能概述

- **多相机采集**：通过 V4L2 从多个摄像头采集视频流
- **鱼眼去畸变**：支持 pinhole（针孔）和 cylindrical（柱面）两种虚拟视角
- **双目前视拼接**：将 camera_1 与 camera_2 的前视柱面图拼接为全景图
- **JPEG 压缩发布**：使用 NVIDIA 硬件编码器进行 JPEG 压缩
- **TF 变换发布**：发布相机到 IMU 的静态坐标变换

## 系统要求

- **硬件**：NVIDIA Jetson 系列（依赖 NvBufSurface、NvJPEGEncoder、CUDA）
- **系统**：Linux，支持 V4L2
- **ROS2**：Humble 或更高版本

## 依赖

- ROS2（rclcpp, sensor_msgs, geometry_msgs, tf2_ros）
- OpenCV（含 CUDA 模块：cudawarping, cudaarithm, cudaimgproc）
- Eigen3
- yaml-cpp
- cv_bridge
- NVIDIA 多媒体 API：NvBufSurface, NvJpegEncoder
- jetson-utils
- CTPL 线程池（ctpl_stl.h，需放在 `../src/` 或调整 include 路径）

## 项目结构

```
cam/
├── main.cc          # 主程序，CameraNode 实现
├── main.h           # 头文件与依赖
├── video.h          # V4L2 视频采集与 DMA 缓冲
├── fisheye_undist.h # 鱼眼去畸变与虚拟视角
├── params.yaml      # 相机参数配置
├── tools/           # 独立工具
│   ├── fisheye_to_cylindrical.cpp  # 鱼眼转柱面视角命令行工具
│   ├── fisheye_undist_cpu.h        # CPU 版去畸变（无 CUDA 依赖）
│   ├── CMakeLists.txt
│   └── README.md
└── README.md        # 本文档
```

## 配置说明

### 1. 设备映射文件

程序从 `/tmp/camera_dev.txt` 读取摄像头设备路径。每行一个或多个设备路径，按空格分隔，最多 8 个设备。

示例：

```
/dev/video0 /dev/video1
/dev/video2
/dev/video6 /dev/video7
```

`params.yaml` 中的 `dev` 为上述列表中的索引（从 0 开始）。

### 2. 参数文件 params.yaml

通过 ROS2 参数 `params` 指定 YAML 路径，例如：

```bash
ros2 run <package_name> <node_name> --ros-args -p params:=/path/to/params.yaml
```

#### 相机配置项

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | string | 相机名称，用于话题和 frame_id |
| `dev` | int | 设备索引，对应 camera_dev.txt 中的顺序 |
| `downscale` | bool | 是否下采样（分辨率减半） |
| `flip` | bool | 是否 180° 翻转 |
| `filter` | bool | 帧率过滤：true≈10fps，false≈30fps |
| `T_ic` | double[16] | 4×4 相机到 IMU 外参矩阵（行优先） |
| `distortion_coeffs` | double[4] | 鱼眼畸变系数 [k1, k2, p1, p2] |
| `intrinsics` | double[5] | 内参 [xi, fx, fy, cx, cy]（鱼眼模型） |
| `vcam` | list | 虚拟视角配置列表 |

#### 虚拟视角 vcam 配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `type` | string | `pinhole` 或 `cylindrical` |
| `direction` | string | 朝向：`front` / `right` / `back` / `left` |
| `fov` | int | 视场角（度） |
| `width` | int | 输出宽度 |
| `height` | int | 输出高度 |

## 话题说明

### 发布话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `{camera_name}/image_raw` | sensor_msgs/Image | 原始/转换后的 RGB 图像 |
| `{camera_name}/image_raw/compressed` | sensor_msgs/CompressedImage | JPEG 压缩图像 |
| `{camera_name}/{direction}_{type}/image_raw` | sensor_msgs/Image | 虚拟视角图像，如 `front_cylindrical` |
| `/stitch/image_raw` | sensor_msgs/Image | 双目前视拼接图（camera_1 + camera_2） |
| `/robot/sensors/params` | std_msgs/String | 拼接参数（left_x, right_x, width, height） |

### TF 变换

- 父坐标系：`imu_frame`
- 子坐标系：各相机 `name`（仅配置了 `T_ic` 的相机会发布）

## 运行方式

### 1. 准备设备映射

```bash
echo "/dev/video1
/dev/video2
/dev/video3
/dev/video6
/dev/video7" > /tmp/camera_dev.txt
```

根据实际设备路径调整。

### 2. 启动节点

```bash
ros2 run <package_name> robot_sensors --ros-args -p params:=/path/to/cam/params.yaml
```

节点名为 `robot_sensors`。

### 3. 查看图像

```bash
# 原始图像
ros2 run rqt_image_view rqt_image_view

# 或使用 rviz2 订阅相应 Image 话题
```

## 分辨率说明

- **前视相机**（dev 为列表第一个）：3840×2160
- **其他相机**：1920×1536
- 若 `downscale: true`，输出分辨率为上述的一半

## 拼接逻辑

- 仅当 camera_1 和 camera_2 均配置 `front_cylindrical` 虚拟视角时进行拼接
- 拼接参数：`m_left_x=0.78`，`m_right_x=0.98`，基准宽度 640，高度 480
- 左图取左侧 78% 区域，右图取右侧 2% 区域，拼接为全景图

## 注意事项

1. **ctpl_stl.h**：`video.h` 中 `#include "../src/ctpl_stl.h"`，需确保该头文件存在，或将项目放在含 `src/ctpl_stl.h` 的父目录下。
2. **ARM 指令**：`TscOffset` 使用 `mrs cntfrq_el0`、`mrs cntvct_el0`，仅适用于 ARM64。
3. **时钟同步**：依赖 `/sys/devices/system/clocksource/clocksource0/offset_ns` 做时间戳校正。
4. **CUDA 设备**：固定使用 `cudaSetDevice(0)`，单 GPU 环境适用。

## 编译

项目需集成到 ROS2 工作空间中，并正确配置：

- OpenCV CUDA
- NVIDIA 多媒体库
- jetson-utils
- Eigen3、yaml-cpp、cv_bridge

若使用 colcon：

```bash
cd /path/to/ros2_ws
colcon build --packages-select <cam_package_name>
source install/setup.bash
```

## 独立工具：鱼眼转柱面视角

`tools/` 目录下提供不依赖 ROS2 和 Jetson 的命令行工具，可将鱼眼图像去畸变为柱面视角：

```bash
cd tools && mkdir build && cd build && cmake .. && make
./fisheye_to_cylindrical -i fisheye.jpg -o cylindrical.jpg -c ../params.yaml -n camera_1
```

详见 [tools/README.md](tools/README.md)。

## 许可证

请参考项目根目录的 LICENSE 文件。
