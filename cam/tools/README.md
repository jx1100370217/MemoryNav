# 鱼眼转柱面视角工具

将鱼眼相机图像去畸变并投影为柱面视角的独立命令行工具，参考 cam 仓库的 `FisheyeUndist` 实现。

## 依赖

- OpenCV
- Eigen3
- yaml-cpp

## 编译

### 方式一：Makefile（推荐，无需 CMake）

```bash
cd /home/zhangning/project/cam/tools
make
```

### 方式二：CMake

若 Cursor 集成终端中 cmake 报 `CMAKE_ROOT` 错误，可在系统终端中执行：

```bash
cd /home/zhangning/project/cam/tools
rm -rf build && mkdir build && cd build
/usr/bin/cmake ..
make
```

## 用法

### 基本用法（使用 params.yaml）

```bash
./fisheye_to_cylindrical -i fisheye.jpg -o cylindrical.jpg -c ../params.yaml -n camera_1
```

### 指定相机

若 `params.yaml` 中有多台相机，用 `-n` 指定名称：

```bash
./fisheye_to_cylindrical -i input.jpg -o output.jpg -c ../params.yaml -n camera_2
```

### 自定义输出尺寸与视场角

```bash
./fisheye_to_cylindrical -i input.jpg -o output.jpg -c ../params.yaml -n camera_1 \
    -f 180 -w 640 -h 480
```

### 自定义朝向

```bash
./fisheye_to_cylindrical -i input.jpg -o output.jpg -c ../params.yaml -d front
# direction: front | right | back | left
```

### 直接指定内参（不依赖 YAML）

```bash
./fisheye_to_cylindrical -i input.jpg -o output.jpg \
    --intrinsics "1.77,1238.9,1238.2,962.7,767.9" \
    --distortion "1.43,-1.78,-0.0014,0.005"
```

## 参数说明

| 参数 | 说明 | 默认 |
|------|------|------|
| `-i, --input` | 输入鱼眼图像路径 | 必填 |
| `-o, --output` | 输出柱面图路径 | 必填 |
| `-c, --config` | 相机参数 YAML（与 cam 的 params.yaml 格式兼容） | - |
| `-n, --name` | 相机名称，从 config 中选取 | 第一个有内参的 |
| `-f, --fov` | 输出视场角（度） | 180 |
| `-w, --width` | 输出宽度 | 640 |
| `-h, --height` | 输出高度 | 480 |
| `-d, --direction` | 朝向 front/right/back/left | front |
| `--intrinsics` | 内参 [xi,fx,fy,cx,cy] 逗号分隔 | - |
| `--distortion` | 畸变 [k1,k2,p1,p2] 逗号分隔 | - |
| `--extrinsic` | 外参 4x4 行优先，逗号分隔 | 单位阵 |

## 相机参数格式

与 cam 仓库 `params.yaml` 一致：

- **intrinsics**: `[xi, fx, fy, cx, cy]` 鱼眼模型内参
- **distortion_coeffs**: `[k1, k2, p1, p2]` 畸变系数
- **T_ic** (可选): 4x4 相机到 IMU 外参矩阵（行优先）

## 批量处理（Python 脚本）

处理文件夹内所有 `*camera_1.jpg` ~ `*camera_4.jpg` 的图片，输出保持原文件名：

```bash
python3 batch_undistort.py <输入目录> <输出目录>
```

示例：

```bash
python3 batch_undistort.py /path/to/fisheye_images /path/to/output
python3 batch_undistort.py ./raw ./undistorted -f 180 -w 640
python3 batch_undistort.py ./raw ./out --dry-run  # 仅预览命令，不执行
```

## 示例

```bash
# 使用 cam 目录的 params.yaml
./fisheye_to_cylindrical -i ~/fisheye_shot.jpg -o ~/cyl.jpg -c ../params.yaml -n camera_1

# 批量处理（使用 Python 脚本）
python3 batch_undistort.py ./images ./output
```
