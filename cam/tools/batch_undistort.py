#!/usr/bin/env python3
"""
批量鱼眼去畸变：调用 fisheye_to_cylindrical 处理文件夹内 camera_1.jpg ~ camera_4.jpg 的图片。
输出文件名与原始保持一致，保存到指定输出目录。
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def find_camera_images(input_dir: Path) -> list[tuple[Path, str]]:
    """查找 *camera_1.jpg ~ *camera_4.jpg 的文件，返回 (路径, 相机名) 列表"""
    pattern = re.compile(r"(.+_)?camera_([1-4])\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
    results = []
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            cam_num = m.group(2)
            results.append((f, f"camera_{cam_num}"))
    return sorted(results, key=lambda x: (x[1], x[0].name))


def main():
    parser = argparse.ArgumentParser(
        description="批量将鱼眼图去畸变为柱面视角（camera_1~4）"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="输入图片所在目录",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="输出图片目录",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="相机参数 YAML 路径（默认 tools 同目录的 ../params.yaml）",
    )
    parser.add_argument(
        "-t", "--tool",
        type=Path,
        default=None,
        help="fisheye_to_cylindrical 可执行文件路径（默认同目录）",
    )
    parser.add_argument(
        "-f", "--fov",
        type=int,
        default=180,
        help="输出视场角，默认 180",
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=640,
        help="输出宽度，默认 640",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=480,
        help="输出高度，默认 480",
    )
    parser.add_argument(
        "-p", "--pitch-up",
        type=float,
        default=0.0,
        help="视角向上偏移角度（度），默认 0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令，不实际运行",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    tool = args.tool or (script_dir / "fisheye_to_cylindrical")
    config = args.config or (script_dir.parent / "params.yaml")

    if not tool.exists():
        print(f"错误: 未找到工具 {tool}，请先执行 make 编译", file=sys.stderr)
        sys.exit(1)
    if not config.exists():
        print(f"错误: 未找到配置 {config}", file=sys.stderr)
        sys.exit(1)
    if not args.input_dir.is_dir():
        print(f"错误: 输入目录不存在 {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = find_camera_images(args.input_dir)
    if not files:
        print(f"未在 {args.input_dir} 中找到 *camera_1.jpg ~ *camera_4.jpg 的图片")
        sys.exit(0)

    print(f"找到 {len(files)} 张图片，输出到 {args.output_dir}")
    for inp, cam_name in files:
        out_path = args.output_dir / inp.name
        cmd = [
            str(tool),
            "-i", str(inp),
            "-o", str(out_path),
            "-c", str(config),
            "-n", cam_name,
            "-f", str(args.fov),
            "-w", str(args.width),
            "--height", str(args.height),
        ]
        if args.pitch_up != 0:
            cmd.extend(["-p", str(args.pitch_up)])
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print(f"处理: {inp.name} -> {out_path.name}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"失败: {inp.name}", file=sys.stderr)
            sys.exit(ret.returncode)

    print("完成")


if __name__ == "__main__":
    main()
