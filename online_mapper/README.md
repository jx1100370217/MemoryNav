# online_mapper — 在线主动建图模块

与 `offline_mapper/` 完全解耦的流式在线建图方案, 产出 schema 完全兼容的
`merged_labeled_data/` 目录, 现有导航 runtime 可直接消费。

## 快速开始
```bash
cd /home/ubuntu/Disk/codes/jianxiong/MemoryNav
/home/ubuntu/miniconda3/envs/internvla/bin/python online_mapper/run_online_map.py \
    --input memory_test_data \
    --output online_mapper/output/merged_labeled_data
```

可选参数:
- `--no_depth` 关闭 Depth-Anything-V2 (回退均匀深度)
- `--no_grounding_dino` 关闭开放集检测
- `--log_level DEBUG`

## 验证输出
```bash
/home/ubuntu/miniconda3/envs/internvla/bin/python offline_mapper/validate_output.py \
    online_mapper/output/merged_labeled_data
```

## 单元测试
```bash
/home/ubuntu/miniconda3/envs/internvla/bin/python online_mapper/tests/test_pipeline.py
```

## 目录
```
online_mapper/
  config.py                      # 全局 dataclass 配置
  run_online_map.py              # CLI 入口
  core/
    stream_loader.py             # 帧流模拟
    online_mapper_core.py        # 主循环
  geometry/
    depth_estimator.py           # Depth-Anything-V2-Small
    pose_graph.py                # scipy 最小二乘位姿图
    occupancy.py                 # 2D 占据栅格
  topology/
    keyframe_selector.py         # 多触发关键帧
    loop_closure.py              # 全局 VPR 闭环
    frontier_nbv.py              # frontier + NBV 评分
    graph.py                     # 真实图结构
  semantics/
    open_set_detector.py         # Grounding-DINO
    scene_graph.py               # 层次场景图
    semantic_dedup.py            # 同 room/landmark 合并
  output/
    merged_data_writer.py        # merged_labeled_data 产出
  tests/
    test_pipeline.py             # 单元测试
```

## 输出
- `online_mapper/output/merged_labeled_data/{node_id}/` — 严格兼容
- `online_mapper/output/scene_graph.json` — floor→room→node→object 层级
- `online_mapper/output/pose_graph.json` — 优化后 metric 位姿图
- `online_mapper/output/online_mapping_log.jsonl` — 每帧在线决策日志
- `online_mapper/output/metrics.json` — 统计 + 运行时分布

## 模拟在线说明
真实机器人未接入。StreamLoader 按时间戳逐帧 yield, 主循环
只使用"当前及之前帧"的信息做 keyframe/闭环/NBV 决策,
日志对比可见其在线行为。
