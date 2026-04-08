'''
MemoryNav Auto Mapper Module

自动建图模块，用于从原始图像序列自动生成符合 merged_labeled_data 格式的导航图结构。

主要功能：
- 自动节点生成
- VPR 距离估计
- DINOv3 子图提取
- 自动地标命名
- 格式验证

使用方式：
    python offline_mapper/run_auto_map.py \
        --input_dir memory_test_data-2 \
        --output_dir offline_mapper/merged_labeled_data \
        --start_id 60 \
        --vpr_config deploy/vpr_config.yaml
'''

__version__ = '1.0.0'
__author__ = 'MemoryNav Auto Mapper'