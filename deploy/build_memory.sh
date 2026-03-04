#!/bin/bash
# ============================================================
# 记忆构建脚本 - 手动重新构建VPR记忆库
# 
# 用法:
#   bash deploy/build_memory.sh                        # 默认AnyLoc VLAD
#   bash deploy/build_memory.sh --method longclip      # 使用LongCLIP
#   bash deploy/build_memory.sh --method anyloc --agg gem  # AnyLoc GeM
#   bash deploy/build_memory.sh --help
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认参数
DATA_DIR="${PROJECT_DIR}/merged_labeled_data"
SAVE_PATH="${SCRIPT_DIR}/memory_nav/memory_cache"
VPR_METHOD="anyloc"
DINO_MODEL="dinov2_vitb14"
AGG_MODE="vlad"
NUM_CLUSTERS=8
DOMAIN="indoor"
MAX_IMG_SIZE=630
GPU_ID=0
CONDA_ENV="internnav"

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --data-dir DIR       标注数据目录 (默认: ${DATA_DIR})"
    echo "  --save-path PATH     保存路径前缀 (默认: ${SAVE_PATH})"
    echo "  --method METHOD      VPR方法: anyloc / longclip (默认: ${VPR_METHOD})"
    echo "  --dino-model MODEL   DINOv2模型: dinov2_vits14/vitb14/vitl14/vitg14 (默认: ${DINO_MODEL})"
    echo "  --agg MODE           聚合模式: vlad / gem (默认: ${AGG_MODE})"
    echo "  --num-clusters N     VLAD聚类数 (默认: ${NUM_CLUSTERS})"
    echo "  --domain DOMAIN      场景类型: indoor/urban/aerial (默认: ${DOMAIN})"
    echo "  --max-img-size N     最大图像边长 (默认: ${MAX_IMG_SIZE})"
    echo "  --gpu ID             GPU编号 (默认: ${GPU_ID})"
    echo "  --conda-env ENV      Conda环境名 (默认: ${CONDA_ENV})"
    echo "  -h, --help           显示帮助"
    exit 0
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)     DATA_DIR="$2";      shift 2;;
        --save-path)    SAVE_PATH="$2";     shift 2;;
        --method)       VPR_METHOD="$2";    shift 2;;
        --dino-model)   DINO_MODEL="$2";    shift 2;;
        --agg)          AGG_MODE="$2";      shift 2;;
        --num-clusters) NUM_CLUSTERS="$2";  shift 2;;
        --domain)       DOMAIN="$2";        shift 2;;
        --max-img-size) MAX_IMG_SIZE="$2";  shift 2;;
        --gpu)          GPU_ID="$2";        shift 2;;
        --conda-env)    CONDA_ENV="$2";     shift 2;;
        -h|--help)      usage;;
        *)              echo "未知参数: $1"; usage;;
    esac
done

# 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================================"
echo "  记忆构建 (Memory Build)"
echo "============================================================"
echo "  数据目录:    ${DATA_DIR}"
echo "  保存路径:    ${SAVE_PATH}"
echo "  VPR方法:     ${VPR_METHOD}"
if [ "${VPR_METHOD}" = "anyloc" ]; then
    echo "  DINOv2模型:  ${DINO_MODEL}"
    echo "  聚合模式:    ${AGG_MODE}"
    echo "  聚类数:      ${NUM_CLUSTERS}"
    echo "  场景类型:    ${DOMAIN}"
    echo "  最大图像尺寸: ${MAX_IMG_SIZE}"
fi
echo "  GPU:         ${GPU_ID}"
echo "  Conda环境:   ${CONDA_ENV}"
echo "============================================================"

python3 -c "
import sys, time, os
sys.path.insert(0, '${PROJECT_DIR}')
from deploy.memory_nav import MemoryBuilder

vpr_method = '${VPR_METHOD}'
save_path  = '${SAVE_PATH}'
data_dir   = '${DATA_DIR}'

anyloc_config = None
if vpr_method == 'anyloc':
    anyloc_config = {
        'dino_model':   '${DINO_MODEL}',
        'agg_mode':     '${AGG_MODE}',
        'num_clusters': ${NUM_CLUSTERS},
        'domain':       '${DOMAIN}',
        'max_img_size': ${MAX_IMG_SIZE},
    }

print('正在构建记忆...')
t0 = time.time()
builder = MemoryBuilder(
    vpr_method=vpr_method,
    anyloc_config=anyloc_config,
    device='cuda:0',
)
graph, vpr = builder.build_from_directory(data_dir, save_path=save_path)
elapsed = time.time() - t0

stats = builder.get_stats()
print()
print(f'构建完成! 耗时 {elapsed:.1f} 秒')
print(f'  节点数: {stats[\"graph\"][\"total_nodes\"]}')
print(f'  边数:   {stats[\"graph\"][\"total_edges\"]}')
print(f'  VPR维度: {stats[\"vpr\"][\"feature_dim\"]}')
print(f'  保存到: {save_path}')
"

echo ""
echo "构建完成 ✓"
