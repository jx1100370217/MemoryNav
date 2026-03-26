#!/bin/bash
# ============================================================
# 记忆构建脚本 - 手动重新构建VPR记忆库
# 
# 用法:
#   bash deploy/build_memory.sh                        # 默认AnyLoc VLAD
#   bash deploy/build_memory.sh --method anyloc --agg gem  # AnyLoc GeM
#   bash deploy/build_memory.sh --help
# ============================================================
set -e
# 网络代理
export https_proxy=http://10.24.99.86:10808 http_proxy=http://10.24.99.86:10808 all_proxy=socks5://10.24.99.86:10808

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 从 vpr_config.yaml 读取默认配置
VPR_CONFIG="${SCRIPT_DIR}/vpr_config.yaml"
if [ -f "${VPR_CONFIG}" ]; then
    _VPR_FROM_CFG=$(python3 -c "import yaml; c=yaml.safe_load(open('${VPR_CONFIG}')); print(c.get('vpr_method','selavpr'))" 2>/dev/null)
    _GPU_FROM_CFG=$(python3 -c "import yaml; c=yaml.safe_load(open('${VPR_CONFIG}')); d=c.get('device','cuda:0'); print(d.replace('cuda:',''))" 2>/dev/null)
fi

# 默认参数 (配置文件优先，命令行参数可覆盖)
DATA_DIR="${PROJECT_DIR}/merged_labeled_data"
SAVE_PATH="${PROJECT_DIR}/memory_nav/memory_cache"
VPR_METHOD="${_VPR_FROM_CFG:-selavpr}"
DINO_MODEL="dinov2_vitb14"
AGG_MODE="vlad"
_DEFAULT_NUM_CLUSTERS=8
NUM_CLUSTERS=8
DOMAIN="indoor"
MAX_IMG_SIZE=630
GPU_ID="${_GPU_FROM_CFG:-0}"
CONDA_ENV="internnav"

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --data-dir DIR       标注数据目录 (默认: ${DATA_DIR})"
    echo "  --save-path PATH     保存路径前缀 (默认: ${SAVE_PATH})"
    echo "  --method METHOD      VPR方法: selavpr/megaloc/effovpr/anyloc (默认: ${VPR_METHOD})"
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
    _CFG_CLUSTERS=$(python3 -c "import yaml; c=yaml.safe_load(open('${VPR_CONFIG}')); print(c.get('anyloc',{}).get('num_clusters',8))" 2>/dev/null)
    echo "  DINOv2模型:  ${DINO_MODEL}"
    echo "  聚合模式:    ${AGG_MODE}"
    echo "  聚类数:      ${_CFG_CLUSTERS:-${NUM_CLUSTERS}} (配置文件) / ${NUM_CLUSTERS} (命令行默认)"
    echo "  场景类型:    ${DOMAIN}"
    echo "  最大图像尺寸: ${MAX_IMG_SIZE}"
fi
echo "  GPU:         ${GPU_ID}"
echo "  Conda环境:   ${CONDA_ENV}"
echo "============================================================"

python3 -c "
import sys, time, os
sys.path.insert(0, '${PROJECT_DIR}')
from memory_nav import MemoryBuilder
from memory_nav.vpr_config_loader import load_vpr_config, get_anyloc_config

# 命令行参数通过环境变量传入
os.environ['VPR_METHOD'] = '${VPR_METHOD}'
cfg = load_vpr_config()
vpr_method = cfg['vpr_method']
save_path  = '${SAVE_PATH}'
data_dir   = '${DATA_DIR}'

anyloc_config = None
if vpr_method == 'anyloc':
    # 从统一配置文件读取 AnyLoc 参数
    anyloc_config = get_anyloc_config(cfg)
    # 命令行参数覆盖 (仅当用户显式传入 --dino-model 等时才生效)
    _cli_overrides = {}
    if '${DINO_MODEL}' != 'dinov2_vitb14':  _cli_overrides['dino_model'] = '${DINO_MODEL}'
    if '${AGG_MODE}' != 'vlad':             _cli_overrides['agg_mode'] = '${AGG_MODE}'
    if '${NUM_CLUSTERS}' != '${_DEFAULT_NUM_CLUSTERS}': _cli_overrides['num_clusters'] = ${NUM_CLUSTERS}
    if '${DOMAIN}' != 'indoor':             _cli_overrides['domain'] = '${DOMAIN}'
    if '${MAX_IMG_SIZE}' != '630':          _cli_overrides['max_img_size'] = ${MAX_IMG_SIZE}
    if _cli_overrides:
        anyloc_config.update(_cli_overrides)
        print(f'  命令行覆盖: {_cli_overrides}')

print('正在构建记忆...')
t0 = time.time()
builder = MemoryBuilder(
    vpr_method=vpr_method,
    anyloc_config=anyloc_config,
    device=cfg['device'],
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
