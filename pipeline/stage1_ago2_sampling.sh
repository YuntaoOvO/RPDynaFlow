#!/bin/bash
# stage1_ago2_sampling.sh — 阶段一：Ago2 apo/holo 采样（3 轮数据量递进）
# 用途：为 4W5N（apo）和 9K6T（holo）生成构象系综，使用 r5/r10/r15 三个检查点
# 输出：results/samples/ago2/{r5,r10,r15}/{4W5N,9K6T}.npz

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/examples}"
export PDB_DIR="${PDB_DIR:-$REPO_ROOT/examples/pdb}"

CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
OUT_BASE="${OUT_BASE:-$REPO_ROOT/results/samples/ago2}"
SYSTEMS="4W5N,9K6T"
N_GEN=200
STEPS=30
CHUNK_SIZE=20

echo "=========================================="
echo "阶段一：Ago2 apo/holo 采样"
echo "=========================================="
echo ""

# 检查检查点
for round in r5 r10 r15; do
    ckpt="$CKPT_DIR/flow_model_${round}.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[WARN] 检查点缺失: $ckpt"
        echo "       跳过此轮次，继续其他轮次"
        continue
    fi

    out_dir="$OUT_BASE/$round"
    mkdir -p "$out_dir"

    echo "[采样] 轮次: $round"
    echo "  检查点: $ckpt"
    echo "  输出: $out_dir"
    echo "  系统: $SYSTEMS"
    echo "  样本数/系统: $N_GEN"
    echo ""

    python gen_ensembles.py \
        --ckpt "$ckpt" \
        --systems "$SYSTEMS" \
        --n-gen "$N_GEN" \
        --out "$out_dir" \
        --static \
        --steps "$STEPS" \
        --chunk-size "$CHUNK_SIZE" \
        --no-pdb

    echo ""
    echo "[完成] $round: $(ls -1 $out_dir/*.npz 2>/dev/null | wc -l) 个 npz 文件"
    echo ""
done

echo "=========================================="
echo "采样完成"
echo "输出位置: $OUT_BASE/{r5,r10,r15}/"
echo ""
echo "下一步：运行跨结构评估"
echo "  bash stage1_ago2_eval.sh"
echo "=========================================="
