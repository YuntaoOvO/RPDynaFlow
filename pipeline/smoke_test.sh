#!/bin/bash
# smoke_test.sh — 冒烟测试：验证推理管线可用性与显存控制
# 用途：生成 10 个 4W5N 样本，检查语法、显存、输出完整性
# 运行前需：conda 环境激活（numpy/torch/MDAnalysis）、检查点存在

set -e

# Default to the bundled Ago2 examples; override with DATA_DIR for other data.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/examples}"

CKPT="${CKPT:-$REPO_ROOT/checkpoints/flow_model_r15.pt}"
OUT="/tmp/smoke_test_$(date +%s)"
SYSTEM="4W5N"
N_GEN=10

echo "[smoke_test] 验证推理管线"
echo "  检查点: $CKPT"
echo "  输出目录: $OUT"
echo "  系统: $SYSTEM"
echo "  样本数: $N_GEN"
echo ""

# 前置检查
if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] 检查点不存在: $CKPT"
    echo "        请先训练模型或指定正确路径"
    exit 1
fi

if ! python -c "import torch; import numpy; import MDAnalysis" 2>/dev/null; then
    echo "[ERROR] Python 依赖缺失（torch/numpy/MDAnalysis）"
    echo "        请先创建环境：conda env create -f ../environment.yml && conda activate rpdynaflow"
    exit 1
fi

# 冒烟测试
echo "[1/3] 运行推理（静态 PDB 特征化模式，10 样本）..."
python gen_ensembles.py \
    --ckpt "$CKPT" \
    --systems "$SYSTEM" \
    --n-gen "$N_GEN" \
    --out "$OUT" \
    --static \
    --no-pdb \
    --steps 30 \
    --chunk-size 5

echo ""
echo "[2/3] 检查输出..."
NPZ="$OUT/${SYSTEM}.npz"
if [[ ! -f "$NPZ" ]]; then
    echo "[ERROR] 输出文件不存在: $NPZ"
    exit 1
fi

# 验证 npz 内容
python - <<EOF
import numpy as np
import sys

npz = "$NPZ"
z = np.load(npz, allow_pickle=True)

print(f"[INFO] {npz}")
print(f"  keys: {list(z.keys())}")
print(f"  gen.shape: {z['gen'].shape}")
print(f"  static.shape: {z['static'].shape}")
print(f"  finite: {np.isfinite(z['gen']).all()}")
print(f"  元素类型混合: {len(np.unique(z['atom_elements']))} 类")

# 验证关键条件
assert z['gen'].shape[0] == $N_GEN, f"样本数不匹配: {z['gen'].shape[0]} != $N_GEN"
assert np.isfinite(z['gen']).all(), "生成坐标包含 NaN/Inf"
assert len(np.unique(z['atom_elements'])) > 1, "元素类型单一（可能特征化错误）"
print("\n[PASS] 输出验证通过")
EOF

if [[ $? -ne 0 ]]; then
    echo "[ERROR] 输出验证失败"
    exit 1
fi

echo ""
echo "[3/3] 清理..."
rm -rf "$OUT"

echo ""
echo "=========================================="
echo "[SUCCESS] 冒烟测试通过"
echo "推理管线工作正常，可以开始阶段一采样"
echo "=========================================="
