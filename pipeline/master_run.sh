#!/bin/bash
# master_run.sh — 主控脚本：完整执行三阶段评估流程
# 用途：从采样到最终报告的一键运行

set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PIPELINE_DIR"

echo "=========================================="
echo "RPDynaFlow 评估流程 — 主控脚本"
echo "=========================================="
echo ""

# 阶段零：冒烟测试（可选）
read -p "运行阶段零冒烟测试？(y/n，默认 n): " RUN_SMOKE
RUN_SMOKE=${RUN_SMOKE:-n}
if [[ "$RUN_SMOKE" == "y" ]]; then
    echo ""
    echo "========== 阶段零：冒烟测试 =========="
    bash smoke_test.sh
    echo ""
fi

# 阶段一：Ago2 采样与评估
read -p "运行阶段一（Ago2 采样与评估）？(y/n，默认 y): " RUN_STAGE1
RUN_STAGE1=${RUN_STAGE1:-y}
if [[ "$RUN_STAGE1" == "y" ]]; then
    echo ""
    echo "========== 阶段一A：Ago2 采样 =========="
    bash stage1_ago2_sampling.sh
    echo ""
    echo "========== 阶段一B：跨结构评估 =========="
    bash stage1_ago2_eval.sh
    echo ""
    echo "========== 阶段一C：状态分析 =========="
    for round in r5 r10 r15; do
        samples_dir="../results/samples/ago2/$round"
        if [[ -d "$samples_dir" ]]; then
            python ago2_state_analysis.py --samples-dir "$samples_dir"
        fi
    done
    echo ""
fi

# 阶段二：BioEmu 对比（需要手动准备 BioEmu 样本）
read -p "运行阶段二（BioEmu 对比）？(y/n，默认 n，需事先运行 BioEmu): " RUN_STAGE2
RUN_STAGE2=${RUN_STAGE2:-n}
if [[ "$RUN_STAGE2" == "y" ]]; then
    echo ""
    echo "========== 阶段二：BioEmu 对比 =========="
    BIOEMU_DIR="../data/bioemu_samples"
    if [[ ! -d "$BIOEMU_DIR" ]] || [[ -z "$(ls -A $BIOEMU_DIR 2>/dev/null)" ]]; then
        echo "[WARN] BioEmu 样本目录为空: $BIOEMU_DIR"
        echo "       请手动运行 BioEmu 生成 4W5N/9K6T 系综"
        echo "       跳过此阶段"
    else
        python compare_bioemu.py \
            --dynaflow ../results/samples/ago2/r15 \
            --bioemu "$BIOEMU_DIR"
    fi
    echo ""
fi

# 阶段三：ESMDynamic 覆盖层（需要手动准备 ESMDynamic 输出）
read -p "运行阶段三（ESMDynamic 可视化）？(y/n，默认 n，需事先运行 ESMDynamic): " RUN_STAGE3
RUN_STAGE3=${RUN_STAGE3:-n}
if [[ "$RUN_STAGE3" == "y" ]]; then
    echo ""
    echo "========== 阶段三：ESMDynamic 覆盖层 =========="
    ESM_BASE="../data/esmdynamic_outputs"
    for pid in 4W5N 9K6T; do
        esm_dir="$ESM_BASE/$pid"
        npz="../results/samples/ago2/r15/${pid}.npz"
        out="../results/esmdynamic_overlay/$pid"
        if [[ -d "$esm_dir" ]] && [[ -f "$npz" ]]; then
            python visualize_esmdynamic_overlay.py \
                --dynaflow "$npz" \
                --esmdynamic "$esm_dir" \
                --out "$out" \
                --temp 320
        else
            echo "[SKIP] $pid: ESMDynamic 输出或 DynaFlow 样本缺失"
        fi
    done
    echo ""
fi

# 生成最终报告
echo ""
echo "========== 生成最终报告 =========="
python generate_final_report.py

echo ""
echo "=========================================="
echo "评估流程完成"
echo ""
echo "主要输出："
echo "  - results/samples/ago2/  （采样系综）"
echo "  - results/ago2/          （跨结构评估 + 状态分析）"
echo "  - results/comparison/    （BioEmu 对比）"
echo "  - results/esmdynamic_overlay/  （ESMDynamic 可视化）"
echo "  - results/FINAL_REPORT.md  （汇总报告）"
echo "=========================================="
