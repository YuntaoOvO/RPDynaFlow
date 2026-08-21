#!/bin/bash
# stage1_ago2_eval.sh — 阶段一：Ago2 跨结构评估
# 用途：运行 ago2_multiconf_eval.py 对三轮采样结果进行 Coverage 评估
# 输出：results/ago2/{r5,r10,r15}/multiconf_summary.csv + coverage_curves.png

set -e

SAMPLES_BASE="../results/samples/ago2"
OUT_BASE="../results/ago2"

echo "=========================================="
echo "阶段一：Ago2 跨结构评估"
echo "=========================================="
echo ""

for round in r5 r10 r15; do
    samples_dir="$SAMPLES_BASE/$round"
    if [[ ! -d "$samples_dir" ]]; then
        echo "[SKIP] 样本目录不存在: $samples_dir"
        continue
    fi

    # 检查必需文件
    if [[ ! -f "$samples_dir/4W5N.npz" ]] || [[ ! -f "$samples_dir/9K6T.npz" ]]; then
        echo "[SKIP] $round: 缺少 4W5N.npz 或 9K6T.npz"
        continue
    fi

    out_dir="$OUT_BASE/$round"
    mkdir -p "$out_dir"

    echo "[评估] 轮次: $round"
    echo "  样本: $samples_dir"
    echo "  输出: $out_dir"
    echo ""

    python ago2_multiconf_eval.py \
        --samples-dir "$samples_dir" \
        --out "$out_dir"

    echo ""
    echo "[完成] $round 评估结果写入 $out_dir"
    echo ""
done

echo "=========================================="
echo "评估完成"
echo ""
echo "查看汇总："
for round in r5 r10 r15; do
    csv="$OUT_BASE/$round/multiconf_summary.csv"
    if [[ -f "$csv" ]]; then
        echo ""
        echo "=== $round ==="
        cat "$csv"
    fi
done
echo ""
echo "图件位置: $OUT_BASE/{r5,r10,r15}/coverage_curves.png"
echo "=========================================="
