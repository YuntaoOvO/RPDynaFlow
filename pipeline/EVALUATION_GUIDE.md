# RPDynaFlow 评估流程使用指南

## 概述

本流程实现 Ago2 apo/holo 系统（4W5N/9K6T）的构象系综采样与多维度评估，包括：

1. **跨结构 Coverage 评估**：BioEmu 风格的 apo↔holo 覆盖指标
2. **状态特异性分析**：RMSD 分布、结构域响应、接触差异
3. **BioEmu 基线对比**：与生成式基线的量化对比
4. **ESMDynamic 覆盖层**：动态接触预测验证（ROC/相关性）

---

## 前置准备

### 1. 环境依赖

```bash
# Python 3.8+
pip install numpy scipy pandas matplotlib scikit-learn MDAnalysis

# 或使用 conda 环境
conda activate calvados  # 或 pebble_env
```

### 2. 检查点文件

确保存在以下检查点（至少一个）：

```
results/checkpoints/
├── flow_model_r5.pt   # 5 轮训练
├── flow_model_r10.pt  # 10 轮训练
└── flow_model_r15.pt  # 15 轮训练（推荐）
```

验证检查点：

```bash
cd DynaFlow/pipeline
python3 verify_checkpoints.py
```

### 3. 结构文件

确保 PDB 文件可访问：

```
RNA-protein complexes/
├── 4W5N/
│   └── *.pdb  # apo（仅 guide）
└── 9K6T/
    └── *.pdb  # holo（guide+target 三元复合物）
```

---

## 快速开始

### 一键运行（推荐）

```bash
cd DynaFlow/pipeline
bash master_run.sh
```

交互式选择运行的阶段：
- 阶段零：冒烟测试（验证环境，可选）
- 阶段一：Ago2 采样与跨结构评估（必选）
- 阶段二：BioEmu 对比（需事先准备 BioEmu 样本）
- 阶段三：ESMDynamic 覆盖层（需事先运行 ESMDynamic）

---

## 分步运行

### 阶段零：冒烟测试

验证推理管线和环境配置：

```bash
cd DynaFlow/pipeline
bash smoke_test.sh
```

**输出**：临时 10 样本 npz（自动清理），验证通过后显示"[SUCCESS]"。

---

### 阶段一A：Ago2 采样

为 4W5N/9K6T 生成 200 样本系综（3 轮检查点）：

```bash
bash stage1_ago2_sampling.sh
```

**输出**：
```
results/samples/ago2/
├── r5/
│   ├── 4W5N.npz  # 200 样本
│   └── 9K6T.npz
├── r10/
│   ├── 4W5N.npz
│   └── 9K6T.npz
└── r15/
    ├── 4W5N.npz
    └── 9K6T.npz
```

**时间估计**：约 10-30 分钟/轮（取决于 GPU）。

---

### 阶段一B：跨结构评估

运行 BioEmu 风格的 Coverage 评估：

```bash
bash stage1_ago2_eval.sh
```

**输出**：
```
results/ago2/
├── r5/
│   ├── multiconf_summary.csv      # Coverage@3Å/5Å, RMSD 统计
│   ├── multiconf_detail.json      # 详细曲线数据
│   └── coverage_curves.png        # 可视化
├── r10/
└── r15/
```

**关键指标**：
- `cov_3A`/`cov_5A`: 3Å/5Å 阈值下的覆盖率（0-1）
- `rmsd_min`: 最佳采样 RMSD
- `cov_auc`: Coverage 曲线下面积

---

### 阶段一C：状态分析

RMSD 分布、结构域拆分、接触差异：

```bash
python3 ago2_state_analysis.py --samples-dir ../results/samples/ago2/r15
```

**输出**（`results/ago2/r15_state_analysis/`）：
- `rmsd_distributions.png`: 4 子图（自 RMSD vs 跨结构 RMSD）
- `domain_rmsd_barplot.png`: 6 个结构域的响应
- `contact_freq_diff.png`: 接触频率差异热图
- `state_analysis_summary.csv`: 汇总表格

---

### 阶段二：BioEmu 对比

**前置**：手动运行 BioEmu 生成 4W5N/9K6T 系综，放置于 `data/bioemu_samples/`。

```bash
python3 compare_bioemu.py \
    --dynaflow ../results/samples/ago2/r15 \
    --bioemu ../data/bioemu_samples
```

**输出**（`results/comparison/bioemu/`）：
- `comparison_summary.csv`: 两方法的 Coverage/RMSD 对比
- `coverage_all.png`: 叠加曲线图

---

### 阶段三：ESMDynamic 覆盖层

**前置**：手动运行 ESMDynamic 生成动态接触预测，放置于 `data/esmdynamic_outputs/`。

```bash
python3 visualize_esmdynamic_overlay.py \
    --dynaflow ../results/samples/ago2/r15/4W5N.npz \
    --esmdynamic ../data/esmdynamic_outputs/4W5N \
    --out ../results/esmdynamic_overlay/4W5N \
    --temp 320
```

**输出**（`results/esmdynamic_overlay/4W5N/`）：
- `heatmap_comparison.png`: 预测 vs 实际接触频率（3 子图）
- `roc_curve.png`: 二分类 ROC（接触 vs 非接触）
- `residue_dynamics.png`: 残基级动态性评分
- `metrics.txt`: MAE, Pearson 相关, ROC AUC

---

## 最终报告

所有阶段运行完成后，生成汇总报告：

```bash
python3 generate_final_report.py
```

**输出**：`results/FINAL_REPORT.md`（Markdown 格式，包含所有表格与图件引用）。

---

## 常见问题

### Q1: Python 可执行文件路径问题

**症状**：运行脚本时报错"ModuleNotFoundError: No module named 'torch'"，但 `python -c "import torch"` 成功。

**原因**：脚本使用 `python3` 而当前 conda 环境的 Python 可执行文件为 `python`。

**解决**：
```bash
# 验证当前环境
which python
python --version
python -c "import torch; import numpy; import MDAnalysis"  # 应该成功

# 如果上述命令通过，说明依赖已安装
# 脚本已修复为使用 `python` 而非 `python3`
```

### Q2: 显存不足（CUDA OOM）

**解决**：调整 `--chunk-size` 参数（默认 20）：

```bash
python gen_ensembles.py --ckpt <ckpt> --systems 4W5N \
    --n-gen 200 --out <out> --static --chunk-size 5
```

或修改 `stage1_ago2_sampling.sh` 中的 `CHUNK_SIZE` 变量。

### Q3: 检查点缺失

**排查**：
```bash
python verify_checkpoints.py
```

如缺失，需先训练模型或从备份恢复。

### Q4: PDB 文件找不到

**排查**：检查环境变量 `PDB_DIR`（默认 `../RNA-protein complexes`），或使用：

```bash
export PDB_DIR=/path/to/pdb/directory
```

### Q5: ESMDynamic 输出格式不匹配

`esmdynamic_overlay.py` 使用 glob 通配符适配多种命名（`*frequency_pred_320K.txt`）。如仍失败，检查文件名是否包含温度后缀。

---

## 文件结构

```
DynaFlow/pipeline/
├── smoke_test.sh                     # 冒烟测试
├── stage1_ago2_sampling.sh           # 阶段一A
├── stage1_ago2_eval.sh               # 阶段一B
├── ago2_multiconf_eval.py            # 跨结构评估核心
├── ago2_state_analysis.py            # 状态分析
├── compare_bioemu.py                 # BioEmu 对比
├── visualize_esmdynamic_overlay.py   # ESMDynamic 覆盖层
├── generate_final_report.py          # 报告生成
├── master_run.sh                     # 主控脚本
├── verify_checkpoints.py             # 检查点验证
└── EVALUATION_GUIDE.md               # 本文档

results/
├── samples/ago2/                     # 采样系综
├── ago2/                             # 跨结构评估 + 状态分析
├── comparison/                       # 基线对比
├── esmdynamic_overlay/               # ESMDynamic 验证
└── FINAL_REPORT.md                   # 汇总报告
```

---

## 引用

本评估流程整合以下方法学：

1. **BioEmu**: Jing et al., *Nature Methods*, 2023
2. **ESMDynamic**: *Nature Communications*, 2025 (DOI: 10.1101/2025.08.20.671365)
3. **mdCATH**: 动态接触数据集（5398 蛋白，~464 ns 轨迹）

---

## 许可证

本流程代码遵循 MIT License。数据使用需遵守原始数据集的许可条款。
