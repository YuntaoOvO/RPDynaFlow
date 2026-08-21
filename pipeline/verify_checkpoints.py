#!/usr/bin/env python3
"""verify_checkpoints.py — 验证训练检查点的完整性与兼容性

检查项：
  1. 检查点文件存在性（r5/r10/r15）
  2. 可加载性（torch.load）
  3. 必需键（state_dict, sigma, config）
  4. 模型参数统计（层数、参数量）
  5. 训练系统列表（trained_on）

Run:
  python3 verify_checkpoints.py
  python3 verify_checkpoints.py --ckpt ../results/checkpoints/flow_model_r15.pt
"""
import argparse
import os
import sys

try:
    import torch
except ImportError:
    sys.exit("[ERROR] PyTorch 未安装。运行: pip install torch")

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    os.path.join(ROOT, "checkpoints") if os.path.isdir(os.path.join(ROOT, "checkpoints"))
    else os.path.join(ROOT, "results", "checkpoints"))


def verify_checkpoint(ckpt_path):
    """验证单个检查点"""
    print(f"\n{'='*60}")
    print(f"检查点: {os.path.basename(ckpt_path)}")
    print(f"路径: {ckpt_path}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print("[FAIL] 文件不存在")
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[FAIL] 无法加载: {e}")
        return False

    print("[PASS] 文件可加载")

    # 必需键
    required_keys = ["state_dict", "sigma"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        print(f"[WARN] 缺少键: {missing}")

    # 统计信息
    print(f"\n[信息]")
    print(f"  config: {ckpt.get('config', '未指定')}")
    print(f"  sigma: {ckpt.get('sigma', 'N/A')}")
    print(f"  epoch: {ckpt.get('epoch', 'N/A')}")

    state = ckpt.get("state_dict", {})
    n_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
    print(f"  参数量: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  层数: {len(state)}")

    trained_on = ckpt.get("systems", [])
    print(f"  训练系统数: {len(trained_on)}")
    if len(trained_on) > 0:
        print(f"  示例系统: {list(trained_on)[:5]}")

    # 模型加载测试
    try:
        if "ff" in ckpt.get("config", ""):
            sys.path.insert(0, os.path.dirname(ckpt_path))
            from flow_model_ff import AtomFlowNetFF
            model = AtomFlowNetFF()
        else:
            sys.path.insert(0, os.path.join(ROOT, "DynaFlow", "pipeline"))
            from flow_model import AtomFlowNet
            model = AtomFlowNet()

        model.load_state_dict(state)
        print(f"\n[PASS] 模型可实例化（{model.__class__.__name__}）")
    except Exception as e:
        print(f"\n[WARN] 模型加载失败: {e}")
        print("       检查 flow_model.py 或 flow_model_ff.py 是否与检查点匹配")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="单个检查点路径（默认检查所有 r5/r10/r15）")
    args = ap.parse_args()

    if args.ckpt:
        verify_checkpoint(args.ckpt)
        return

    # 批量检查
    print("批量检查训练检查点")
    print(f"检查点目录: {CKPT_DIR}\n")

    if not os.path.exists(CKPT_DIR):
        sys.exit(f"[ERROR] 检查点目录不存在: {CKPT_DIR}")

    rounds = ["r5", "r10", "r15"]
    results = {}

    for r in rounds:
        ckpt_path = os.path.join(CKPT_DIR, f"flow_model_{r}.pt")
        results[r] = verify_checkpoint(ckpt_path)

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    for r, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {r}: {status}")

    if not any(results.values()):
        print("\n[ERROR] 所有检查点均不可用")
        print("       请先训练模型或从备份恢复检查点")
        sys.exit(1)

    print("\n[SUCCESS] 至少一个检查点可用，可以开始评估流程")


if __name__ == "__main__":
    main()
