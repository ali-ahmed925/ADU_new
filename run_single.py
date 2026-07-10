"""
Single-run wrapper — trains one experiment without the power-set sweep in train_loop.py.

Baseline (no forgetting):
    python run_single.py --root ... --forget_domains none

Unlearning (tiger/sketch):
    python run_single.py --root ... --forget_domains sketch --forget_classes tiger
"""
import argparse
import os
import sys
import os.path as osp

from train_loop import main, setup_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainer", type=str, default="IVLP_VL_Adapter_Prompt")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--forget_domains", default=[], nargs="*")
    parser.add_argument("--forget_classes", default=[], nargs="*")
    parser.add_argument("--mmd_weight", type=float, default=0.0)
    parser.add_argument("--domainloss_weight", type=float, default=0.0)
    parser.add_argument("--use_domain_cls_loss", action="store_true")
    parser.add_argument("--is_domain_divided", action="store_true")
    parser.add_argument("--domain_class_divided", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--source-domains", nargs="+", default=[])
    parser.add_argument("--target-domains", nargs="+", default=[])
    parser.add_argument("--transforms", nargs="+", default=[])
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--csv_file_path", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.csv_file_path:
        args.csv_file_path = osp.join(args.output_dir, "results.csv")

    results = main(args)
    print("\nDone.")
    if results:
        print(f"  A (retain acc):   {results.get('A', 'N/A'):.4f}")
        print(f"  F (forget error): {results.get('F', 'N/A'):.4f}")
        print(f"  H (harmonic):     {results.get('H', 'N/A'):.4f}")
