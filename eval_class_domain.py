"""
Per-(class, domain) accuracy evaluation and delta computation.

Usage:
    python eval_class_domain.py \
        --baseline_dir /output/baseline/seed1/ForgetDomainNone/... \
        --unlearned_dir /output/unlearned/seed1/ForgetDomain1/sketch/... \
        --root "/home/owais/machine unlearning/ebm_unlearning/data/domainnet" \
        --config-file configs/trainers/vit_b16_ep50.yaml \
        --dataset-config-file configs/datasets/domainnet_df.yaml
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from collections import defaultdict

# Add repo root to path so dassl + trainers are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EVAL_CLASSES = ["tiger", "lion", "bear", "zebra", "dog", "car", "guitar"]
EVAL_DOMAINS = ["real", "sketch", "painting"]

DOMAIN_LIST = ["clipart", "painting", "real", "sketch"]


def load_model_and_dm(checkpoint_dir, args):
    from train_loop import setup_cfg
    from dassl.engine import build_trainer

    # Patch args for eval-only
    args.eval_only = True
    args.model_dir = checkpoint_dir
    args.load_epoch = getattr(args, "load_epoch", None)
    args.forget_domains = ["none"]
    args.forget_classes = []
    args.no_train = True

    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.TRAIN.PRINT_FREQ = 9999
    cfg.freeze()

    trainer = build_trainer(cfg)
    trainer.load_model(checkpoint_dir, epoch=args.load_epoch)
    return trainer


@torch.no_grad()
def compute_per_class_domain_acc(trainer):
    """Returns dict: {(classname, domain_name): (correct, total)}"""
    trainer.set_model_mode("eval")
    loader = trainer.test_loader

    classnames = trainer.dm.dataset.classnames
    results = defaultdict(lambda: [0, 0])  # [correct, total]

    for batch in loader:
        images = batch["img"].to(trainer.device)
        labels = batch["label"].to(trainer.device)
        domains = batch["domain"].to(trainer.device)

        output, _, _, _, _ = trainer.model(images)
        preds = output.argmax(dim=1)

        for pred, lbl, dom in zip(preds, labels, domains):
            cname = classnames[lbl.item()]
            dname = DOMAIN_LIST[dom.item()]
            results[(cname, dname)][1] += 1
            if pred.item() == lbl.item():
                results[(cname, dname)][0] += 1

    return {k: (v[0] / v[1] * 100 if v[1] > 0 else 0.0) for k, v in results.items()}


def print_delta_table(baseline_acc, unlearned_acc):
    col = f"{'base':>7} {'unl':>7} {'∆':>6}"
    header = f"{'Class':<12}  {'REAL':^23}  {'SKETCH':^23}  {'PAINTING':^23}  Interpretation"
    subhdr = f"{'':12}  {col}  {col}  {col}"
    print(header)
    print(subhdr)
    print("-" * len(header))

    for cls in EVAL_CLASSES:
        row = f"{cls:<12}"
        for dom in EVAL_DOMAINS:
            key = (cls, dom)
            base = baseline_acc.get(key, 0.0)
            unl = unlearned_acc.get(key, 0.0)
            delta = unl - base
            row += f"  {base:>6.1f}% {unl:>6.1f}% {delta:>+6.1f}%"

        interp = "FORGET TARGET" if cls == "tiger" else "cross-class"
        print(row + f"  {interp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--unlearned_dir", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-config-file", required=True)
    parser.add_argument("--trainer", default="IVLP_VL_Adapter_Prompt")
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_name", default="domainnet_df")
    parser.add_argument("--domainloss_weight", type=float, default=0.0)
    parser.add_argument("--mmd_weight", type=float, default=0.0)
    parser.add_argument("--use_domain_cls_loss", action="store_true")
    parser.add_argument("--is_domain_divided", action="store_true")
    parser.add_argument("--domain_class_divided", action="store_true")
    parser.add_argument("--source-domains", nargs="+", default=[])
    parser.add_argument("--target-domains", nargs="+", default=[])
    parser.add_argument("--transforms", nargs="+", default=[])
    parser.add_argument("--backbone", default="")
    parser.add_argument("--head", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--output-dir", default="/tmp/eval_tmp")
    parser.add_argument("--csv_file_path", default="/tmp/eval_tmp.csv")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading baseline model...")
    baseline_trainer = load_model_and_dm(args.baseline_dir, args)

    print("Computing baseline per-(class, domain) accuracy...")
    baseline_acc = compute_per_class_domain_acc(baseline_trainer)

    print("Loading unlearned model...")
    unlearned_trainer = load_model_and_dm(args.unlearned_dir, args)

    print("Computing unlearned per-(class, domain) accuracy...")
    unlearned_acc = compute_per_class_domain_acc(unlearned_trainer)

    print("\n" + "=" * 80)
    print("ACCURACY AND DELTA TABLE  (forget target: tiger/sketch)")
    print("=" * 80)
    print_delta_table(baseline_acc, unlearned_acc)
    print("=" * 80)


if __name__ == "__main__":
    main()
