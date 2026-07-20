"""
Phase-0 diagnostic: Open-vocabulary domain forgetting.

QUESTION
--------
ADU is trained to forget a domain, but the forget loss only ever sees the
training vocabulary (the classes present in the 8-shot train set). Does the
"forgetting" it learns generalize to classes it never trained on, or does it
only suppress the specific (class, domain) CELLS it saw?

METHOD (everything else identical to the ADU paper)
---------------------------------------------------
- Mini-DomainNet (DomainNetMiniPaperDF), 126 classes, 4 domains.
- Forget domain: sketch (single domain, single seed -> fast proxy).
- Hold out HELDOUT_NUM classes from TRAINING ONLY. All 126 text anchors stay
  available at test time (see datasets/domainnet_mini_paper_df.py).
- Train stock ADU: forget_loss_type=entropy, DDL (gamma=30, lambda=10), InstaPG.
- At test, measure top-1 accuracy in a 2x2 grid:
        {SEEN, HELD-OUT} classes  x  {RETAIN domains, FORGET domain}

READOUT
-------
Suppression is how far forget-domain accuracy sits below the same split's
retain-domain accuracy (a per-split control that cancels the fact that held-out
classes have lower absolute accuracy):
    suppression = 1 - acc_forget / acc_retain
    transfer    = suppression_heldout / suppression_seen

  transfer ~ 1  -> forgetting TRANSFERS to unseen vocab. ADU forgets the domain.
                   (Branch B: open-vocab direction is weak; ADU stronger than we think.)
  transfer ~ 0  -> forgetting does NOT transfer. ADU forgets CELLS, not domains.
                   (Branch A: the strong-paper world; vocabulary randomization is essential.)
  in between    -> partial (Branch C).

USAGE
-----
    conda run -n myn_again python phase0_diagnostic.py \
        --gpu 0 --forget sketch --seed 1 --heldout_num 26 \
        --output-dir ~/adu_results/phase0_sketch_s1
"""

import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

import torch
from tqdm import tqdm

from dassl.utils import set_random_seed, setup_logger
from dassl.engine import build_trainer

# Register datasets + trainer (same modules train_loop.py imports).
import datasets.domainnet_df          # noqa: F401
import datasets.office_home_df        # noqa: F401
import datasets.domainnet_mini_df     # noqa: F401
import datasets.domainnet_mini_paper_df  # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401

from train_loop import setup_cfg

# Paper-exact reproduction hyper-parameters (see run_domainnet_mini.sh).
DATA_ROOT = "/home/owais/machine unlearning/ebm_unlearning/data/domainnet"
DATASET_CFG = "configs/datasets/domainnet_mini_paper_df.yaml"
TRAINER_CFG = "configs/trainers/vit_b16_ep50.yaml"  # paper config: bs=8, 50ep
DOMAINLOSS_WEIGHT = 30.0  # gamma
MMD_WEIGHT = 10.0         # lambda


def build_args(cli):
    """A full argparse-shaped Namespace, so train_loop.setup_cfg() is happy.

    Values mirror run_domainnet_mini.sh exactly, plus the held-out flags. The
    only deviations from stock ADU are heldout_num/heldout_seed.
    """
    return SimpleNamespace(
        # paths / core
        root=DATA_ROOT,
        output_dir=cli.output_dir,
        resume="",
        seed=cli.seed,
        dataset_config_file=DATASET_CFG,
        config_file=TRAINER_CFG,
        trainer="IVLP_VL_Adapter_Prompt",
        dataset_name="domainnet_mini_paper_df",
        backbone="",
        head="",
        source_domains=None,
        target_domains=None,
        transforms=None,
        num_shots=8,
        # forgetting (ADU baseline path)
        forget_domains=[cli.forget],
        forget_classes=[],
        forget_loss_type="entropy",       # ADU: L_forget = CE-to-uniform = entropy max
        no_retain_loss=False,
        forget_weight=1.0,
        flat_weight=1.0,
        suppress_cap=6.0,
        marg_weight=1.0,
        forget_pool_size=0,
        forget_chunk=0,
        exclude_forget_class_from_retain=False,
        # DDL (paper: gamma=30, lambda=10) + domain classifier
        domainloss_weight=DOMAINLOSS_WEIGHT,
        mmd_weight=MMD_WEIGHT,
        use_domain_cls_loss=True,
        is_domain_divided=True,
        domain_class_divided=False,
        lmd_domain_loss=1.0,
        # eval / misc
        eval_only=False,
        no_train=False,
        model_dir="",
        load_epoch=None,
        csv_file_path=osp.join(cli.output_dir, "results.csv"),
        # >>> the only real change from stock ADU <<<
        heldout_num=cli.heldout_num,
        heldout_seed=cli.heldout_seed,
        opts=[],
    )


@torch.no_grad()
def partitioned_eval(trainer, heldout_labels):
    """Per-(label, domain) top-1 over the full 126-class test set, then folded
    into the SEEN/HELD-OUT x RETAIN/FORGET 2x2 grid."""
    trainer.set_model_mode("eval")
    heldout = set(heldout_labels)

    forget_doms = set(trainer.del_domain_list)
    dom_names = trainer.domain_list  # index -> name

    # (is_heldout, is_forget) -> [correct, total]
    grid = defaultdict(lambda: [0, 0])
    # per forget-domain sanity: domain_name -> [correct, total] (all classes)
    per_domain = defaultdict(lambda: [0, 0])

    for batch in tqdm(trainer.test_loader, desc="partitioned eval"):
        input, label, domain = trainer.parse_batch_test(batch)
        out = trainer.model_inference(input)
        logits = out[0]
        pred = logits.argmax(dim=1)
        correct = pred.eq(label)

        for i in range(label.size(0)):
            lab = int(label[i].item())
            dom = int(domain[i].item())
            is_held = lab in heldout
            is_forget = dom_names[dom] in forget_doms
            cell = grid[(is_held, is_forget)]
            cell[1] += 1
            cell[0] += int(correct[i].item())
            pd = per_domain[dom_names[dom]]
            pd[1] += 1
            pd[0] += int(correct[i].item())

    def acc(is_held, is_forget):
        c, t = grid[(is_held, is_forget)]
        return (100.0 * c / t) if t else float("nan"), t

    res = {
        "seen_retain":     acc(False, False),
        "seen_forget":     acc(False, True),
        "heldout_retain":  acc(True, False),
        "heldout_forget":  acc(True, True),
        "per_domain": {k: (100.0 * v[0] / v[1] if v[1] else float("nan"), v[1])
                       for k, v in per_domain.items()},
    }
    return res


def verdict(res):
    sr = res["seen_retain"][0]
    sf = res["seen_forget"][0]
    hr = res["heldout_retain"][0]
    hf = res["heldout_forget"][0]

    supp_seen = 1.0 - sf / sr if sr > 0 else float("nan")
    supp_held = 1.0 - hf / hr if hr > 0 else float("nan")
    transfer = supp_held / supp_seen if supp_seen and supp_seen == supp_seen and supp_seen != 0 else float("nan")

    lines = []
    lines.append("")
    lines.append("=" * 68)
    lines.append("PHASE-0 RESULT: open-vocabulary domain forgetting")
    lines.append("=" * 68)
    lines.append(f"{'':>14} | {'RETAIN doms':>12} | {'FORGET dom':>11}")
    lines.append("-" * 46)
    lines.append(f"{'SEEN cls':>14} | {sr:>11.2f}% | {sf:>10.2f}%")
    lines.append(f"{'HELD-OUT cls':>14} | {hr:>11.2f}% | {hf:>10.2f}%")
    lines.append("-" * 46)
    lines.append(f"suppression (1 - forget/retain):  seen={supp_seen:.3f}  held-out={supp_held:.3f}")
    lines.append(f"TRANSFER (held-out supp / seen supp) = {transfer:.3f}")
    lines.append("")

    # control sanity
    if hr < 25.0:
        lines.append("!! WARNING: held-out RETAIN accuracy is very low (<25%). The prompt-")
        lines.append("!! tuned model may not generalize to unseen classes at all, which makes")
        lines.append("!! the transfer ratio unreliable. Inspect absolute numbers directly.")
        lines.append("")

    if transfer != transfer:  # NaN
        branch = "UNDETERMINED (degenerate retain accuracy)"
    elif transfer >= 0.70:
        branch = ("B  -> forgetting TRANSFERS to unseen vocabulary. ADU forgets the\n"
                  "        DOMAIN, not cells. Open-vocab randomization (M1) is weak here;\n"
                  "        keep M2 (redirection) for the trade-off story, drop M1.")
    elif transfer <= 0.30:
        branch = ("A  -> forgetting does NOT transfer. ADU forgets vocabulary CELLS,\n"
                  "        not the domain. This is the strong-paper world: every published\n"
                  "        For number measures cell-suppression. M1 (vocab randomization)\n"
                  "        becomes ESSENTIAL. Proceed M2 -> M1.")
    else:
        branch = ("C  -> PARTIAL transfer. Both mechanisms have headroom; M1 targets the\n"
                  "        residual leak on unseen vocabulary. Proceed M2 -> M1.")

    lines.append(f"VERDICT: Branch {branch}")
    lines.append("=" * 68)

    metrics = dict(
        seen_retain=sr, seen_forget=sf, heldout_retain=hr, heldout_forget=hf,
        suppression_seen=supp_seen, suppression_heldout=supp_held, transfer=transfer,
    )
    return "\n".join(lines), metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--forget", type=str, default="sketch")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--heldout_num", type=int, default=26)
    p.add_argument("--heldout_seed", type=int, default=1234)
    p.add_argument("--output-dir", type=str,
                   default=osp.join(os.path.expanduser("~"),
                                    "adu_results", "phase0_sketch"))
    cli = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
    os.makedirs(cli.output_dir, exist_ok=True)

    args = build_args(cli)
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"[phase0] forget={cli.forget} seed={cli.seed} "
          f"heldout_num={cli.heldout_num} heldout_seed={cli.heldout_seed}")
    print(f"[phase0] config: {TRAINER_CFG} (paper: bs=8, 50ep, gamma=30, lambda=10)")

    trainer = build_trainer(cfg)

    heldout_labels = list(getattr(trainer.dm.dataset, "heldout_labels", []))
    assert len(heldout_labels) == cli.heldout_num, (
        f"expected {cli.heldout_num} held-out classes, got {len(heldout_labels)} "
        "-- held-out split did not apply (check dataset name / config plumbing)."
    )
    # persist which classes were held out, for reproducibility
    id2name = {i: n for i, n in enumerate(trainer.classnames)}
    with open(osp.join(cli.output_dir, "heldout_classes.json"), "w") as f:
        json.dump({"labels": heldout_labels,
                   "names": [id2name[l] for l in heldout_labels]}, f, indent=2)

    # ---- train stock ADU (with held-out classes removed from train only) ----
    trainer.train_loop()

    # ---- partitioned eval + verdict ----
    res = partitioned_eval(trainer, heldout_labels)
    report, metrics = verdict(res)
    print(report)

    print("\nper-domain top-1 (all classes, sanity):")
    for dom, (a, n) in res["per_domain"].items():
        tag = "FORGET" if dom in set(trainer.del_domain_list) else "retain"
        print(f"  {dom:<10} {a:6.2f}%  (n={n})  [{tag}]")

    out = {
        "config": {
            "forget": cli.forget, "seed": cli.seed,
            "heldout_num": cli.heldout_num, "heldout_seed": cli.heldout_seed,
            "trainer_cfg": TRAINER_CFG, "domainloss_weight": DOMAINLOSS_WEIGHT,
            "mmd_weight": MMD_WEIGHT,
        },
        "grid": {
            "seen_retain": res["seen_retain"],
            "seen_forget": res["seen_forget"],
            "heldout_retain": res["heldout_retain"],
            "heldout_forget": res["heldout_forget"],
        },
        "per_domain": res["per_domain"],
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = osp.join(cli.output_dir, "phase0_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[phase0] wrote {out_path}")
    with open(osp.join(cli.output_dir, "phase0_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
