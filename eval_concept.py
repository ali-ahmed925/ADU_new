"""
Fine-grained concept-forgetting evaluation — FULL per-class × per-domain table.

The built-in test set has only 5 images per (class, domain) — too coarse to read
a single class. Here we score the model's OWN predictions on EVERY unseen
train-split image (hundreds per class per domain), for ALL classes, giving a
trustworthy per-class × per-domain accuracy table.

Why all classes: to prove the forget target (e.g. tiger) drops while EVERY other
class — related (lion) and unrelated — is preserved, you need a reliable number
for each class, not just tiger/lion.

Read-only. Uses the model's text head (argmax of image·text), i.e. the actual
quantity forgetting acts on — not a linear probe.
"""
import argparse
import os
from collections import defaultdict
import numpy as np
import torch

from dassl.engine import build_trainer
from dassl.data.transforms import build_transform

import datasets.domainnet_mini_paper_df  # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401
from train_loop import setup_cfg
from probe_experiment import read_split, extract_features, DATA_ROOT, DOMAINS


def cls_of(path):
    return path.split("/")[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir with VLPromptLearner/model.pth.tar-*")
    ap.add_argument("--load-epoch", type=int, required=True)
    ap.add_argument("--forget-domain", default="sketch")
    ap.add_argument("--forget-class", default="tiger")
    ap.add_argument("--neighbor", default="lion")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-test-split", action="store_true",
                    help="also fold in the official test-split images (the extra ~5/cell)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per (class,domain) cell; 0 = ALL unseen (default). "
                         "Use a small value (e.g. 20) for a fast pipeline smoke test.")
    args = ap.parse_args()

    os.makedirs("/tmp/eval_concept_tmp", exist_ok=True)
    device = torch.device("cuda")
    ns = argparse.Namespace(
        root=DATA_ROOT, output_dir="/tmp/eval_concept_tmp", resume="",
        seed=args.seed, source_domains=None, target_domains=None, transforms=None,
        trainer="IVLP_VL_Adapter_Prompt", backbone="", head="",
        eval_only=True, model_dir=args.ckpt_dir, load_epoch=args.load_epoch, no_train=True,
        num_shots=8, forget_domains=[args.forget_domain], forget_classes=[args.forget_class],
        domain_class_divided=False, lmd_domain_loss=1.0,
        use_domain_cls_loss=False, is_domain_divided=True,
        forget_loss_type="entropy", no_retain_loss=False,
        csv_file_path="/tmp/eval_concept_tmp/dummy.csv",
        dataset_name="domainnet_mini_paper_df",
        domainloss_weight=0.0, mmd_weight=0.0,
        dataset_config_file="configs/datasets/domainnet_mini_paper_df.yaml",
        config_file="configs/trainers/vit_b16_ep50.yaml",
        opts=[],
    )
    cfg = setup_cfg(ns)
    print("== building trainer & loading checkpoint ==", flush=True)
    trainer = build_trainer(cfg)
    trainer.load_model(args.ckpt_dir, epoch=args.load_epoch)
    model = trainer.model.eval()
    dtype = model.dtype
    tfm = build_transform(cfg, is_train=False)

    with torch.no_grad():
        prompts = model.prompt_learner()
        txt = model.text_encoder(prompts, model.tokenized_prompts)
        txt = (txt / txt.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    # 8-shot images the model actually trained on (to exclude)
    used = set(it.impath for it in trainer.dm.dataset.train_x)

    # ---- build the full unseen eval set: every train-split image minus the 8-shot ----
    splits = ["train", "test"] if args.include_test_split else ["train"]
    cell = defaultdict(list)                      # (classname, domain) -> [(path,label)]
    for d in DOMAINS:
        for sp in splits:
            for p, l in read_split(d, sp):
                if p not in used:
                    cell[(cls_of(p), d)].append((p, l))
    if args.limit > 0:
        cell = {k: v[:args.limit] for k, v in cell.items()}
        print(f"[SMOKE] capping to {args.limit} imgs/cell", flush=True)

    # flatten to one list, extract features in a single pass, predict
    flat, index = [], []
    for (c, d), items in cell.items():
        for p, l in items:
            flat.append((p, l)); index.append((c, d))
    print(f"== scoring {len(flat)} unseen images across {len(cell)} (class,domain) cells ==", flush=True)
    feats, labels = extract_features(model.image_encoder, dtype, flat, tfm, device, "all")
    preds = np.argmax(feats @ txt.T, axis=1)
    correct = (preds == labels).astype(float)

    # aggregate per (class, domain)
    acc = defaultdict(lambda: {})   # class -> {domain: (acc, n)}
    agg = defaultdict(lambda: [0.0, 0])
    for ok, (c, d) in zip(correct, index):
        agg[(c, d)][0] += ok; agg[(c, d)][1] += 1
    for (c, d), (s, n) in agg.items():
        acc[c][d] = (s / n if n else float("nan"), n)

    classnames = sorted(acc.keys())
    fc, nb, fd = args.forget_class, args.neighbor, args.forget_domain

    # ---- full per-class per-domain table ----
    print("\n================ PER-CLASS × PER-DOMAIN ACCURACY (unseen pool) ================", flush=True)
    print(f"{'class':<24}" + "".join(f"{d:<10}" for d in DOMAINS))
    for c in classnames:
        line = f"{c:<24}"
        for d in DOMAINS:
            a, n = acc[c].get(d, (float('nan'), 0))
            line += f"{a*100:5.1f}({n:<3}) " if n else f"{'--':<9} "
        mark = "  <== FORGET" if c == fc else ("  <== neighbor" if c == nb else "")
        print(line + mark, flush=True)

    # ---- summary ----
    def cell_acc(c, d):
        return acc[c].get(d, (float('nan'), 0))[0] * 100

    fsk = cell_acc(fc, fd)
    f_other = np.nanmean([cell_acc(fc, d) for d in DOMAINS if d != fd])
    # retain = every class except the forget class, averaged
    retain_cells = [cell_acc(c, d) for c in classnames if c != fc for d in DOMAINS]
    retain_mean = np.nanmean(retain_cells)
    nb_mean = np.nanmean([cell_acc(nb, d) for d in DOMAINS])

    print("\n================ SUMMARY ================", flush=True)
    print(f"{fc} in forget domain ({fd}):     {fsk:5.1f}%   <- want LOW")
    print(f"{fc} in other domains (mean):      {f_other:5.1f}%   <- LOW=propagated, HIGH=contained")
    print(f"{nb} (neighbor) mean, all domains: {nb_mean:5.1f}%   <- want HIGH (preserved)")
    print(f"retain (ALL other classes) mean:  {retain_mean:5.1f}%   <- want HIGH (preserved)")


if __name__ == "__main__":
    main()
