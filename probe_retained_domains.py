"""
Extend the retained-representation drift measurement to the remaining retained
domains (painting, real) for the sketch-forget checkpoint.

For each retained domain d:
  probe(zero-shot CLIP d features)  <- CEILING
  probe(ADU d features)             <- retained
  gap = CEILING - ADU               <- representation drift

Same probe / same images / same split as clipart. Only the extractor differs.
Read-only. No training of the ADU model.
"""
import argparse
import os.path as osp
import numpy as np
import torch

from dassl.engine import build_trainer
from dassl.data.transforms import build_transform

import datasets.domainnet_mini_paper_df  # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401
from train_loop import setup_cfg
from probe_experiment import (
    read_split, build_probe_pool, extract_features, load_zeroshot_clip,
    fit_probe, DATA_ROOT, CKPT_DIR, OUT_DIR, DOMAINS, SHOTS_PER_CLASS, EPOCH,
)

RETAINED = ["painting", "real"]   # clipart already measured (84.13 / 72.70)


def main():
    device = torch.device("cuda")

    args = argparse.Namespace(
        root=DATA_ROOT, output_dir=osp.join(OUT_DIR, "tmp"), resume="",
        seed=1, source_domains=None, target_domains=None, transforms=None,
        trainer="IVLP_VL_Adapter_Prompt", backbone="", head="",
        eval_only=True, model_dir=CKPT_DIR, load_epoch=EPOCH, no_train=True,
        num_shots=8, forget_domains=["sketch"], forget_classes=[],
        domain_class_divided=False, lmd_domain_loss=1.0,
        use_domain_cls_loss=True, is_domain_divided=True,
        csv_file_path=osp.join(OUT_DIR, "dummy.csv"),
        dataset_name="domainnet_mini_paper_df",
        domainloss_weight=30.0, mmd_weight=10.0,
        dataset_config_file="configs/datasets/domainnet_mini_paper_df.yaml",
        config_file="configs/trainers/vit_b16_ep50.yaml",
        opts=[],
    )
    cfg = setup_cfg(args)
    print("== building trainer & loading ADU checkpoint ==", flush=True)
    trainer = build_trainer(cfg)
    trainer.load_model(CKPT_DIR, epoch=EPOCH)
    model = trainer.model.eval()
    dtype = model.dtype
    tfm = build_transform(cfg, is_train=False)

    used = {d: set() for d in DOMAINS}
    for item in trainer.dm.dataset.train_x:
        used[DOMAINS[item.domain]].add(item.impath)

    print("== loading zero-shot CLIP ==", flush=True)
    zs = load_zeroshot_clip(device)

    results = {"clipart": (84.13, 72.70)}  # from prior run, for the final table
    for d in RETAINED:
        pool = build_probe_pool(d, used[d], SHOTS_PER_CLASS)
        test = read_split(d, "test")
        print(f"\n== {d}: pool={len(pool)} test={len(test)} ==", flush=True)

        adu_tr_X, adu_tr_y = extract_features(
            model.image_encoder, dtype, pool, tfm, device, f"ADU {d}-train")
        adu_te_X, adu_te_y = extract_features(
            model.image_encoder, dtype, test, tfm, device, f"ADU {d}-test")
        zs_tr_X, zs_tr_y = extract_features(
            zs.visual, zs.dtype, pool, tfm, device, f"ZS {d}-train")
        zs_te_X, zs_te_y = extract_features(
            zs.visual, zs.dtype, test, tfm, device, f"ZS {d}-test")

        ceil = fit_probe(zs_tr_X, zs_tr_y, zs_te_X, zs_te_y,
                         f"Zero-shot CLIP features, {d} (CEILING)")
        adu = fit_probe(adu_tr_X, adu_tr_y, adu_te_X, adu_te_y,
                        f"ADU features, {d} (retained)")
        results[d] = (ceil * 100, adu * 100)

    print("\n============ RETAINED-DOMAIN DRIFT (forget = sketch) ============", flush=True)
    print(f"{'domain':<10}{'ceiling':>10}{'ADU':>10}{'drift':>10}")
    for d in ["clipart"] + RETAINED:
        c, a = results[d]
        print(f"{d:<10}{c:>9.2f}%{a:>9.2f}%{a-c:>+9.1f}")
    drifts = [results[d][1] - results[d][0] for d in ["clipart"] + RETAINED]
    print(f"\nmean retained drift: {np.mean(drifts):+.1f} points", flush=True)
    if np.mean(drifts) <= -8:
        print(">>> consistent sizable drift across retained domains — channel is REAL")
    elif np.mean(drifts) <= -3:
        print(">>> modest/mixed drift — discuss")
    else:
        print(">>> negligible on average — finding does not generalize")


if __name__ == "__main__":
    main()
