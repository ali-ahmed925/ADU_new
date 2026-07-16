"""
Fill the empty slot: measure the zero-shot CLIP linear-probe accuracy on the
RETAINED domain (clipart) features, using the exact same probe / images / split
that produced the ADU clipart number (72.7%).

This is the mechanism-matched comparison:
  probe(zero-shot CLIP clipart features)  <- CEILING (this script measures it)
  probe(ADU clipart features)             <- 72.7% (reused from cache, identical images)
Only the feature extractor differs, so the gap isolates representation change.

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


def main():
    device = torch.device("cuda")

    # Rebuild the SAME cfg/trainer as the sketch_seed1 run — needed to recover the
    # exact 8-shot 'used' set so the clipart pool is byte-identical to before.
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

    # Identical clipart image lists (same seed=0, same exclusion) as the 72.7% run
    clipart_pool = build_probe_pool("clipart", used["clipart"], SHOTS_PER_CLASS)
    clipart_test = read_split("clipart", "test")
    print(f"clipart pool={len(clipart_pool)}  test={len(clipart_test)}", flush=True)

    # --- ADU clipart side: reuse cached features iff they match these exact lists ---
    cache = osp.join(OUT_DIR, "features.npz")
    z = np.load(cache)
    if (len(z["adu_cl_tr_y"]) == len(clipart_pool)
            and len(z["adu_cl_te_y"]) == len(clipart_test)):
        print("== reusing cached ADU clipart features (identical images) ==", flush=True)
        adu_tr_X, adu_tr_y = z["adu_cl_tr_X"], z["adu_cl_tr_y"]
        adu_te_X, adu_te_y = z["adu_cl_te_X"], z["adu_cl_te_y"]
    else:
        print("== cache mismatch -> re-extracting ADU clipart features ==", flush=True)
        adu_tr_X, adu_tr_y = extract_features(
            model.image_encoder, dtype, clipart_pool, tfm, device, "ADU clipart-train")
        adu_te_X, adu_te_y = extract_features(
            model.image_encoder, dtype, clipart_test, tfm, device, "ADU clipart-test")

    # --- Zero-shot CLIP clipart side: NEW measurement on the identical lists ---
    print("== extracting zero-shot CLIP clipart features ==", flush=True)
    zs = load_zeroshot_clip(device)
    zs_tr_X, zs_tr_y = extract_features(
        zs.visual, zs.dtype, clipart_pool, tfm, device, "ZS clipart-train")
    zs_te_X, zs_te_y = extract_features(
        zs.visual, zs.dtype, clipart_test, tfm, device, "ZS clipart-test")

    print("\n================ RETAINED-DOMAIN (clipart) PROBE ================", flush=True)
    ceil = fit_probe(zs_tr_X, zs_tr_y, zs_te_X, zs_te_y,
                     "Zero-shot CLIP features, clipart (CEILING)")
    adu = fit_probe(adu_tr_X, adu_tr_y, adu_te_X, adu_te_y,
                    "ADU features, clipart (retained)")

    gap = (ceil - adu) * 100
    print("\n================ VERDICT ================", flush=True)
    print(f"CEILING  (zero-shot clipart probe): {ceil*100:.2f}%")
    print(f"ADU      (retained clipart probe):  {adu*100:.2f}%")
    print(f"retained-representation drift (ceiling - ADU): {gap:+.1f} points")
    if gap >= 8:
        print(">>> sizable retained drift — 'silent damage' channel is REAL")
    elif gap >= 3:
        print(">>> modest drift — borderline, discuss")
    else:
        print(">>> negligible drift — retained representation is preserved (finding is DEAD)")


if __name__ == "__main__":
    main()
