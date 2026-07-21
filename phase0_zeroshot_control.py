"""Zero-shot CLIP reference for the Phase-0 restricted eval.

WHY
---
The data audit proves the plumbing is clean, but it cannot tell us whether
30.77% on held-out sketches is real forgetting or an artifact. This script runs
UNTOUCHED CLIP (no prompt tuning, no unlearning, no checkpoint) through the
EXACT same restricted 26-way eval, giving the reference row.

It answers two things at once:

 (1) Are the held-out classes intrinsically harder?
     Zero-shot CLIP never trained on ANY of these classes, so it should score
     roughly the SAME on seen and held-out classes. If it does, then the
     92.46% vs 64.87% gap in the ADU model is caused by our training, not by
     the 26-class draw being unlucky.

 (2) Are sketch images recognizable at all?
     If zero-shot CLIP scores high on sketch, then ADU's low sketch numbers are
     genuine forgetting rather than broken anchors / unreadable images.

USAGE
    python phase0_zeroshot_control.py --heldout_num 26 --heldout_seed 1234
"""
import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

import torch
from tqdm import tqdm

import datasets.domainnet_mini_paper_df  # noqa: F401
from train_loop import setup_cfg
from engine.dataset_manager import DataManager
from phase0_restricted_eval import restricted_acc
from phase0_diagnostic import build_args, DATA_ROOT
from trainers.independent_VLAdapter_Prompt import load_clip_to_cpu

from clip import clip


def load_vanilla_clip(cfg):
    """Plain CLIP, built through the repo's OWN loader so every design_details
    key stays in sync with the codebase. We only clone the cfg and zero the
    prompt depths: clip/model.py sets VPT_shallow=False when vision_depth==0,
    and use_cross_attention=False disables InstaPG. Result is stock CLIP."""
    cfg2 = cfg.clone()
    cfg2.defrost()
    cfg2.TRAINER.IVLP.PROMPT_DEPTH_VISION = 0
    cfg2.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 0
    cfg2.USE_CROSSATTENTION = False
    cfg2.INDEPENDENT_CROSS_ATTENTION = False
    cfg2.freeze()
    return load_clip_to_cpu(cfg2)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--forget", type=str, default="sketch")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--heldout_num", type=int, default=26)
    p.add_argument("--heldout_seed", type=int, default=1234)
    p.add_argument("--root", type=str, default=DATA_ROOT)
    p.add_argument("--n_draws", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="/tmp/phase0_zeroshot")
    cli = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
    os.makedirs(cli.output_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = setup_cfg(build_args(cli))
    dm = DataManager(cfg)
    ds = dm.dataset
    heldout = sorted(ds.heldout_labels)
    classnames = [c.replace("_", " ") for c in ds.classnames]

    model = load_vanilla_clip(cfg).to(dev).eval()

    # text anchors, same prompt template as the paper: "a photo of a [class]."
    prompts = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in classnames]).to(dev)
    txt = model.encode_text(prompts).float()
    txt = txt / txt.norm(dim=-1, keepdim=True)
    txt = txt.cpu()

    feats, labels, domains = [], [], []
    for batch in tqdm(dm.test_loader, desc="zero-shot CLIP features"):
        img = batch["img"].to(dev)
        f = model.encode_image(img)
        if isinstance(f, (tuple, list)):
            f = f[0]
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        labels.append(batch["label"])
        domains.append(batch["domain"])
    feats = torch.cat(feats); labels = torch.cat(labels); domains = torch.cat(domains)

    dom_names = ["clipart", "painting", "real", "sketch"]
    fid = [dom_names.index(cli.forget)]
    fmask = torch.isin(domains, torch.tensor(fid))
    rmask = ~fmask

    seen = sorted(set(range(len(classnames))) - set(heldout))
    k = len(heldout)

    ho_r, n_hr = restricted_acc(feats, labels, domains, txt, heldout, rmask)
    ho_f, n_hf = restricted_acc(feats, labels, domains, txt, heldout, fmask)

    g = torch.Generator().manual_seed(0)
    sr_l, sf_l = [], []
    for _ in range(cli.n_draws):
        perm = torch.randperm(len(seen), generator=g)[:k]
        sub = [seen[i] for i in perm.tolist()]
        a, _ = restricted_acc(feats, labels, domains, txt, sub, rmask)
        b, _ = restricted_acc(feats, labels, domains, txt, sub, fmask)
        sr_l.append(a); sf_l.append(b)
    sr = sum(sr_l) / len(sr_l); sf = sum(sf_l) / len(sf_l)

    chance = 100.0 / k
    print("\n" + "=" * 70)
    print(f"ZERO-SHOT CLIP reference (no tuning, no unlearning) -- {k}-way, chance {chance:.2f}%")
    print("=" * 70)
    print(f"{'':>16} | {'RETAIN doms':>12} | {'FORGET dom':>11}")
    print("-" * 46)
    print(f"{'SEEN cls':>16} | {sr:>11.2f}% | {sf:>10.2f}%")
    print(f"{'HELD-OUT cls':>16} | {ho_r:>11.2f}% | {ho_f:>10.2f}%")
    print("-" * 46)
    print(f"seen-vs-heldout gap (retain): {sr - ho_r:+.2f} pts")
    print("  -> near 0 means the held-out class draw is NOT intrinsically harder,")
    print("     so any gap in the ADU model is caused by training.")
    print(f"zero-shot sketch accuracy: seen {sf:.2f}% / held-out {ho_f:.2f}%")
    print("  -> high means sketches ARE recognizable, so ADU's low sketch")
    print("     numbers are genuine forgetting, not broken anchors.")
    print("=" * 70)

    out = dict(seen_retain=sr, seen_forget=sf, heldout_retain=ho_r, heldout_forget=ho_f,
               chance=chance, k=k, n_heldout_retain=n_hr, n_heldout_forget=n_hf)
    with open(osp.join(cli.output_dir, "phase0_zeroshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[zeroshot] wrote {osp.join(cli.output_dir, 'phase0_zeroshot.json')}")


if __name__ == "__main__":
    main()
