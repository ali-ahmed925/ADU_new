"""
Pre-register the evaluation concept set.

The claim under test concerns how a model behaves once a concept is erased, and
that behaviour plausibly depends on whether the erased concept has a close
semantic neighbour to absorb the freed probability mass. So the concept set is
stratified on exactly that variable, measured objectively rather than chosen by
hand:

    proximity(c) = max_{c' != c} cos( t_c , t_c' )      in zero-shot CLIP text space

Classes are split into proximity terciles (close-neighbour / mid / isolated) and
sampled uniformly at random within each tercile under a fixed seed. Classes with
too few images in the forget domain to support the forget pool are excluded
first, and that exclusion is reported.

Run this ONCE, record the printed set in the paper, and do not re-roll the seed.
"""
import argparse
import os.path as osp
from collections import defaultdict

import numpy as np
import torch

DOMAINS = ["clipart", "painting", "real", "sketch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--forget-domain", default="sketch")
    ap.add_argument("--per-tier", type=int, default=3, help="concepts sampled per tercile")
    ap.add_argument("--min-images", type=int, default=120,
                    help="minimum forget-domain train images (must cover the forget pool)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-concept", default="tiger",
                    help="development concept: excluded from sampling, reported separately")
    ap.add_argument("--backbone", default="ViT-B/16")
    args = ap.parse_args()

    # ---- class names in label order, and per-class image counts ----
    img_root = osp.join(args.root, "DomainNet")
    lab2name, counts = {}, defaultdict(int)
    for d in DOMAINS:
        with open(osp.join(img_root, "splits_mini", f"{d}_train.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, lab = line.split(" ")
                lab2name[int(lab)] = rel.split("/")[-2]
                if d == args.forget_domain:
                    counts[rel.split("/")[-2]] += 1
    C = max(lab2name) + 1
    names = [lab2name[i] for i in range(C)]
    print(f"== {C} classes; counting {args.forget_domain} train images ==")

    # ---- zero-shot CLIP text embeddings ----
    from clip import clip as _clip
    model_path = _clip._download(_clip._MODELS[args.backbone])
    try:
        m = torch.jit.load(model_path, map_location="cpu").eval(); sd = None
    except RuntimeError:
        sd = torch.load(model_path, map_location="cpu"); m = None
    design = {"trainer": "IVLP_VL_Adapter_Prompt", "vision_depth": 0, "language_depth": 0,
              "vision_ctx": 0, "language_ctx": 0, "use_classtoken": False,
              "use_cross_attention": False, "independent_cross_attention": False,
              "independent_learnable_vision": False, "insert_layer": 9}
    base = _clip.build_model(sd if sd is not None else m.state_dict(), design).eval()
    tok = _clip.tokenize([f"a photo of a {n.replace('_', ' ')}." for n in names])
    with torch.no_grad():
        t = base.encode_text(tok)
        t = (t / t.norm(dim=-1, keepdim=True)).float().numpy()

    S = t @ t.T
    np.fill_diagonal(S, -np.inf)
    prox = S.max(1)                     # similarity to the single nearest other class
    nearest = [names[i] for i in S.argmax(1)]

    # ---- eligibility ----
    eligible = [i for i in range(C)
                if counts[names[i]] >= args.min_images and names[i] != args.dev_concept]
    dropped = C - len(eligible) - 1
    print(f"== {len(eligible)} eligible ({dropped} dropped for <{args.min_images} "
          f"{args.forget_domain} images; '{args.dev_concept}' held out as development concept) ==")

    # ---- terciles on proximity, then random sample within each ----
    order = sorted(eligible, key=lambda i: prox[i])
    k = len(order) // 3
    tiers = {"isolated": order[:k], "mid": order[k:2 * k], "close-neighbour": order[2 * k:]}

    rng = np.random.RandomState(args.seed)
    chosen = {}
    for tname, idxs in tiers.items():
        pick = rng.choice(idxs, size=min(args.per_tier, len(idxs)), replace=False)
        chosen[tname] = sorted(pick.tolist(), key=lambda i: -prox[i])

    print("\n" + "=" * 78)
    print(f"EVALUATION CONCEPT SET  (seed={args.seed}, {args.per_tier} per tercile)")
    print("=" * 78)
    for tname in ["close-neighbour", "mid", "isolated"]:
        print(f"\n[{tname}]")
        for i in chosen[tname]:
            print(f"  {names[i]:<26} proximity={prox[i]:.3f}  nearest={nearest[i]:<20} "
                  f"n_{args.forget_domain}={counts[names[i]]}")

    di = names.index(args.dev_concept) if args.dev_concept in names else None
    if di is not None:
        print(f"\n[development concept, reported separately]")
        print(f"  {args.dev_concept:<26} proximity={prox[di]:.3f}  nearest={nearest[di]:<20} "
              f"n_{args.forget_domain}={counts[args.dev_concept]}")

    flat = [names[i] for tn in ["close-neighbour", "mid", "isolated"] for i in chosen[tn]]
    print("\n" + "-" * 78)
    print("CONCEPTS=\"" + " ".join(flat) + "\"")
    print("-" * 78)
    print("\nPaper sentence:")
    print(f'  "Concepts were sampled uniformly at random (seed {args.seed}) within terciles of')
    print(f'   nearest-neighbour text-embedding similarity, from the {len(eligible)} classes with at')
    print(f'   least {args.min_images} {args.forget_domain} training images. \'{args.dev_concept}\' was used for method')
    print('   development and is reported separately."\n')


if __name__ == "__main__":
    main()
