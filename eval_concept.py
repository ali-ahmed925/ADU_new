"""
Fine-grained concept-forgetting evaluation — FULL per-class × per-domain table.

The built-in test set has only 5 images per (class, domain) — too coarse to read
a single class. Here we score the model's OWN predictions on EVERY unseen
train-split image (hundreds per class per domain), for ALL classes, giving a
trustworthy per-class × per-domain accuracy table.

Self-contained: takes --root (the DomainNet parent dir), so it runs on any machine.

Read-only. Uses the model's text head (argmax of image·text), i.e. the actual
quantity forgetting acts on — not a linear probe.
"""
import argparse
import os
import os.path as osp
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dassl.engine import build_trainer
from dassl.data.transforms import build_transform

import datasets.domainnet_mini_paper_df  # noqa: F401  (registers dataset)
import trainers.independent_VLAdapter_Prompt  # noqa: F401  (registers trainer)
from train_loop import setup_cfg

DOMAINS = ["clipart", "painting", "real", "sketch"]


def read_split(root, domain, split):
    """Return [(full_path, label)] from <root>/DomainNet/splits_mini/<domain>_<split>.txt."""
    img_root = osp.join(root, "DomainNet")
    split_file = osp.join(img_root, "splits_mini", f"{domain}_{split}.txt")
    items = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, label = line.split(" ")
            items.append((osp.join(img_root, rel), int(label)))
    return items


class ImageListDataset(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tfm(Image.open(path).convert("RGB")), label


@torch.no_grad()
def extract_features(image_encoder, dtype, items, tfm, device, desc="", batch=64):
    dl = DataLoader(ImageListDataset(items, tfm), batch_size=batch,
                    shuffle=False, num_workers=4, pin_memory=True)
    feats, labels = [], []
    for bi, (imgs, lbs) in enumerate(dl):
        f = image_encoder(imgs.to(device).type(dtype))[0]
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
        labels.append(lbs.numpy())
        if (bi + 1) % 100 == 0:
            print(f"  {desc}: batch {bi+1}/{len(dl)}", flush=True)
    return np.concatenate(feats), np.concatenate(labels)


def cls_of(path):
    return path.split("/")[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="parent dir of DomainNet/ (the --root you train with)")
    ap.add_argument("--ckpt-dir", required=True, help="dir with VLPromptLearner/model.pth.tar-*")
    ap.add_argument("--load-epoch", type=int, required=True)
    ap.add_argument("--forget-domain", default="sketch")
    ap.add_argument("--forget-class", default="tiger")
    ap.add_argument("--neighbor", default="lion")
    ap.add_argument("--seed", type=int, default=1,
                    help="MUST match the seed the checkpoint was trained with (exclusion set depends on it)")
    ap.add_argument("--num-shots", type=int, default=8,
                    help="MUST match --num_shots used in training. The eval reconstructs the exact "
                         "same few-shot set (via seed+num_shots) and excludes precisely those images.")
    ap.add_argument("--include-test-split", action="store_true",
                    help="also fold in the official test-split images (extra ~5/cell)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap imgs per (class,domain) cell; 0 = ALL unseen (default). small value = smoke test")
    args = ap.parse_args()

    os.makedirs("/tmp/eval_concept_tmp", exist_ok=True)
    device = torch.device("cuda")
    ns = argparse.Namespace(
        root=args.root, output_dir="/tmp/eval_concept_tmp", resume="",
        seed=args.seed, source_domains=None, target_domains=None, transforms=None,
        trainer="IVLP_VL_Adapter_Prompt", backbone="", head="",
        eval_only=True, model_dir=args.ckpt_dir, load_epoch=args.load_epoch, no_train=True,
        num_shots=args.num_shots, forget_domains=[args.forget_domain], forget_classes=[args.forget_class],
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

    # ---- full unseen eval set: every train-split image minus the 8-shot ----
    splits = ["train", "test"] if args.include_test_split else ["train"]
    cell = defaultdict(list)
    for d in DOMAINS:
        for sp in splits:
            for p, l in read_split(args.root, d, sp):
                if p not in used:
                    cell[(cls_of(p), d)].append((p, l))
    if args.limit > 0:
        cell = {k: v[:args.limit] for k, v in cell.items()}
        print(f"[SMOKE] capping to {args.limit} imgs/cell", flush=True)

    flat, index = [], []
    for (c, d), items in cell.items():
        for p, l in items:
            flat.append((p, l)); index.append((c, d))
    print(f"== scoring {len(flat)} unseen images across {len(cell)} (class,domain) cells ==", flush=True)
    feats, labels = extract_features(model.image_encoder, dtype, flat, tfm, device, "all")
    preds = np.argmax(feats @ txt.T, axis=1)
    correct = (preds == labels).astype(float)

    agg = defaultdict(lambda: [0.0, 0])
    for ok, (c, d) in zip(correct, index):
        agg[(c, d)][0] += ok; agg[(c, d)][1] += 1
    acc = defaultdict(dict)
    for (c, d), (s, n) in agg.items():
        acc[c][d] = (s / n if n else float("nan"), n)

    classnames = sorted(acc.keys())
    fc, nb, fd = args.forget_class, args.neighbor, args.forget_domain

    print("\n============ PER-CLASS × PER-DOMAIN ACCURACY (unseen pool) ============", flush=True)
    print(f"{'class':<24}" + "".join(f"{d:<12}" for d in DOMAINS))
    for c in classnames:
        line = f"{c:<24}"
        for d in DOMAINS:
            a, n = acc[c].get(d, (float('nan'), 0))
            line += f"{a*100:5.1f}(n={n:<4})" if n else f"{'--':<11} "
        mark = "  <== FORGET" if c == fc else ("  <== neighbor" if c == nb else "")
        print(line + mark, flush=True)

    # ---- HOW the forget class fails: calibration + leakage of forgotten predictions ----
    fc_rows = np.array([c == fc for (c, d) in index])
    if fc_rows.any():
        lab2cname = trainer.lab2cname  # {label -> classname}
        logit_scale = model.logit_scale.exp().item()
        L = logit_scale * (feats[fc_rows] @ txt.T)
        L = L - L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        conf = P.max(1).mean() * 100
        norm_ent = (-(P * np.log(P + 1e-12)).sum(1)).mean() / np.log(P.shape[1])
        fp = preds[fc_rows]
        n = int(fc_rows.sum())
        from collections import Counter
        top = Counter(int(x) for x in fp).most_common(6)
        print(f"\n======== HOW '{fc}' IS FORGOTTEN (all domains, n={n}) ========", flush=True)
        print(f"mean confidence of the prediction: {conf:5.1f}%   (HIGH = confidently-wrong, LOW = uncertain)")
        print(f"normalized prediction entropy:     {norm_ent:5.3f}   (1.0 = uniform/even spread, 0 = concentrated)")
        print(f"where forgotten-{fc} images are sent (top classes):")
        for cid, cnt in top:
            name = lab2cname.get(cid, str(cid))
            print(f"    {name:<20} {100*cnt/n:5.1f}%")

    # ---- FEATURE-LEAK / FUNNELING diagnostic ----
    # Var_c(<f,t_c>) = f^T Sigma_t f. Sigma_t (cov of C text embeddings in d dims)
    # is rank <= C-1, so its NULL SPACE (dim ~ d-C+1) is exactly the set of f giving
    # uniform logits -> the "flat" region is a large subspace, not a single line,
    # so funnel-to-a-line is not forced. The real risk (caveat 2) is that limited
    # prompt capacity clusters the forgotten features WITHIN that subspace. Measure
    # that directly: effective dimensionality (participation ratio) + pairwise
    # cohesion of forgotten features vs a normal retain class.
    if fc_rows.any():
        Tc_ = txt - txt.mean(0, keepdims=True)
        evals = np.linalg.eigvalsh(Tc_.T @ Tc_ / txt.shape[0])
        lam_max = float(evals[-1])
        flat_dim = int((evals < 1e-4 * lam_max).sum())     # ~ null space of Sigma_t

        def spread(F):
            n = len(F)
            if n < 2:
                return float("nan"), float("nan")
            Fc = F - F.mean(0, keepdims=True)
            ev = np.clip(np.linalg.eigvalsh(Fc.T @ Fc / n), 0, None)
            pr = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12))   # effective dims spanned
            G = F @ F.T
            pc = float((G.sum() - np.trace(G)) / (n * (n - 1)))     # mean pairwise cosine
            return pr, pc

        nb_rows = np.array([c == nb for (c, d) in index])
        fpr, fpc = spread(feats[fc_rows])
        rpr, rpc = spread(feats[nb_rows]) if nb_rows.any() else (float("nan"), float("nan"))

        print(f"\n======== FEATURE-LEAK / FUNNELING (forget='{fc}' vs reference='{nb}') ========", flush=True)
        print(f"uniform-logit (flat) subspace dim = {flat_dim} of {txt.shape[1]}  "
              f"(large => spread is geometrically possible; funnel-to-a-line not forced)")
        print(f"{'':<30}{'forget':>9}{'reference':>11}")
        print(f"{'effective dims (part. ratio)':<30}{fpr:>9.1f}{rpr:>11.1f}   (forget << ref => funneled to fewer dims)")
        print(f"{'mean pairwise cosine':<30}{fpc:>9.3f}{rpc:>11.3f}   (forget >> ref => features clustered)")

    def ca(c, d):
        return acc[c].get(d, (float('nan'), 0))[0] * 100

    fsk = ca(fc, fd)
    f_other = np.nanmean([ca(fc, d) for d in DOMAINS if d != fd])
    retain_mean = np.nanmean([ca(c, d) for c in classnames if c != fc for d in DOMAINS])
    nb_mean = np.nanmean([ca(nb, d) for d in DOMAINS])

    print("\n================ SUMMARY ================", flush=True)
    print(f"{fc} in forget domain ({fd}):     {fsk:5.1f}%   <- want LOW")
    print(f"{fc} in other domains (mean):      {f_other:5.1f}%   <- LOW=propagated, HIGH=contained")
    print(f"{nb} (neighbor) mean, all domains: {nb_mean:5.1f}%   <- want HIGH (preserved)")
    print(f"retain (ALL other classes) mean:  {retain_mean:5.1f}%   <- want HIGH (preserved)")


if __name__ == "__main__":
    main()
