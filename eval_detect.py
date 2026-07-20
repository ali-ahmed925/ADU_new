"""
Deployment-facing evaluation of an unlearned checkpoint.

Two measurements the standard forget/retain accuracy pair cannot provide:

  1. DETECTABILITY (AUROC). If the unlearned model is genuinely uncertain on the
     erased concept, a confidence threshold should separate erased-concept inputs
     from ordinary retained inputs. This turns "our predictions are calibrated"
     from a preference into an operational capability: forgotten inputs can be
     caught by routine confidence monitoring instead of silently entering
     downstream decisions. A confidently-wrong unlearner should score near 0.5.

  2. HELD-OUT RECOVERY. Subtracting a single fixed bias vector from the logits
     can restore the erased class. Estimating that vector on the SAME images it
     is applied to is oracle-contaminated, so here we estimate it on one half of
     the erased-concept images and apply it to the held-out half. A control
     estimates the vector from retained-class images instead: if only the
     concept-estimated vector recovers the class, the erasure is a
     concept-specific, removable offset rather than a deletion.

Read-only. Runs on any checkpoint (ours, NegGrad, retain-only).
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
    img_root = osp.join(root, "DomainNet")
    with open(osp.join(img_root, "splits_mini", f"{domain}_{split}.txt")) as f:
        out = []
        for line in f:
            line = line.strip()
            if line:
                rel, lab = line.split(" ")
                out.append((osp.join(img_root, rel), int(lab)))
    return out


class ImageListDataset(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, l = self.items[i]
        return self.tfm(Image.open(p).convert("RGB")), l


@torch.no_grad()
def extract(image_encoder, dtype, items, tfm, device, batch=64):
    dl = DataLoader(ImageListDataset(items, tfm), batch_size=batch,
                    shuffle=False, num_workers=4, pin_memory=True)
    feats = []
    for imgs, _ in dl:
        f = image_encoder(imgs.to(device).type(dtype))[0]
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def auroc(pos, neg):
    """Mann-Whitney U / rank-based AUROC. pos = scores for the positive class."""
    x = np.concatenate([pos, neg])
    r = np.empty(len(x))
    order = np.argsort(x)
    # average ranks for ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def softmax(L):
    L = L - L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--load-epoch", type=int, required=True)
    ap.add_argument("--forget-domain", default="sketch")
    ap.add_argument("--forget-class", default="tiger")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-shots", type=int, default=8)
    ap.add_argument("--per-cell", type=int, default=8,
                    help="retained images sampled per (class,domain) for the negative set")
    ap.add_argument("--tag", default="", help="label printed with the summary line")
    args = ap.parse_args()

    os.makedirs("/tmp/eval_detect_tmp", exist_ok=True)
    device = torch.device("cuda")
    ns = argparse.Namespace(
        root=args.root, output_dir="/tmp/eval_detect_tmp", resume="",
        seed=args.seed, source_domains=None, target_domains=None, transforms=None,
        trainer="IVLP_VL_Adapter_Prompt", backbone="", head="",
        eval_only=True, model_dir=args.ckpt_dir, load_epoch=args.load_epoch, no_train=True,
        num_shots=args.num_shots, forget_domains=[args.forget_domain],
        forget_classes=[args.forget_class],
        domain_class_divided=False, lmd_domain_loss=1.0,
        use_domain_cls_loss=False, is_domain_divided=True,
        forget_loss_type="entropy", no_retain_loss=False,
        csv_file_path="/tmp/eval_detect_tmp/dummy.csv",
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
    lab2cname = trainer.lab2cname

    with torch.no_grad():
        prompts = model.prompt_learner()
        txt = model.text_encoder(prompts, model.tokenized_prompts)
        txt = (txt / txt.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    scale = model.logit_scale.exp().item()
    C = txt.shape[0]
    classnames = [lab2cname[i] for i in range(C)]
    fc = classnames.index(args.forget_class)

    # ---- unseen images: erased concept (positives) and retained classes (negatives) ----
    used = set(it.impath for it in trainer.dm.dataset.train_x)
    pool_file = osp.join(args.ckpt_dir, "forget_pool.txt")
    if osp.exists(pool_file):
        with open(pool_file) as f:
            used |= set(ln.strip() for ln in f if ln.strip())

    forget_items, cell = [], defaultdict(list)
    for d in DOMAINS:
        for p, l in read_split(args.root, d, "train"):
            if p in used:
                continue
            if p.split("/")[-2] == args.forget_class:
                forget_items.append((p, l))
            else:
                cell[(l, d)].append((p, l))
    rng = np.random.RandomState(0)
    retain_items = []
    for k, v in cell.items():
        idx = rng.permutation(len(v))[: args.per_cell]
        retain_items += [v[i] for i in idx]

    print(f"== {len(forget_items)} erased-concept imgs | {len(retain_items)} retained imgs ==",
          flush=True)
    Ff = extract(model.image_encoder, dtype, forget_items, tfm, device)
    Fr = extract(model.image_encoder, dtype, retain_items, tfm, device)

    Lf = scale * (Ff @ txt.T)
    Lr = scale * (Fr @ txt.T)
    conf_f = softmax(Lf).max(1)
    conf_r = softmax(Lr).max(1)

    # positives = erased-concept inputs, detected by LOW confidence -> score = -conf
    a = auroc(-conf_f, -conf_r)

    print("\n" + "=" * 74)
    print("DETECTABILITY: can a confidence threshold flag erased-concept inputs?")
    print("=" * 74)
    print(f"  mean confidence, erased concept : {100*conf_f.mean():5.1f}%")
    print(f"  mean confidence, retained       : {100*conf_r.mean():5.1f}%")
    print(f"  AUROC (higher = more separable) : {a:.3f}")
    print("  >> " + ("erased inputs are FLAGGABLE by routine confidence monitoring."
                     if a > 0.8 else
                     "erased inputs are NOT separable by confidence -- they enter downstream "
                     "decisions silently."))

    # ---- held-out recovery ----
    n = len(Lf)
    perm = np.random.RandomState(1).permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    base_rate = 100.0 * (Lf[B].argmax(1) == fc).mean()
    mu_concept = Lf[A].mean(0)
    rec_concept = 100.0 * ((Lf[B] - mu_concept[None, :]).argmax(1) == fc).mean()
    mu_retain = Lr.mean(0)                     # control: offset from retained data only
    rec_retain = 100.0 * ((Lf[B] - mu_retain[None, :]).argmax(1) == fc).mean()

    print("\n" + "=" * 74)
    print("HELD-OUT RECOVERY: does one fixed bias vector restore the erased class?")
    print("=" * 74)
    print(f"  target top-1 on held-out half, as reported      : {base_rate:5.1f}%")
    print(f"  after subtracting offset estimated on other half: {rec_concept:5.1f}%")
    print(f"  after subtracting offset from RETAINED data     : {rec_retain:5.1f}%  (control)")
    print("  >> " + ("ERASURE IS A REMOVABLE OFFSET: a concept-specific bias vector, estimated "
                     "on disjoint data, restores the class."
                     if rec_concept > 20 and rec_concept > 3 * max(rec_retain, 1e-6) else
                     "no linear recovery: the class is not restored by a fixed offset."))

    print(f"\n[SUMMARY]{(' ' + args.tag) if args.tag else ''} "
          f"conf_forget={100*conf_f.mean():.1f} conf_retain={100*conf_r.mean():.1f} "
          f"auroc={a:.3f} recovery={rec_concept:.1f} recovery_ctrl={rec_retain:.1f}\n")


if __name__ == "__main__":
    main()
