"""
Sink mechanism diagnostics — training-free, run on a trained checkpoint.

We have been ASSUMING the sink (all forgotten tigers -> one class) is explained
by the first two moments of the forget-logit distribution (mean vector mu and
deviation covariance S). That is only true if the deviations are Gaussian /
exchangeable. argmax is an EXTREME-VALUE statistic, so the assumption may be
false. Before building an objective on it, test it.

Four tests:

  A. GEOMETRIC FLOOR (hubness). If features were perfectly isotropic, what would
     the argmax distribution be? This measures the Voronoi-cell sizes of the class
     text embeddings on the sphere. If it is far from uniform, the sink is a
     property of CLIP's TEXT GEOMETRY, not of our method, and no feature-side
     objective can beat that floor.
     A2 repeats it in our actual operating regime: a fixed point in the flat
     (uniform-logit) subspace + small isotropic noise -- i.e. exactly the state
     the proposed objective drives toward. If A2 is non-uniform, the TARGET
     itself is unreachable.

  B. MOMENT SUFFICIENCY. Fit a Gaussian with the SAME mu and S as the real
     forget logits, sample it, compare argmax distributions. Match => moments 1&2
     determine the sink (assumption holds). Mismatch => higher moments / tails
     drive it and the moment-based objective is built on sand.

  C. MEAN COUNTERFACTUAL. Subtract mu from the real logits (perfectly flatten the
     shared component -- what the L_mean term would achieve if it worked
     perfectly) and recompute argmax. Sink collapses => mu-anisotropy is the
     cause. Sink survives => the L_mean term is worthless.

  D. MULTIMODALITY. Cluster the forget features; is the argmax homogeneous within
     clusters and different across them? If so the population is a mixture with
     per-mode leans, and GLOBAL moment control is the wrong tool.

Read-only. Prints a verdict for each test.
"""
import argparse
import os
import os.path as osp
from collections import Counter, defaultdict

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
def extract_features(image_encoder, dtype, items, tfm, device, batch=64):
    dl = DataLoader(ImageListDataset(items, tfm), batch_size=batch,
                    shuffle=False, num_workers=4, pin_memory=True)
    feats = []
    for imgs, _ in dl:
        f = image_encoder(imgs.to(device).type(dtype))[0]
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


# ---------------------------------------------------------------- helpers

def argmax_summary(preds, C, lab2cname, k=6):
    """Return (top-k list, max share, normalized entropy of the argmax histogram)."""
    n = len(preds)
    cnt = Counter(int(x) for x in preds)
    hist = np.zeros(C)
    for c, v in cnt.items():
        hist[c] = v / n
    nz = hist[hist > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(C))
    top = [(lab2cname.get(c, str(c)), 100.0 * v / n) for c, v in cnt.most_common(k)]
    return top, 100.0 * max(hist), ent


def show(title, top, max_share, ent, note=""):
    print(f"\n--- {title} ---", flush=True)
    print(f"  max single-class share : {max_share:5.1f}%   (uniform would be {100.0/126:.2f}%)")
    print(f"  argmax-spread (norm ent): {ent:5.3f}   (1.0 = perfectly uniform sink, 0 = one class)")
    print("  top classes: " + ", ".join(f"{nm} {sh:.1f}%" for nm, sh in top))
    if note:
        print(f"  >> {note}")


def kmeans(X, k, iters=50, seed=0):
    """Minimal numpy k-means (avoids a sklearn dependency)."""
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--load-epoch", type=int, required=True)
    ap.add_argument("--forget-domain", default="sketch")
    ap.add_argument("--forget-class", default="tiger")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-shots", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=20000, help="samples for Test A")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="Test A2 noise scale added to a flat-subspace point")
    ap.add_argument("--clusters", type=int, default=5, help="k for Test D")
    args = ap.parse_args()

    os.makedirs("/tmp/diagnose_sink_tmp", exist_ok=True)
    device = torch.device("cuda")
    ns = argparse.Namespace(
        root=args.root, output_dir="/tmp/diagnose_sink_tmp", resume="",
        seed=args.seed, source_domains=None, target_domains=None, transforms=None,
        trainer="IVLP_VL_Adapter_Prompt", backbone="", head="",
        eval_only=True, model_dir=args.ckpt_dir, load_epoch=args.load_epoch, no_train=True,
        num_shots=args.num_shots, forget_domains=[args.forget_domain],
        forget_classes=[args.forget_class],
        domain_class_divided=False, lmd_domain_loss=1.0,
        use_domain_cls_loss=False, is_domain_divided=True,
        forget_loss_type="entropy", no_retain_loss=False,
        csv_file_path="/tmp/diagnose_sink_tmp/dummy.csv",
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
        txt = (txt / txt.norm(dim=-1, keepdim=True)).float().cpu().numpy()   # (C, d)
    scale = model.logit_scale.exp().item()
    C, d = txt.shape

    # ---- unseen forget-class images across ALL domains ----
    used = set(it.impath for it in trainer.dm.dataset.train_x)
    pool_file = osp.join(args.ckpt_dir, "forget_pool.txt")
    if osp.exists(pool_file):
        with open(pool_file) as f:
            used |= set(ln.strip() for ln in f if ln.strip())
        print(f"== excluding {len(used)} trained images (incl. forget_pool.txt) ==", flush=True)

    items = []
    for dom in DOMAINS:
        for p, l in read_split(args.root, dom, "train"):
            if p not in used and p.split("/")[-2] == args.forget_class:
                items.append((p, l))
    print(f"== extracting {len(items)} unseen '{args.forget_class}' images ==", flush=True)
    F = extract_features(model.image_encoder, dtype, items, tfm, device)   # (n, d)

    L = scale * (F @ txt.T)                    # real forget logits (n, C)
    real_preds = L.argmax(1)
    n = len(L)

    print("\n" + "=" * 78)
    print(f"SINK MECHANISM DIAGNOSTICS   (n={n} unseen '{args.forget_class}', C={C})")
    print("=" * 78)

    top, ms, ent = argmax_summary(real_preds, C, lab2cname)
    show("REAL (observed sink, the thing we are explaining)", top, ms, ent)
    real_hist = np.bincount(real_preds, minlength=C) / n

    # ---------------- Test A: geometric floor / hubness ----------------
    rng = np.random.RandomState(0)
    R = rng.randn(args.n_random, d)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    top_a, ms_a, ent_a = argmax_summary((R @ txt.T).argmax(1), C, lab2cname)
    show("TEST A1: isotropic random features (pure text geometry)", top_a, ms_a, ent_a,
         note=("TEXT GEOMETRY IS SKEWED -> this is a HARD FLOOR no feature-side "
               "objective can beat." if ms_a > 5 else
               "Text geometry is close to symmetric -> no geometric floor; the sink is ours to fix."))

    # A2: our actual target state -- flat-subspace point + small isotropic noise
    Tc = txt - txt.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(Tc, full_matrices=True)
    r = int((sv > 1e-6 * sv[0]).sum())
    flat_basis = Vt[r:].T                       # (d, d-r) null space of centered text
    f0 = flat_basis @ rng.randn(flat_basis.shape[1])
    f0 /= np.linalg.norm(f0)
    N2 = R * args.noise + f0[None, :]
    N2 /= np.linalg.norm(N2, axis=1, keepdims=True)
    top_a2, ms_a2, ent_a2 = argmax_summary((N2 @ txt.T).argmax(1), C, lab2cname)
    show(f"TEST A2: flat-subspace point + isotropic noise (our TARGET state, "
         f"flat dim={flat_basis.shape[1]}, noise={args.noise})", top_a2, ms_a2, ent_a2,
         note=("TARGET STATE IS ITSELF SUNK -> the proposed objective cannot reach a "
               "uniform sink even if it fully converges." if ms_a2 > 5 else
               "Target state gives a near-uniform sink -> the objective's goal is reachable."))

    # ---------------- Test B: moment sufficiency ----------------
    mu = L.mean(0)
    S = np.cov(L, rowvar=False)
    w, Q = np.linalg.eigh(S)
    w = np.clip(w, 0, None)
    A = Q @ np.diag(np.sqrt(w))
    G = mu[None, :] + rng.randn(max(n, 20000), C) @ A.T          # Gaussian surrogate
    g_preds = G.argmax(1)
    top_b, ms_b, ent_b = argmax_summary(g_preds, C, lab2cname)
    g_hist = np.bincount(g_preds, minlength=C) / len(g_preds)
    tv = 0.5 * np.abs(real_hist - g_hist).sum()
    show("TEST B: Gaussian surrogate with the SAME mu and S", top_b, ms_b, ent_b)
    print(f"  total-variation distance to REAL argmax distribution: {tv:.3f}")
    print("  >> " + ("MOMENTS 1&2 ARE SUFFICIENT -> a mu/S-based objective is justified."
                     if tv < 0.15 else
                     "MOMENTS 1&2 DO NOT DETERMINE THE SINK -> tails/higher moments drive it; "
                     "the mu/S objective is built on a false assumption."))

    # ---------------- Test C: mean counterfactual ----------------
    top_c, ms_c, ent_c = argmax_summary((L - mu[None, :]).argmax(1), C, lab2cname)
    show("TEST C: real logits with mu subtracted (perfect L_mean)", top_c, ms_c, ent_c,
         note=("mu WAS the culprit -> flattening the shared component fixes the sink."
               if ms_c < 0.5 * ms else
               "SINK SURVIVES mu REMOVAL -> the shared mean is NOT the cause; the L_mean "
               "term would be worthless."))

    # ---------------- Test D: multimodality ----------------
    print(f"\n--- TEST D: multimodality (k={args.clusters} clusters of forget features) ---")
    lab = kmeans(F, args.clusters, seed=0)
    dominants = []
    for j in range(args.clusters):
        m = lab == j
        if m.sum() < 2:
            continue
        cnt = Counter(int(x) for x in real_preds[m])
        cls, k_ = cnt.most_common(1)[0]
        dominants.append(cls)
        print(f"  cluster {j}: n={int(m.sum()):5d}  dominant = {lab2cname.get(cls, cls):<20} "
              f"({100.0*k_/m.sum():5.1f}% of cluster)")
    uniq = len(set(dominants))
    print("  >> " + (f"MIXTURE STRUCTURE: {uniq} different dominant classes across clusters -> "
                     "global moment control is the wrong tool; the leans are per-mode."
                     if uniq > 1 else
                     "All clusters share ONE dominant class -> a single global lean, not a mixture."))

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  observed sink                     : {ms:5.1f}%")
    print(f"  geometric floor (isotropic feats) : {ms_a:5.1f}%")
    print(f"  floor at our target state (A2)    : {ms_a2:5.1f}%")
    print(f"  sink after perfect mean-flattening: {ms_c:5.1f}%")
    print(f"  moments 1&2 sufficient?           : {'YES' if tv < 0.15 else 'NO'}  (TV={tv:.3f})")
    print(f"  mixture structure?                : {'YES' if uniq > 1 else 'NO'}")
    print("\n  Build the mu/S objective ONLY if: A2 floor is low, TV is small, and Test C")
    print("  shows the sink drops. Otherwise the mechanism is elsewhere.\n")


if __name__ == "__main__":
    main()
