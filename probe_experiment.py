"""
Linear-probe de-risk experiment for the "shallow forgetting" hypothesis.

Question: after ADU unlearns the sketch domain (top-1 collapses to ~25%),
is the class information still LINEARLY RECOVERABLE from the image features?

Three key numbers:
  1. ADU top-1 on sketch test          (the paper's "forgotten" metric, ~25%)
  2. Linear probe on ADU sketch feats  (THE headline number)
  3. Linear probe on zero-shot CLIP sketch feats (upper bound / control)
Plus a retained-domain contrast (clipart) as sanity.

No retraining. Read-only on the checkpoint.
"""
import argparse
import os
import os.path as osp
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dassl.engine import build_trainer
from dassl.data.transforms import build_transform

# registers datasets/trainers + gives us setup_cfg
import datasets.domainnet_mini_paper_df  # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401
from train_loop import setup_cfg
from clip import clip
from clip.model import build_model as build_clip_model

# ------------------------------------------------------------------
REPO = osp.dirname(osp.abspath(__file__))
DATA_ROOT = "/home/owais/machine unlearning/ebm_unlearning/data/domainnet"
SPLIT_DIR = osp.join(DATA_ROOT, "DomainNet", "splits_mini")
IMG_ROOT = osp.join(DATA_ROOT, "DomainNet")
CKPT_DIR = osp.expanduser("~/adu_results/sketch_seed1/checkpoint_and_log")
OUT_DIR = osp.expanduser("~/adu_results/probe_experiment")
DOMAINS = ["clipart", "painting", "real", "sketch"]
SHOTS_PER_CLASS = 100   # probe-train samples per class
BATCH = 64
EPOCH = 50
# ------------------------------------------------------------------


def read_split(domain, split):
    items = []
    with open(osp.join(SPLIT_DIR, f"{domain}_{split}.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            impath, label = line.split(" ")
            items.append((osp.join(IMG_ROOT, impath), int(label)))
    return items


class ImageListDataset(Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), label


@torch.no_grad()
def extract_features(image_encoder, dtype, items, tfm, device, desc=""):
    ds = ImageListDataset(items, tfm)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4,
                    pin_memory=True)
    feats, labels = [], []
    for bi, (imgs, lbs) in enumerate(dl):
        imgs = imgs.to(device)
        f = image_encoder(imgs.type(dtype))[0]  # (cls_feat, patch_feat)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
        labels.append(lbs.numpy())
        if (bi + 1) % 50 == 0:
            print(f"  {desc}: batch {bi+1}/{len(dl)}", flush=True)
    return np.concatenate(feats), np.concatenate(labels)


def build_probe_pool(domain, used_impaths, n_per_class, seed=0):
    """All train-split items of `domain`, minus images ADU trained on,
    subsampled to n_per_class per class."""
    items = read_split(domain, "train")
    items = [(p, l) for p, l in items if p not in used_impaths]
    by_class = {}
    for p, l in items:
        by_class.setdefault(l, []).append((p, l))
    rng = random.Random(seed)
    pool = []
    for l, lst in sorted(by_class.items()):
        k = min(n_per_class, len(lst))
        pool.extend(rng.sample(lst, k))
    return pool


def load_zeroshot_clip(device):
    """Vanilla CLIP ViT-B/16 through this repo's model code.
    depths=0 and all InstaPG flags off => mathematically original CLIP."""
    url = clip._MODELS["ViT-B/16"]
    model_path = clip._download(url)
    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design = {"trainer": "IVLP_VL_Adapter_Prompt",
              "vision_depth": 0, "language_depth": 0,
              "vision_ctx": 0, "language_ctx": 0,
              "use_classtoken": False, "use_cross_attention": False,
              "independent_cross_attention": False,
              "independent_learnable_vision": True,
              "insert_layer": 9}
    model = build_clip_model(state_dict, design)
    return model.to(device).eval()


def fit_probe(train_X, train_y, test_X, test_y, name):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
    clf.fit(train_X, train_y)
    acc = clf.score(test_X, test_y)
    print(f"[PROBE] {name}: {acc*100:.2f}%", flush=True)
    return acc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    # ---- rebuild training cfg exactly as the sketch_seed1 run ----
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
    tfm_test = build_transform(cfg, is_train=False)

    # ---- exact 8-shot images ADU saw (to exclude from probe pools) ----
    used = {d: set() for d in DOMAINS}
    for item in trainer.dm.dataset.train_x:
        used[DOMAINS[item.domain]].add(item.impath)
    print({d: len(s) for d, s in used.items()}, flush=True)

    # ---- data lists ----
    sketch_pool = build_probe_pool("sketch", used["sketch"], SHOTS_PER_CLASS)
    clipart_pool = build_probe_pool("clipart", used["clipart"], SHOTS_PER_CLASS)
    sketch_test = read_split("sketch", "test")
    clipart_test = read_split("clipart", "test")
    print(f"probe pools: sketch={len(sketch_pool)} clipart={len(clipart_pool)}",
          flush=True)

    cache = osp.join(OUT_DIR, "features.npz")
    if osp.exists(cache):
        print("== loading cached features ==", flush=True)
        z = np.load(cache)
        d = {k: z[k] for k in z.files}
    else:
        d = {}
        print("== extracting ADU features ==", flush=True)
        enc = model.image_encoder
        d["adu_sk_tr_X"], d["adu_sk_tr_y"] = extract_features(
            enc, dtype, sketch_pool, tfm_test, device, "ADU sketch-train")
        d["adu_sk_te_X"], d["adu_sk_te_y"] = extract_features(
            enc, dtype, sketch_test, tfm_test, device, "ADU sketch-test")
        d["adu_cl_tr_X"], d["adu_cl_tr_y"] = extract_features(
            enc, dtype, clipart_pool, tfm_test, device, "ADU clipart-train")
        d["adu_cl_te_X"], d["adu_cl_te_y"] = extract_features(
            enc, dtype, clipart_test, tfm_test, device, "ADU clipart-test")

        print("== extracting zero-shot CLIP features ==", flush=True)
        zs = load_zeroshot_clip(device)
        d["zs_sk_tr_X"], d["zs_sk_tr_y"] = extract_features(
            zs.visual, zs.dtype, sketch_pool, tfm_test, device, "ZS sketch-train")
        d["zs_sk_te_X"], d["zs_sk_te_y"] = extract_features(
            zs.visual, zs.dtype, sketch_test, tfm_test, device, "ZS sketch-test")
        del zs
        np.savez(cache, **d)
        print(f"features cached -> {cache}", flush=True)

    # ---- ADU top-1 via its own text head (sanity: ~24.9% sketch) ----
    with torch.no_grad():
        prompts = model.prompt_learner()
        txt = model.text_encoder(prompts, model.tokenized_prompts)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        txt = txt.float().cpu().numpy()
    top1_sk = (np.argmax(d["adu_sk_te_X"] @ txt.T, 1) == d["adu_sk_te_y"]).mean()
    top1_cl = (np.argmax(d["adu_cl_te_X"] @ txt.T, 1) == d["adu_cl_te_y"]).mean()

    print("\n================ RESULTS ================", flush=True)
    print(f"ADU top-1  sketch (forgotten): {top1_sk*100:.2f}%")
    print(f"ADU top-1  clipart (retained): {top1_cl*100:.2f}%")
    a = fit_probe(d["adu_sk_tr_X"], d["adu_sk_tr_y"],
                  d["adu_sk_te_X"], d["adu_sk_te_y"],
                  "ADU features, sketch (HEADLINE)")
    b = fit_probe(d["zs_sk_tr_X"], d["zs_sk_tr_y"],
                  d["zs_sk_te_X"], d["zs_sk_te_y"],
                  "Zero-shot CLIP features, sketch (upper bound)")
    c = fit_probe(d["adu_cl_tr_X"], d["adu_cl_tr_y"],
                  d["adu_cl_te_X"], d["adu_cl_te_y"],
                  "ADU features, clipart (retained sanity)")

    print("\n================ VERDICT ================")
    print(f"top-1 says forgotten: {top1_sk*100:.1f}%  |  "
          f"probe recovers: {a*100:.1f}%  |  ceiling: {b*100:.1f}%")
    gap = (a - top1_sk) * 100
    rel = (a / b) * 100 if b > 0 else 0
    print(f"recoverability gap (probe - top1): {gap:.1f} points")
    print(f"probe retains {rel:.1f}% of the zero-shot ceiling")
    if a >= 0.40:
        print(">>> SHALLOW FORGETTING CONFIRMED — direction is GREEN")
    elif a >= 0.30:
        print(">>> partial signal — borderline, discuss")
    else:
        print(">>> forgetting appears deep — direction is RED")


if __name__ == "__main__":
    main()
