"""Data-integrity + eval-math audit for the Phase-0 diagnostic.

Checks, in order:
  1. held-out classes have ZERO training images
  2. train and test image sets are disjoint (no image-level leakage)
  3. classnames are ordered so that classnames[i] <-> label i (text anchor alignment)
  4. test set composition is as expected (5 img/class/domain)
  5. restricted_acc() math is correct, verified against a hand-built synthetic case
No GPU / no model needed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from types import SimpleNamespace

import torch
import datasets.domainnet_mini_paper_df  # noqa
from train_loop import setup_cfg
from dassl.data.datasets import build_dataset
from phase0_restricted_eval import restricted_acc

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def mk(heldout):
    a = SimpleNamespace(
        root="/home/owais/machine unlearning/ebm_unlearning/data/domainnet",
        output_dir="/tmp/p0audit", resume="", seed=1,
        dataset_config_file="configs/datasets/domainnet_mini_paper_df.yaml",
        config_file="configs/trainers/vit_b16_ep50.yaml", trainer="IVLP_VL_Adapter_Prompt",
        dataset_name="domainnet_mini_paper_df", backbone="", head="", source_domains=None,
        target_domains=None, transforms=None, num_shots=8, forget_domains=["sketch"],
        forget_classes=[], forget_loss_type="entropy", no_retain_loss=False, forget_weight=1.0,
        flat_weight=1.0, suppress_cap=6.0, marg_weight=1.0, forget_pool_size=0, forget_chunk=0,
        exclude_forget_class_from_retain=False, domainloss_weight=30.0, mmd_weight=10.0,
        use_domain_cls_loss=True, is_domain_divided=True, domain_class_divided=False,
        lmd_domain_loss=1.0, eval_only=False, no_train=False, model_dir="", load_epoch=None,
        csv_file_path="/tmp/p0audit/r.csv", heldout_num=heldout, heldout_seed=1234, opts=[])
    return setup_cfg(a)


print("\n=== 1-4: DATA INTEGRITY (heldout_num=26, heldout_seed=1234) ===")
ds = build_dataset(mk(26))
heldout = set(ds.heldout_labels)

# 1. no held-out class in training
leak = [it for it in ds.train_x if it.label in heldout]
check("held-out classes absent from training", len(leak) == 0,
      f"{len(leak)} leaked items")

# 1b. train really is 100 classes x 4 domains x 8 shots
tr_labels = {it.label for it in ds.train_x}
check("training covers exactly 100 classes", len(tr_labels) == 100, f"{len(tr_labels)}")
check("train_x size == 8*100*4", len(ds.train_x) == 3200, f"{len(ds.train_x)}")

# 2. image-level train/test disjointness
tr_paths = {it.impath for it in ds.train_x}
te_paths = {it.impath for it in ds.test}
overlap = tr_paths & te_paths
check("no image appears in both train and test", len(overlap) == 0,
      f"{len(overlap)} overlapping images")

# 3. text-anchor alignment: classnames[i] must correspond to label i
lab2c = ds.lab2cname
aligned = all(ds.classnames[i] == lab2c[i] for i in range(ds.num_classes))
check("classnames[i] <-> label i (text anchors aligned)", aligned)
check("num_classes == 126 (all anchors kept)", ds.num_classes == 126, f"{ds.num_classes}")
check("len(classnames) == 126", len(ds.classnames) == 126, f"{len(ds.classnames)}")

# 4. test composition
from collections import Counter
per_cd = Counter((it.label, it.domain) for it in ds.test)
counts = set(per_cd.values())
check("test has all 126 classes x 4 domains", len(per_cd) == 126 * 4, f"{len(per_cd)} cells")
check("test is 5 img/class/domain", counts == {5}, f"counts={sorted(counts)}")
n_ho_retain = sum(v for (l, d), v in per_cd.items() if l in heldout and d != 3)
n_ho_forget = sum(v for (l, d), v in per_cd.items() if l in heldout and d == 3)
check("held-out retain n == 390", n_ho_retain == 390, f"{n_ho_retain}")
check("held-out forget n == 130", n_ho_forget == 130, f"{n_ho_forget}")

# 4b. held-out test images were never trained on (path level)
ho_te = {it.impath for it in ds.test if it.label in heldout}
check("held-out test images never in training", len(ho_te & tr_paths) == 0)

print("\n=== 5: restricted_acc() MATH, synthetic ground truth ===")
# 4 classes, 2 domains. Text anchors = identity basis so logits are readable.
d = 4
txt = torch.eye(d)
# images: label0 feature points at anchor0 (correct), label1 points at anchor2 (wrong),
# label2 -> anchor2 (correct), label3 -> anchor3 (correct)
feats = torch.stack([txt[0], txt[2], txt[2], txt[3]])
labels = torch.tensor([0, 1, 2, 3])
domains = torch.tensor([0, 0, 1, 1])

# subset {0,1}: only labels 0,1 count. domain mask = all
acc, n = restricted_acc(feats, labels, domains, txt, [0, 1], torch.ones(4, dtype=torch.bool))
# label0 -> scores [1,0] over anchors {0,1} -> picks 0 -> correct
# label1 -> feature is anchor2; scores over {0,1} = [0,0] -> argmax=0 -> predicts 0 != 1 -> wrong
check("subset {0,1}, all domains -> 50% over n=2", abs(acc - 50.0) < 1e-6 and n == 2,
      f"acc={acc} n={n}")

# subset {2,3}: labels 2,3 both correct -> 100%
acc, n = restricted_acc(feats, labels, domains, txt, [2, 3], torch.ones(4, dtype=torch.bool))
check("subset {2,3}, all domains -> 100% over n=2", abs(acc - 100.0) < 1e-6 and n == 2,
      f"acc={acc} n={n}")

# domain filter: subset {2,3} restricted to domain 1 -> both are domain 1 -> 100%, n=2
acc, n = restricted_acc(feats, labels, domains, txt, [2, 3], domains == 1)
check("domain mask honoured", abs(acc - 100.0) < 1e-6 and n == 2, f"acc={acc} n={n}")

# only images whose TRUE label is in the subset are scored (no cross-group contamination)
acc, n = restricted_acc(feats, labels, domains, txt, [0], torch.ones(4, dtype=torch.bool))
check("only true-label-in-subset images are counted", n == 1, f"n={n}")

print("\n" + "=" * 60)
print("AUDIT FAILED: " + ", ".join(FAIL) if FAIL else "ALL AUDIT CHECKS PASSED")
print("=" * 60)
