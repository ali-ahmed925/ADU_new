"""
Phase-0 (corrected): restricted-label-space open-vocabulary eval.

WHY THIS EXISTS
---------------
The first Phase-0 readout was contaminated. The softmax is 126-way, but only
the 100 SEEN classes ever appear as training targets, so the 26 HELD-OUT classes
are *permanent negatives*: retain-CE drives their logits down on every step,
purely because they are never the right answer. That suppression hits the
held-out RETAIN cell and the held-out FORGET cell alike, which is why held-out
retain collapsed to ~15% and the transfer ratio became meaningless.

THE FIX
-------
Score each image ONLY against a label space in which every candidate had the
same training status. Held-out images are ranked over the 26 held-out anchors
only (chance = 1/26). Seen images are ranked over a random 26-subset of the
seen anchors (chance = 1/26 as well), averaged over several draws. Matched
label-space size => matched chance => the SEEN vs HELD-OUT comparison is
apples-to-apples, and the never-correct-negative artifact is gone (within a
restricted space, all candidates share the same status).

This is eval-only: it loads the checkpoint the diagnostic already trained.

USAGE
-----
    python phase0_restricted_eval.py \
        --output-dir ~/adu_results/phase0_sketch_s1 \
        --forget sketch --seed 1 --heldout_num 26 --heldout_seed 1234
"""

import argparse
import glob
import json
import os
import os.path as osp
from collections import defaultdict
from types import SimpleNamespace

import torch
from tqdm import tqdm

from dassl.utils import set_random_seed
from dassl.engine import build_trainer

import datasets.domainnet_df               # noqa: F401
import datasets.office_home_df             # noqa: F401
import datasets.domainnet_mini_df          # noqa: F401
import datasets.domainnet_mini_paper_df    # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401

from train_loop import setup_cfg
from phase0_diagnostic import build_args, DATA_ROOT


def find_checkpoint(output_dir):
    """Locate the saved model dir + latest epoch under output_dir."""
    cands = glob.glob(osp.join(output_dir, "*", "model.pth.tar-*"))
    if not cands:
        raise FileNotFoundError(
            f"no checkpoint under {output_dir} (looked for */model.pth.tar-*).\n"
            "  The checkpoint is written ONLY at the final epoch, so the most\n"
            "  likely cause is that phase0_diagnostic.py is still training --\n"
            "  wait for it to finish, then re-run this.\n"
            "  Otherwise: check --output-dir points at the right run."
        )
    best_ep, best_dir = -1, None
    for c in cands:
        try:
            ep = int(c.rsplit("-", 1)[1])
        except ValueError:
            continue
        if ep > best_ep:
            best_ep, best_dir = ep, osp.dirname(c)
    return osp.dirname(best_dir), best_ep  # (dir containing <model_name>/, epoch)


@torch.no_grad()
def cache_features(trainer):
    """One pass over the test set: cache normalized image features, labels,
    domains, plus the (shared) text features. All later label-space
    restrictions are then just slicing -- no extra forward passes."""
    trainer.set_model_mode("eval")
    feats, labels, domains = [], [], []
    txt = None
    for batch in tqdm(trainer.test_loader, desc="caching features"):
        input, label, domain = trainer.parse_batch_test(batch)
        out = trainer.model_inference(input)
        img_feat, txt_feat = out[1], out[2]
        feats.append(img_feat.float().cpu())
        labels.append(label.cpu())
        domains.append(domain.cpu())
        if txt is None:
            txt = txt_feat.float().cpu()
    return torch.cat(feats), torch.cat(labels), torch.cat(domains), txt


def restricted_acc(feats, labels, domains, txt, class_subset, dom_mask):
    """Top-1 accuracy scoring ONLY over `class_subset` text anchors,
    restricted to images whose true label is in that subset and whose domain
    satisfies dom_mask."""
    subset = sorted(class_subset)
    idx = torch.tensor(subset)
    in_subset = torch.isin(labels, idx) & dom_mask
    if in_subset.sum() == 0:
        return float("nan"), 0
    f = feats[in_subset]
    y = labels[in_subset]
    logits = f @ txt[idx].t()              # (n, |subset|)
    pred_local = logits.argmax(dim=1)
    pred = idx[pred_local]
    return (100.0 * (pred == y).float().mean().item()), int(in_subset.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--forget", type=str, default="sketch")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--heldout_num", type=int, default=26)
    p.add_argument("--heldout_seed", type=int, default=1234)
    p.add_argument("--root", type=str, default=DATA_ROOT)
    p.add_argument("--arm", type=str, default=None,
                   choices=["adu", "control", "forget_only", "ddl_only"],
                   help="the arm this checkpoint was trained with, so the cfg "
                        "used to rebuild the model matches")
    p.add_argument("--control", action="store_true",
                   help="alias for --arm control (backwards compatibility)")
    p.add_argument("--n_draws", type=int, default=20,
                   help="random 26-subsets of SEEN classes to average over")
    p.add_argument("--output-dir", type=str, required=True,
                   help="the phase0 run dir containing the checkpoint")
    cli = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
    args = build_args(cli)
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    trainer = build_trainer(cfg)
    heldout = sorted(getattr(trainer.dm.dataset, "heldout_labels", []))
    assert len(heldout) == cli.heldout_num, "held-out split mismatch"

    ckpt_dir, epoch = find_checkpoint(cli.output_dir)
    print(f"[restricted] loading checkpoint from {ckpt_dir} (epoch {epoch})")
    trainer.load_model(ckpt_dir, epoch=epoch)

    feats, labels, domains, txt = cache_features(trainer)

    all_labels = sorted({int(l) for l in labels.tolist()})
    seen = sorted(set(all_labels) - set(heldout))
    k = len(heldout)

    dom_names = trainer.domain_list
    forget_ids = [dom_names.index(d) for d in trainer.del_domain_list]
    forget_mask = torch.isin(domains, torch.tensor(forget_ids))
    retain_mask = ~forget_mask

    # held-out side: the one true 26-way space
    ho_ret, n_hr = restricted_acc(feats, labels, domains, txt, heldout, retain_mask)
    ho_for, n_hf = restricted_acc(feats, labels, domains, txt, heldout, forget_mask)

    # seen side: average over random 26-subsets (matched chance)
    g = torch.Generator().manual_seed(0)
    s_ret, s_for = [], []
    for _ in range(cli.n_draws):
        perm = torch.randperm(len(seen), generator=g)[:k]
        sub = [seen[i] for i in perm.tolist()]
        a, _ = restricted_acc(feats, labels, domains, txt, sub, retain_mask)
        b, _ = restricted_acc(feats, labels, domains, txt, sub, forget_mask)
        s_ret.append(a); s_for.append(b)
    sr = sum(s_ret) / len(s_ret)
    sf = sum(s_for) / len(s_for)

    chance = 100.0 / k
    supp_seen = 1.0 - sf / sr if sr > 0 else float("nan")
    supp_held = 1.0 - ho_for / ho_ret if ho_ret > 0 else float("nan")
    transfer = supp_held / supp_seen if supp_seen else float("nan")

    L = []
    L.append("")
    L.append("=" * 70)
    L.append("PHASE-0 (CORRECTED): restricted label space, matched %d-way" % k)
    L.append("=" * 70)
    L.append(f"chance = {chance:.2f}%   (both rows ranked over {k} anchors)")
    L.append("")
    L.append(f"{'':>14} | {'RETAIN doms':>12} | {'FORGET dom':>11}")
    L.append("-" * 46)
    L.append(f"{'SEEN cls':>14} | {sr:>11.2f}% | {sf:>10.2f}%   (avg of {cli.n_draws} draws)")
    L.append(f"{'HELD-OUT cls':>14} | {ho_ret:>11.2f}% | {ho_for:>10.2f}%   (n={n_hr}/{n_hf})")
    L.append("-" * 46)
    L.append(f"suppression (1 - forget/retain):  seen={supp_seen:.3f}  held-out={supp_held:.3f}")
    L.append(f"TRANSFER = {transfer:.3f}")
    L.append("")

    if ho_ret < 2 * chance:
        L.append("!! held-out RETAIN is still near chance even in the restricted space.")
        L.append("!! That means open-vocabulary class structure is genuinely destroyed")
        L.append("!! on RETAIN domains -- a finding in itself, but transfer stays")
        L.append("!! unmeasurable. The prompt-tuned-no-unlearning control is required.")
        branch = "UNDETERMINED (open-vocab collapse on retain domains)"
    elif transfer != transfer:
        branch = "UNDETERMINED"
    elif transfer >= 0.70:
        branch = ("B -> forgetting TRANSFERS to unseen vocabulary. Drop M1, keep M2.")
    elif transfer <= 0.30:
        branch = ("A -> forgetting does NOT transfer; ADU suppresses CELLS. M1 essential.")
    else:
        branch = ("C -> PARTIAL transfer. Proceed M2 -> M1, M1 targets the residual leak.")
    L.append(f"VERDICT: Branch {branch}")
    L.append("=" * 70)
    report = "\n".join(L)
    print(report)

    out = {
        "restricted_k": k, "chance": chance,
        "seen_retain": sr, "seen_forget": sf,
        "heldout_retain": ho_ret, "heldout_forget": ho_for,
        "suppression_seen": supp_seen, "suppression_heldout": supp_held,
        "transfer": transfer, "n_draws": cli.n_draws,
        "checkpoint_epoch": epoch,
    }
    with open(osp.join(cli.output_dir, "phase0_restricted.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(osp.join(cli.output_dir, "phase0_restricted.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n[restricted] wrote {osp.join(cli.output_dir, 'phase0_restricted.json')}")


if __name__ == "__main__":
    main()
