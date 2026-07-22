"""Representation geometry: does unlearning COLLAPSE the feature space?

WHY
---
Phase-0 established that unlearning costs 14.1 (sketch) / 50.8 (real) accuracy
points on UNSEEN vocabulary in the RETAIN domains. The proposed remedy (M2)
assumes a mechanism: entropy maximisation is minimised by driving the
class-subspace component of the features to zero, and that collapse -- carried
into the retain domains by the shared prompts -- destroys the geometry unseen
text anchors depend on.

That mechanism has never been measured. This script measures it, on checkpoints
that already exist. If the geometry looks the same in ADU and Control arms, the
mechanism story is wrong and M2 is aimed at nothing.

METRICS (all from one cached forward pass per arm)
--------------------------------------------------
  erank   effective rank, exp(entropy of the singular-value distribution) of the
          centred feature matrix. "How many dimensions are actually in use."
          Lower => more degenerate.
  PR      participation ratio (sum L)^2 / sum L^2 over covariance eigenvalues.
          Same idea, different weighting; reported as a robustness check.
  E_cls   class-subspace energy: ||P z||^2 where P projects onto span(text
          anchors). THIS is the direct test of the collapse hypothesis -- if
          unlearning drives z_parallel -> 0, E_cls drops.
  margin  cos(z, w_true) - mean_{c != true} cos(z, w_c). The quantity accuracy
          actually depends on; included so geometry can be tied to behaviour.

Each is reported for the four groups: {forget, retain} domains x {seen, held-out}
classes, so we can see WHERE any collapse lives.

USAGE
    python phase0_geometry.py \
        --run "sketch-ADU=$HOME/adu_results/phase0_sketch_s1=sketch" \
        --run "sketch-CTL=$HOME/adu_results/phase0_control_s1=sketch:control" \
        --run "real-ADU=$HOME/adu_results/phase0_real_s1=real" \
        --run "real-CTL=$HOME/adu_results/phase0_real_control_s1=real:control" \
        --zeroshot sketch --zeroshot real
"""
import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

import torch
from tqdm import tqdm

import datasets.domainnet_mini_paper_df  # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401
from dassl.engine import build_trainer

from train_loop import setup_cfg
from phase0_diagnostic import build_args, DATA_ROOT
from phase0_restricted_eval import find_checkpoint
from phase0_zeroshot_control import load_vanilla_clip
from engine.dataset_manager import DataManager
from clip import clip

DOMS = ["clipart", "painting", "real", "sketch"]


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def effective_rank(feats):
    """exp(entropy of normalised singular values) of the centred matrix."""
    x = feats - feats.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.double())
    s = s[s > 1e-10]
    if s.numel() == 0:
        return float("nan"), float("nan")
    p = s / s.sum()
    erank = float(torch.exp(-(p * p.log()).sum()))
    lam = s ** 2                      # covariance eigenvalues
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    return erank, pr


def class_subspace_energy(feats, txt):
    """Fraction of (unit-norm) feature energy lying in span(text anchors)."""
    # orthonormal basis of the row space of txt  (anchors x dim)
    u, s, vh = torch.linalg.svd(txt.double(), full_matrices=False)
    basis = vh[s > 1e-8]                      # (r, dim)
    z = feats.double()
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    proj = z @ basis.T                        # (n, r)
    return float((proj ** 2).sum(1).mean())


def mean_margin(feats, labels, txt):
    z = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w = txt / txt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = z @ w.T                             # (n, K) cosines
    true = sim.gather(1, labels.view(-1, 1)).squeeze(1)
    K = sim.size(1)
    others = (sim.sum(1) - true) / (K - 1)
    return float((true - others).mean())


# ----------------------------------------------------------------------------
# feature extraction
# ----------------------------------------------------------------------------
@torch.no_grad()
def features_from_checkpoint(ckpt_dir, forget, control, cli):
    c = SimpleNamespace(output_dir=ckpt_dir, seed=cli.seed, forget=forget,
                        heldout_num=cli.heldout_num, heldout_seed=cli.heldout_seed,
                        gpu=cli.gpu, root=cli.root, control=control)
    cfg = setup_cfg(build_args(c))
    trainer = build_trainer(cfg)
    d, ep = find_checkpoint(ckpt_dir)
    trainer.load_model(d, epoch=ep)
    trainer.set_model_mode("eval")

    feats, labels, domains, txt = [], [], [], None
    for batch in tqdm(trainer.test_loader, desc=osp.basename(ckpt_dir), leave=False):
        img, lab, dom = trainer.parse_batch_test(batch)
        out = trainer.model_inference(img)
        feats.append(out[1].float().cpu())
        labels.append(lab.cpu())
        domains.append(dom.cpu())
        if txt is None:
            txt = out[2].float().cpu()
    heldout = sorted(getattr(trainer.dm.dataset, "heldout_labels", []))
    return torch.cat(feats), torch.cat(labels), torch.cat(domains), txt, heldout


@torch.no_grad()
def features_zeroshot(forget, cli):
    c = SimpleNamespace(output_dir="/tmp/geo_zs", seed=cli.seed, forget=forget,
                        heldout_num=cli.heldout_num, heldout_seed=cli.heldout_seed,
                        gpu=cli.gpu, root=cli.root, control=False)
    cfg = setup_cfg(build_args(c))
    dm = DataManager(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vanilla_clip(cfg).to(dev).eval()
    names = [n.replace("_", " ") for n in dm.dataset.classnames]
    tok = torch.cat([clip.tokenize(f"a photo of a {n}.") for n in names]).to(dev)
    txt = model.encode_text(tok).float().cpu()

    feats, labels, domains = [], [], []
    for batch in tqdm(dm.test_loader, desc=f"zeroshot-{forget}", leave=False):
        f = model.encode_image(batch["img"].to(dev))
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.float().cpu())
        labels.append(batch["label"])
        domains.append(batch["domain"])
    return (torch.cat(feats), torch.cat(labels), torch.cat(domains), txt,
            sorted(dm.dataset.heldout_labels))


# ----------------------------------------------------------------------------
def analyse(name, feats, labels, domains, txt, heldout, forget):
    ho = torch.tensor(sorted(heldout))
    is_held = torch.isin(labels, ho)
    is_forget = domains == DOMS.index(forget)
    rows = []
    for dgroup, dmask in (("forget", is_forget), ("retain", ~is_forget)):
        for cgroup, cmask in (("seen", ~is_held), ("held-out", is_held)):
            m = dmask & cmask
            n = int(m.sum())
            if n < 10:
                continue
            f = feats[m]
            er, pr = effective_rank(f)
            rows.append(dict(arm=name, domains=dgroup, classes=cgroup, n=n,
                             erank=er, pr=pr,
                             e_cls=class_subspace_energy(f, txt),
                             margin=mean_margin(f, labels[m], txt)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[],
                   help='"label=/path/to/run=forgetdomain[:control]"')
    p.add_argument("--zeroshot", action="append", default=[],
                   help="forget domain to build a zero-shot reference for")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--heldout_num", type=int, default=26)
    p.add_argument("--heldout_seed", type=int, default=1234)
    p.add_argument("--root", type=str, default=DATA_ROOT)
    p.add_argument("--out", type=str, default="phase0_geometry.json")
    cli = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu

    rows = []
    for fd in cli.zeroshot:
        rows += analyse(f"zeroshot({fd})", *features_zeroshot(fd, cli), fd)
    for spec in cli.run:
        label, path, dom = spec.split("=", 2)
        control = dom.endswith(":control")
        dom = dom.replace(":control", "")
        rows += analyse(label, *features_from_checkpoint(
            osp.expanduser(path), dom, control, cli), dom)

    hdr = f"{'arm':<16}{'domains':<9}{'classes':<10}{'n':>5}" \
          f"{'erank':>9}{'PR':>8}{'E_cls':>9}{'margin':>9}"
    print("\n" + "=" * len(hdr))
    print("REPRESENTATION GEOMETRY")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    last = None
    for r in rows:
        if last and r["arm"] != last:
            print("-" * len(hdr))
        print(f"{r['arm']:<16}{r['domains']:<9}{r['classes']:<10}{r['n']:>5}"
              f"{r['erank']:>9.2f}{r['pr']:>8.2f}{r['e_cls']:>9.4f}{r['margin']:>9.4f}")
        last = r["arm"]
    print("=" * len(hdr))
    print("erank/PR low  => degenerate (collapsed) representation")
    print("E_cls low     => features pushed OUT of the class subspace (z_par -> 0)")
    print("margin low    => true class no longer stands out; predicts low accuracy")

    with open(cli.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {cli.out}")


if __name__ == "__main__":
    main()
