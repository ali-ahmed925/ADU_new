"""Gating test for Subspace-Constrained DDL.

THE PROPOSAL
------------
DDL separates domains using all 512 dimensions, which collapses variance onto a
few between-domain directions (PR -> 3.6) and consumes the class subspace that
unseen text anchors read from. The proposed fix is to run DDL only in the
orthogonal complement of that subspace:

    P    = projector onto span(frozen zero-shot text anchors)
    Pperp= I - P
    L_domain = gamma * CE(g(Pperp z), d) - lambda * MMD^2(Pperp z)

THE ASSUMPTION THIS TESTS
-------------------------
That assumes domain identity is *linearly available* in Pperp. If domain
information lives inside the class subspace instead, DDL cannot work there and
the whole design is dead. This script measures it directly, with no training.

WHAT IS REPORTED
----------------
For each feature source, we fit a linear (multinomial logistic) probe on half the
images and evaluate on the other half:

    domain acc from  z        how separable domains are at all (upper bound)
    domain acc from  P z      domain info sitting INSIDE the class subspace
    domain acc from  Pperp z  <-- THE NUMBER. Is there enough signal for DDL?

    class acc  from  z / P z / Pperp z    sanity: class info should concentrate
                                          in P, not Pperp.

DECISION
--------
  Pperp domain acc high (say >80%)  -> Tier 1 viable; domain identity is carried
                                       by directions orthogonal to class content.
  Pperp domain acc near chance (25%)-> Tier 1 dead; domain and class share
                                       directions and DDL cannot avoid the
                                       class subspace.

USAGE
    python phase0_subspace_check.py --forget real \
        --control-dir ~/adu_results/phase0_real_control_s1
"""
import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

import torch
from sklearn.linear_model import LogisticRegression

import datasets.domainnet_mini_paper_df  # noqa: F401
import trainers.independent_VLAdapter_Prompt  # noqa: F401

from phase0_diagnostic import DATA_ROOT
from phase0_geometry import features_zeroshot, features_from_checkpoint

DOMS = ["clipart", "painting", "real", "sketch"]


def projector(txt):
    """Orthonormal basis of span(text anchors). Returns (basis, rank)."""
    w = txt / txt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u, s, vh = torch.linalg.svd(w.double(), full_matrices=False)
    basis = vh[s > 1e-6]                     # (r, d)
    return basis.float(), basis.shape[0]


def split_components(feats, basis):
    """Return (z, Pz, Pperp z), all in the original d-dim space."""
    z = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    par = (z @ basis.T) @ basis              # component inside span(W)
    perp = z - par
    return z, par, perp


def probe(x, y, seed=0):
    """Linear probe accuracy, 50/50 split, trained on half."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)
    half = x.shape[0] // 2
    tr, te = idx[:half], idx[half:]
    # NOTE: no multi_class/n_jobs -- removed/deprecated in sklearn >=1.8, and
    # multinomial is the default for multiclass with lbfgs across versions.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x[tr].numpy(), y[tr].numpy())
    return 100.0 * clf.score(x[te].numpy(), y[te].numpy())


def report(name, feats, labels, domains, basis, rank, out):
    z, par, perp = split_components(feats, basis)
    e_par = float((par ** 2).sum(1).mean())
    e_perp = float((perp ** 2).sum(1).mean())
    row = dict(source=name, rank=rank, E_par=e_par, E_perp=e_perp)
    for tag, x in (("z", z), ("Pz", par), ("Pperp_z", perp)):
        row[f"dom_{tag}"] = probe(x, domains)
        row[f"cls_{tag}"] = probe(x, labels)
    out.append(row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--forget", type=str, default="real")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--heldout_num", type=int, default=26)
    p.add_argument("--heldout_seed", type=int, default=1234)
    p.add_argument("--root", type=str, default=DATA_ROOT)
    p.add_argument("--control-dir", type=str, default=None,
                   help="optional: also probe the CONTROL checkpoint's features")
    p.add_argument("--out", type=str, default="phase0_subspace.json")
    cli = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu

    # Frozen zero-shot anchors define P -- this is what the mechanism would use,
    # because the geometry we want to protect is CLIP's original one.
    zs_feats, labels, domains, zs_txt, _ = features_zeroshot(cli.forget, cli)
    basis, rank = projector(zs_txt)

    rows = []
    r = report("zero-shot CLIP", zs_feats, labels, domains, basis, rank, rows)
    if cli.control_dir:
        c_feats, c_lab, c_dom, _, _ = features_from_checkpoint(
            osp.expanduser(cli.control_dir), cli.forget, "control", cli)
        report("control (tuned)", c_feats, c_lab, c_dom, basis, rank, rows)

    hdr = (f"{'features':<18}{'E_par':>7}{'E_perp':>8}"
           f"{'dom(z)':>9}{'dom(Pz)':>9}{'dom(Pperp)':>12}"
           f"{'cls(z)':>9}{'cls(Pz)':>9}{'cls(Pperp)':>12}")
    print("\n" + "=" * len(hdr))
    print(f"SUBSPACE SEPARABILITY  (span(W) rank = {rank} of {zs_feats.shape[1]} dims)")
    print(f"chance: domain 25.00%, class {100/126:.2f}%")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['source']:<18}{r['E_par']:>7.3f}{r['E_perp']:>8.3f}"
              f"{r['dom_z']:>9.2f}{r['dom_Pz']:>9.2f}{r['dom_Pperp_z']:>12.2f}"
              f"{r['cls_z']:>9.2f}{r['cls_Pz']:>9.2f}{r['cls_Pperp_z']:>12.2f}")
    print("=" * len(hdr))

    d_perp = rows[0]["dom_Pperp_z"]
    d_full = rows[0]["dom_z"]
    print(f"\nVERDICT (zero-shot): domain accuracy in Pperp = {d_perp:.2f}% "
          f"({d_perp/max(d_full,1e-9)*100:.0f}% of the {d_full:.2f}% available in z)")
    if d_perp >= 80:
        print("  -> TIER 1 VIABLE. Domain identity is carried by directions")
        print("     orthogonal to the class subspace; DDL can be confined there.")
    elif d_perp <= 40:
        print("  -> TIER 1 DEAD. Domain identity lives inside the class subspace;")
        print("     constraining DDL to Pperp would destroy its ability to separate")
        print("     domains. A different design is needed.")
    else:
        print("  -> MARGINAL. Some domain signal survives in Pperp but it is")
        print("     degraded; expect weaker forgetting. Consider a soft penalty on")
        print("     the class-subspace component instead of a hard projection.")

    with open(cli.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {cli.out}")


if __name__ == "__main__":
    main()
