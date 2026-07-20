"""
Parse the per-run evaluation logs into one table.

Reads <out>/<method>__<concept>__s<seed>.{concept,detect}.log and emits both a
CSV and a printed summary grouped by method, with per-concept rows and
mean +/- std across concepts. Safe to re-run at any time; cells that have not
finished yet are simply reported as missing.
"""
import argparse
import csv
import glob
import math
import os.path as osp
import re
from collections import defaultdict

LN_C = math.log(126)   # 126 classes; effective support = exp(entropy * ln C)


def grab(text, pattern, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_concept_log(path):
    try:
        t = open(path).read()
    except OSError:
        return {}
    d = {
        "conf": grab(t, r"mean confidence of the prediction:\s*([\d.]+)%"),
        "entropy": grab(t, r"normalized prediction entropy:\s*([\d.]+)"),
        "forget_dom": grab(t, r"in forget domain \([a-z_]+\):\s*([\d.]+)%"),
        "other_dom": grab(t, r"in other domains \(mean\):\s*([\d.]+)%"),
        "neighbor": grab(t, r"\(neighbor\) mean, all domains:\s*([\d.]+)%"),
        "retain": grab(t, r"retain \(ALL other classes\) mean:\s*([\d.]+)%"),
    }
    m = re.search(r"are sent \(top classes\):\s*\n\s*(\S+)\s+([\d.]+)%", t)
    if m:
        d["top_sink_class"], d["top_sink"] = m.group(1), float(m.group(2))
    if d.get("entropy") is not None:
        d["eff_support"] = math.exp(d["entropy"] * LN_C)
    return d


def parse_detect_log(path):
    try:
        t = open(path).read()
    except OSError:
        return {}
    return {
        "auroc": grab(t, r"auroc=([\d.]+)"),
        "recovery": grab(t, r"recovery=([\d.]+)"),
        "recovery_ctrl": grab(t, r"recovery_ctrl=([\d.]+)"),
    }


def fmt(v, w=6, p=1):
    return f"{v:{w}.{p}f}" if isinstance(v, float) else " " * (w - 1) + "-"


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, float)]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory containing the run logs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(osp.join(args.out, "*.concept.log"))):
        base = osp.basename(p)[: -len(".concept.log")]
        try:
            method, concept, seed = base.split("__")
        except ValueError:
            continue
        r = {"method": method, "concept": concept, "seed": seed.lstrip("s")}
        r.update(parse_concept_log(p))
        r.update(parse_detect_log(osp.join(args.out, base + ".detect.log")))
        rows.append(r)

    if not rows:
        print(f"no *.concept.log found under {args.out}")
        return

    cols = ["method", "concept", "seed", "forget_dom", "other_dom", "retain", "neighbor",
            "conf", "entropy", "eff_support", "top_sink", "top_sink_class",
            "auroc", "recovery", "recovery_ctrl"]
    out_csv = args.csv or osp.join(args.out, "results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)

    hdr = (f"{'concept':<18}{'forget':>8}{'other':>8}{'retain':>8}{'neigh':>8}"
           f"{'conf':>8}{'eff':>8}{'sink':>8}  {'sink@':<14}{'auroc':>7}{'recov':>7}")
    for method in sorted(by):
        print("\n" + "=" * 108)
        print(f"  {method}   (n={len(by[method])})")
        print("=" * 108)
        print(hdr)
        print("-" * 108)
        for r in sorted(by[method], key=lambda x: x["concept"]):
            print(f"{r['concept']:<18}"
                  f"{fmt(r.get('forget_dom'),8)}{fmt(r.get('other_dom'),8)}"
                  f"{fmt(r.get('retain'),8)}{fmt(r.get('neighbor'),8)}"
                  f"{fmt(r.get('conf'),8)}{fmt(r.get('eff_support'),8)}"
                  f"{fmt(r.get('top_sink'),8)}  {str(r.get('top_sink_class','-')):<14}"
                  f"{fmt(r.get('auroc'),7,3)}{fmt(r.get('recovery'),7)}")
        print("-" * 108)
        line = f"{'MEAN +/- STD':<18}"
        for k, w, p in [("forget_dom", 8, 1), ("other_dom", 8, 1), ("retain", 8, 1),
                        ("neighbor", 8, 1), ("conf", 8, 1), ("eff_support", 8, 1),
                        ("top_sink", 8, 1)]:
            m, s = mean_std([r.get(k) for r in by[method]])
            line += fmt(m, w, p)
        print(line + "  " + " " * 14 +
              fmt(mean_std([r.get('auroc') for r in by[method]])[0], 7, 3) +
              fmt(mean_std([r.get('recovery') for r in by[method]])[0], 7))
        sd = f"{'  (std)':<18}"
        for k in ["forget_dom", "other_dom", "retain", "neighbor", "conf",
                  "eff_support", "top_sink"]:
            _, s = mean_std([r.get(k) for r in by[method]])
            sd += fmt(s, 8)
        print(sd)

    print(f"\nwrote {out_csv}\n")
    print("eff = effective support = exp(entropy x ln 126); 126 = uniform, 1 = one class.")
    print("sink = share of erased-concept inputs landing on the single most common class.")


if __name__ == "__main__":
    main()
