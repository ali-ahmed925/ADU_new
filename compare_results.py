"""
Compare pretrained baseline vs unlearned model.
Reads the per-class/per-domain table from log.txt files and prints a delta table.

Usage:
    python compare_results.py \
        --baseline_log /tmp/adu/baseline/log.txt \
        --unlearned_log /tmp/adu/unlearned/log.txt \
        --forget_class tiger \
        --forget_domain sketch \
        --show_domains painting real sketch
"""

import argparse
import re


def parse_per_class_domain_table(log_path):
    """Parse the 'per class per domain acc' table from a log file."""
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Find the table
    table_start = None
    for i, line in enumerate(lines):
        if "per class per domain acc" in line:
            table_start = i + 1  # header line
            break

    if table_start is None:
        raise ValueError(f"Could not find 'per class per domain acc' table in {log_path}")

    # Parse header to get domain order
    header_line = lines[table_start].strip()
    domains = [d.strip() for d in header_line.replace("Class", "").split() if d.strip()]

    # Parse each class row
    results = {}
    i = table_start + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line or "===" in line:
            break
        # Each line looks like: "tiger           78.72    % 89.60    % ..."
        parts = re.split(r'\s+%?\s*', line)
        parts = [p for p in parts if p]
        if len(parts) < len(domains) + 1:
            break
        cls = parts[0]
        accs = [float(parts[j + 1]) for j in range(len(domains))]
        results[cls] = dict(zip(domains, accs))
        i += 1

    return domains, results


def print_delta_table(all_domains, baseline, unlearned, forget_class, forget_domain, show_domains=None):
    """Print accuracy delta table."""

    # Filter to only show requested domains (preserve order from log)
    if show_domains:
        domains = [d for d in all_domains if d in show_domains]
    else:
        domains = all_domains

    # Classify each class
    def get_relation(cls, forget_class):
        relations = {
            "tiger":    "forget target",
            "lion":     "large cat",
            "bear":     "large carnivore",
            "zebra":    "large animal",
            "dog":      "domestic animal",
            "horse":    "domestic animal",
            "car":      "vehicle/unrelated",
            "truck":    "vehicle/unrelated",
            "guitar":   "completely unrelated",
            "airplane": "completely unrelated",
        }
        if cls == forget_class:
            return "forget target"
        return relations.get(cls, "other")

    col_w = 12
    total_w = 12 + 25 + len(domains) * (col_w + 1) + 20
    print(f"\nDelta table (Unlearned - Pretrained) | Forget: {forget_class}/{forget_domain}")
    print("=" * total_w)
    header = f"{'Class':<12} {'Relation':<25}"
    for d in domains:
        header += f" {d:<{col_w}}"
    header += "  Analysis"
    print(header)
    print("-" * total_w)

    for cls in sorted(baseline.keys()):
        relation = get_relation(cls, forget_class)
        row = f"{cls:<12} {relation:<25}"
        analysis_flags = []
        for d in domains:
            b = baseline[cls].get(d, None)
            u = unlearned[cls].get(d, None)
            if b is None or u is None:
                row += f" {'N/A':<{col_w}}"
            else:
                delta = u - b
                delta_str = f"{delta:+.0f}%"
                row += f" {delta_str:<{col_w}}"

                # Analysis flags
                if cls == forget_class and d == forget_domain:
                    if delta < -50:
                        analysis_flags.append(f"✓ strong forgetting ({d})")
                    elif delta < -20:
                        analysis_flags.append(f"~ partial forgetting ({d})")
                    else:
                        analysis_flags.append(f"❌ fails to forget ({d})")
                elif cls == forget_class and d != forget_domain:
                    # Cross-domain forgetting check on forget class in other domains
                    if delta < -10:
                        analysis_flags.append(f"✓ cross-domain spread ({d})")
                elif relation in ["large cat", "large carnivore"] and d == forget_domain:
                    if delta < -10:
                        analysis_flags.append(f"✓ cross-class spread ({d})")
                    elif abs(delta) <= 3:
                        analysis_flags.append(f"❌ no cross-class spread ({d})")
                elif relation == "completely unrelated" and abs(delta) > 5:
                    analysis_flags.append(f"❌ spurious forgetting ({d})")

        row += "  " + ", ".join(analysis_flags) if analysis_flags else ""
        print(row)

    print("=" * total_w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_log", required=True, help="Path to baseline (pretrained) log.txt")
    parser.add_argument("--unlearned_log", required=True, help="Path to unlearned model log.txt")
    parser.add_argument("--forget_class", default="tiger", help="The class being forgotten")
    parser.add_argument("--forget_domain", default="sketch", help="The domain being forgotten")
    parser.add_argument(
        "--show_domains", nargs="*", default=["painting", "real", "sketch"],
        help="Domains to show in table (default: painting real sketch)"
    )
    args = parser.parse_args()

    print(f"Parsing baseline: {args.baseline_log}")
    base_domains, base_results = parse_per_class_domain_table(args.baseline_log)
    print(f"Parsing unlearned: {args.unlearned_log}")
    unl_domains, unl_results = parse_per_class_domain_table(args.unlearned_log)

    assert base_domains == unl_domains, "Domain lists don't match between logs!"

    print_delta_table(base_domains, base_results, unl_results,
                      args.forget_class, args.forget_domain,
                      show_domains=args.show_domains)


if __name__ == "__main__":
    main()
