#!/usr/bin/env python3
"""Manufacturing yield & SPC dashboard (CLI).

Loads a production-line CSV and computes:
  - First-pass yield (FPY) and rolled throughput yield (RTY)
  - Defect Pareto (counts + cumulative %)
  - X-bar and R control charts (PNG, one panel per line)
  - Process capability Cp / Cpk for a critical dimension
  - A Markdown summary report

Input CSV columns (one row per unit produced):
    date, line, shift, unit_id, defect_category, dimension_mm
    defect_category is "none" for a good unit, otherwise one of
    scratch / contamination / crack / dimension / other.
    Lines starting with "#" are ignored (synthetic-data banner).

RTY definition used here (documented, standard textbook form):
    The line is modelled as three process steps:
      1. Forming    -> defects: crack, dimension
      2. Coating    -> defects: scratch, contamination
      3. Final check -> defects: other
    RTY = FPY(step 1) x FPY(step 2) x FPY(step 3)
    where each step FPY = 1 - (defects at that step / units produced).

X-bar / R charts:
    Rational subgroups of n consecutive measurements per line
    (default n=5), control limits from the standard constants
    A2 = 0.577, D3 = 0, D4 = 2.114 (n=5). Points outside the limits
    are flagged out-of-control and highlighted on the chart.

Usage:
    python spc_dashboard.py
    python spc_dashboard.py --input data/sample_production.csv --output output
    python spc_dashboard.py --lsl 9.90 --usl 10.10 --subgroup-size 5
"""

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SPC constants for subgroup size n (A2, D3, D4).
SPC_CONSTANTS = {
    2: (1.880, 0.000, 3.267),
    3: (1.023, 0.000, 2.574),
    4: (0.729, 0.000, 2.282),
    5: (0.577, 0.000, 2.114),
    6: (0.483, 0.000, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
}

# Process steps for RTY, mapped from defect categories.
PROCESS_STEPS = {
    "Forming": {"crack", "dimension"},
    "Coating": {"scratch", "contamination"},
    "Final check": {"other"},
}

DEFECT_CATEGORIES = ["scratch", "contamination", "crack", "dimension", "other"]


def load_data(path):
    """Read the production CSV, ignoring '#' comment lines."""
    df = pd.read_csv(path, comment="#", parse_dates=["date"])
    required = {"date", "line", "shift", "unit_id", "defect_category", "dimension_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    return df


def first_pass_yield(df):
    """FPY = good units / total units."""
    total = len(df)
    good = int((df["defect_category"] == "none").sum())
    return good, total, (good / total if total else float("nan"))


def rolled_throughput_yield(df):
    """RTY = product of the three process-step first-pass yields."""
    total = len(df)
    step_yields = {}
    for step, categories in PROCESS_STEPS.items():
        step_defects = int(df["defect_category"].isin(categories).sum())
        step_yields[step] = 1.0 - (step_defects / total if total else 0.0)
    rty = math.prod(step_yields.values())
    return step_yields, rty


def defect_pareto(df):
    """Defect counts per category, sorted desc, with cumulative %."""
    defects = df[df["defect_category"] != "none"]
    counts = defects["defect_category"].value_counts()
    counts = counts.reindex(
        [c for c in DEFECT_CATEGORIES if c in counts.index]
    ).sort_values(ascending=False)
    total_defects = int(counts.sum())
    pareto = pd.DataFrame(
        {
            "category": counts.index,
            "count": counts.values,
            "pct": 100 * counts.values / total_defects if total_defects else 0,
        }
    )
    pareto["cumulative_pct"] = pareto["pct"].cumsum()
    return pareto, total_defects


def capability(values, lsl, usl):
    """Cp / Cpk for a set of measurements against spec limits."""
    mean = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    cp = (usl - lsl) / (6 * sigma) if sigma > 0 else float("nan")
    cpu = (usl - mean) / (3 * sigma) if sigma > 0 else float("nan")
    cpl = (mean - lsl) / (3 * sigma) if sigma > 0 else float("nan")
    cpk = min(cpu, cpl)
    return {"mean": mean, "sigma": sigma, "cp": cp, "cpk": cpk,
            "cpu": cpu, "cpl": cpl}


def xbar_r_charts(df, subgroup_size):
    """Build X-bar / R subgroups of n consecutive measurements per line.

    Returns a dict keyed by line: subgroup means, ranges, centre lines,
    control limits, and the indices of out-of-control subgroups.
    """
    if subgroup_size not in SPC_CONSTANTS:
        raise ValueError(
            f"Subgroup size {subgroup_size} not supported "
            f"(supported: {sorted(SPC_CONSTANTS)})"
        )
    a2, d3, d4 = SPC_CONSTANTS[subgroup_size]
    results = {}
    for line, group in df.groupby("line"):
        values = (
            group.sort_values(["date", "shift", "unit_id"])["dimension_mm"]
            .to_numpy()
        )
        n_subgroups = len(values) // subgroup_size
        trimmed = values[: n_subgroups * subgroup_size]
        subgroups = trimmed.reshape(n_subgroups, subgroup_size)
        xbars = subgroups.mean(axis=1)
        ranges = subgroups.max(axis=1) - subgroups.min(axis=1)
        xbar_bar = float(xbars.mean())
        r_bar = float(ranges.mean())
        ucl_x = xbar_bar + a2 * r_bar
        lcl_x = xbar_bar - a2 * r_bar
        ucl_r = d4 * r_bar
        lcl_r = d3 * r_bar
        ooc_x = [i for i, x in enumerate(xbars) if x > ucl_x or x < lcl_x]
        ooc_r = [i for i, r in enumerate(ranges) if r > ucl_r or r < lcl_r]
        results[line] = {
            "xbars": xbars,
            "ranges": ranges,
            "xbar_bar": xbar_bar,
            "r_bar": r_bar,
            "ucl_x": ucl_x,
            "lcl_x": lcl_x,
            "ucl_r": ucl_r,
            "lcl_r": lcl_r,
            "ooc_x": ooc_x,
            "ooc_r": ooc_r,
            "n_subgroups": n_subgroups,
        }
    return results


def plot_xbar_r(results, out_dir):
    """X-bar chart and R chart PNGs, one panel per line."""
    lines = sorted(results)
    fig, axes = plt.subplots(len(lines), 1, figsize=(12, 4 * len(lines)),
                             sharex=True)
    if len(lines) == 1:
        axes = [axes]
    for ax, line in zip(axes, lines):
        r = results[line]
        x = np.arange(1, r["n_subgroups"] + 1)
        ax.plot(x, r["xbars"], marker="o", ms=3, lw=1, label="Subgroup mean")
        ax.axhline(r["xbar_bar"], color="green", ls="-", label="Centre line")
        ax.axhline(r["ucl_x"], color="red", ls="--", label="UCL")
        ax.axhline(r["lcl_x"], color="red", ls="--", label="LCL")
        if r["ooc_x"]:
            ax.scatter(
                np.array(r["ooc_x"]) + 1, r["xbars"][r["ooc_x"]],
                color="red", s=60, zorder=5, label="Out of control",
            )
        ax.set_title(f"X-bar chart — {line} (subgroup n=5, dimension mm)")
        ax.set_ylabel("Subgroup mean (mm)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Subgroup (consecutive units, in production order)")
    fig.tight_layout()
    path = os.path.join(out_dir, "xbar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(lines), 1, figsize=(12, 4 * len(lines)),
                             sharex=True)
    if len(lines) == 1:
        axes = [axes]
    for ax, line in zip(axes, lines):
        r = results[line]
        x = np.arange(1, r["n_subgroups"] + 1)
        ax.plot(x, r["ranges"], marker="o", ms=3, lw=1, label="Subgroup range")
        ax.axhline(r["r_bar"], color="green", ls="-", label="Centre line")
        ax.axhline(r["ucl_r"], color="red", ls="--", label="UCL")
        ax.axhline(r["lcl_r"], color="red", ls="--", label="LCL")
        if r["ooc_r"]:
            ax.scatter(
                np.array(r["ooc_r"]) + 1, r["ranges"][r["ooc_r"]],
                color="red", s=60, zorder=5, label="Out of control",
            )
        ax.set_title(f"R chart — {line} (subgroup n=5, dimension mm)")
        ax.set_ylabel("Subgroup range (mm)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Subgroup (consecutive units, in production order)")
    fig.tight_layout()
    path = os.path.join(out_dir, "r_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return (
        os.path.join(out_dir, "xbar_chart.png"),
        os.path.join(out_dir, "r_chart.png"),
    )


def plot_pareto(pareto, out_dir):
    """Pareto bar chart with cumulative-% line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pareto))
    ax.bar(x, pareto["count"], color="#4472C4", label="Defect count")
    ax.set_xticks(x)
    ax.set_xticklabels(pareto["category"])
    ax.set_ylabel("Defect count")
    ax.set_xlabel("Defect category")
    ax.set_title("Defect Pareto — count by category (cumulative % on right)")
    ax.grid(True, axis="y", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, pareto["cumulative_pct"], color="#ED7D31", marker="o",
             label="Cumulative %")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)
    for i, row in pareto.iterrows():
        ax.text(i, row["count"], str(int(row["count"])),
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_yield_trend(df, out_dir):
    """Weekly first-pass yield per line."""
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    rows = []
    for (week, line), group in df.groupby(["week", "line"]):
        _, _, fpy = first_pass_yield(group)
        rows.append({"week": week, "line": line, "fpy_pct": 100 * fpy})
    trend = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for line, group in trend.groupby("line"):
        ax.plot(group["week"], group["fpy_pct"], marker="o",
                label=f"{line} weekly FPY")
    ax.set_ylim(85, 101)
    ax.set_ylabel("First-pass yield (%)")
    ax.set_xlabel("Week starting")
    ax.set_title("Weekly first-pass yield by line")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "yield_trend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(path, summary):
    """Write the Markdown summary report."""
    s = summary
    lines = []
    lines.append("# Manufacturing Yield & SPC Summary Report\n")
    lines.append(f"Source file: `{s['input']}`  ")
    lines.append(f"Date range: {s['date_min']} to {s['date_max']}  ")
    lines.append(f"Lines: {', '.join(s['lines'])}  ")
    lines.append(f"Spec limits (critical dimension): "
                 f"LSL {s['lsl']:.2f} mm, USL {s['usl']:.2f} mm\n")

    lines.append("## Overall yield\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Units produced | {s['total_units']:,} |")
    lines.append(f"| Defective units | {s['total_defects']:,} |")
    lines.append(f"| First-pass yield (FPY) | {100 * s['fpy']:.2f}% |")
    lines.append(f"| Rolled throughput yield (RTY) | {100 * s['rty']:.2f}% |")
    for step, y in s["step_yields"].items():
        lines.append(f"| Step FPY — {step} | {100 * y:.2f}% |")
    lines.append("")

    lines.append("## Yield by line\n")
    lines.append("| Line | Units | Defects | FPY | RTY |")
    lines.append("|---|---|---|---|---|")
    for line, v in s["per_line"].items():
        lines.append(
            f"| {line} | {v['units']:,} | {v['defects']:,} | "
            f"{100 * v['fpy']:.2f}% | {100 * v['rty']:.2f}% |"
        )
    lines.append("")
    lines.append("![Weekly yield trend](yield_trend.png)\n")

    lines.append("## Defect Pareto\n")
    lines.append("| Category | Count | % of defects | Cumulative % |")
    lines.append("|---|---|---|---|")
    for _, row in s["pareto"].iterrows():
        lines.append(
            f"| {row['category']} | {int(row['count'])} | "
            f"{row['pct']:.1f}% | {row['cumulative_pct']:.1f}% |"
        )
    lines.append("")
    lines.append("![Defect Pareto](pareto_chart.png)\n")

    lines.append("## Process capability (critical dimension)\n")
    lines.append("| Line | Mean (mm) | Std dev (mm) | Cp | Cpk | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    for line, c in s["capability"].items():
        verdict = (
            "Capable (Cpk >= 1.33)" if c["cpk"] >= 1.33
            else "Marginal (1.00 <= Cpk < 1.33)" if c["cpk"] >= 1.00
            else "NOT CAPABLE (Cpk < 1.00)"
        )
        lines.append(
            f"| {line} | {c['mean']:.4f} | {c['sigma']:.4f} | "
            f"{c['cp']:.2f} | {c['cpk']:.2f} | {verdict} |"
        )
    lines.append("")
    overall = s["capability"]["Overall"]
    lines.append(
        f"Overall: Cp {overall['cp']:.2f}, Cpk {overall['cpk']:.2f} "
        f"(mean {overall['mean']:.4f} mm, sigma {overall['sigma']:.4f} mm).\n"
    )

    lines.append("## Control charts — out-of-control findings\n")
    for line, r in s["charts"].items():
        ooc_x = r["ooc_x"]
        if ooc_x:
            shown = ", #".join(str(i + 1) for i in ooc_x[:12])
            ooc_x_text = f"#{shown}"
            if len(ooc_x) > 12:
                ooc_x_text += f", … and {len(ooc_x) - 12} more"
        else:
            ooc_x_text = "none"
        lines.append(
            f"- **{line}**: {r['n_subgroups']} subgroups of "
            f"{s['subgroup_size']}; X-bar centre {r['xbar_bar']:.4f} mm, "
            f"UCL {r['ucl_x']:.4f}, LCL {r['lcl_x']:.4f}. "
            f"Out-of-control X-bar subgroups: {len(ooc_x)} "
            f"({ooc_x_text}); "
            f"out-of-control R subgroups: {len(r['ooc_r'])}."
        )
    lines.append("")
    lines.append("![X-bar chart](xbar_chart.png)\n")
    lines.append("![R chart](r_chart.png)\n")

    lines.append("## What to do next (root-cause / countermeasure prompts)\n")
    lines.append(
        "- Any subgroup outside the X-bar limits is an out-of-control "
        "signal: stop, contain the affected units, and run a root-cause "
        "check (5 Whys / fishbone) on what changed at that time — "
        "material lot, tooling, setup, operator, or shift handover."
    )
    lines.append(
        "- Attack the Pareto head first: the top defect categories drive "
        "most of the yield loss, so countermeasures there pay back fastest."
    )
    lines.append(
        "- If Cpk < 1.33, the process is not capable of holding spec: "
        "centre the process on nominal and/or reduce variation before "
        "tightening inspection."
    )
    lines.append(
        "- Re-run this tool after every countermeasure to confirm the "
        "process is back in control (no points outside limits, Cpk "
        "recovering) — that is the continuous-improvement loop.\n"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Manufacturing yield & SPC dashboard."
    )
    parser.add_argument("--input", default="data/sample_production.csv",
                        help="Input production CSV")
    parser.add_argument("--output", default="output",
                        help="Output directory for PNGs and report.md")
    parser.add_argument("--lsl", type=float, default=9.90,
                        help="Lower spec limit, mm (default 9.90)")
    parser.add_argument("--usl", type=float, default=10.10,
                        help="Upper spec limit, mm (default 10.10)")
    parser.add_argument("--subgroup-size", type=int, default=5,
                        help="X-bar/R subgroup size (default 5)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_data(args.input)
    good, total, fpy = first_pass_yield(df)
    step_yields, rty = rolled_throughput_yield(df)
    pareto, total_defects = defect_pareto(df)
    charts = xbar_r_charts(df, args.subgroup_size)

    per_line = {}
    cap_by_line = {}
    for line, group in df.groupby("line"):
        g, t, f = first_pass_yield(group)
        _, r = rolled_throughput_yield(group)
        per_line[line] = {"units": t, "defects": t - g, "fpy": f, "rty": r}
        cap_by_line[line] = capability(group["dimension_mm"], args.lsl, args.usl)
    cap_by_line["Overall"] = capability(df["dimension_mm"], args.lsl, args.usl)

    xbar_png, r_png = plot_xbar_r(charts, args.output)
    pareto_png = plot_pareto(pareto, args.output)
    trend_png = plot_yield_trend(df, args.output)

    summary = {
        "input": args.input,
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "lines": sorted(df["line"].unique()),
        "lsl": args.lsl,
        "usl": args.usl,
        "subgroup_size": args.subgroup_size,
        "total_units": total,
        "total_defects": total_defects,
        "fpy": fpy,
        "rty": rty,
        "step_yields": step_yields,
        "per_line": per_line,
        "pareto": pareto,
        "capability": cap_by_line,
        "charts": charts,
    }
    report_path = write_report(os.path.join(args.output, "report.md"), summary)

    print(f"Units: {total:,}  Defects: {total_defects:,}  "
          f"FPY: {100 * fpy:.2f}%  RTY: {100 * rty:.2f}%")
    for line, v in per_line.items():
        print(f"  {line}: FPY {100 * v['fpy']:.2f}%  RTY {100 * v['rty']:.2f}%")
    for line, c in cap_by_line.items():
        print(f"  {line}: Cp {c['cp']:.2f}  Cpk {c['cpk']:.2f}  "
              f"(mean {c['mean']:.4f} mm, sigma {c['sigma']:.4f} mm)")
    for line, r in charts.items():
        print(f"  {line}: {len(r['ooc_x'])} out-of-control X-bar subgroups, "
              f"{len(r['ooc_r'])} out-of-control R subgroups")
    print(f"Wrote: {xbar_png}, {r_png}, {pareto_png}, {trend_png}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
