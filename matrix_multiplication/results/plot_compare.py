#!/usr/bin/env python3
"""
Compare matrix-multiplication latency across two (or more) implementations,
each given as its own CSV, with a user-supplied name (e.g. Tiling, WMMA).

Colour encodes implementation; marker encodes datatype; linestyle
encodes device (solid = CPU, dashed = GPU) when a CSV contains more
than one device (e.g. a DatatypeBenchmark-style CPU+GPU file) --
for single-device files (e.g. GPU-only) linestyle is just solid
throughout and the device is omitted from the legend labels.

Uses the SciencePlots matplotlib style if available
(pip install scienceplots) for a clean, paper-ready look; falls back
to manual styling otherwise.

Produces two figures:
  1. Full range, log-log axes: "<output>.png"
  2. Linear y-axis version, saved as "<output>_linear.png" (useful only
     if the latency range in your data isn't too wide -- see note below).

Usage:
    python plot_compare.py [--output PATH] CSV:NAME [CSV:NAME ...]

Example:
    python plot_compare.py \\
        results/TilingBenchmark.csv:Tiling \\
        results/TensorCoresBenchmark.csv:WMMA \\
        --output results/CompareBenchmark.png

Expects each ';'-separated, quoted CSV with a header, columns in order:
    "size";"datatype";"approach";"latency(us)";
(a trailing ';' per line, creating an empty extra column, is handled.)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid", "no-latex"])
    HAVE_SCIENCEPLOTS = True
except ImportError:
    print("Note: 'scienceplots' not installed (pip install scienceplots); "
          "falling back to manual styling.", file=sys.stderr)
    HAVE_SCIENCEPLOTS = False

COLUMN_NAMES = ["size", "dtype", "device", "latency_us"]

IMPL_PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
DTYPE_MARKERS = {
    "int8": "o", "int16": "s", "int32": "^", "int64": "D",
    "float32": "v", "float64": "P", "half": "X", "bf16": "*", "fp16": "X",
}
FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DEVICE_LINESTYLES = {"cpu": "-", "gpu": "--"}
FALLBACK_LINESTYLES = ["-", "--", "-.", ":"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="CSV:NAME pairs, e.g. results/TilingBenchmark.csv:Tiling")
    p.add_argument("--output", "-o", default="results/CompareBenchmark.png", help="Output PNG path")
    return p.parse_args()


def load_csv(path: Path, impl_name: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"CSV not found: {path}")
    df = pd.read_csv(path, sep=";", header=0, quotechar='"', engine="python")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = COLUMN_NAMES
    for col in ["dtype", "device"]:
        df[col] = df[col].astype(str).str.strip().str.strip('"').str.lower()
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["latency_us"] = pd.to_numeric(df["latency_us"], errors="coerce")
    df = df.dropna(subset=["size", "latency_us"])
    df["impl"] = impl_name
    return df


def main():
    args = parse_args()

    frames = []
    impl_names = []
    for item in args.inputs:
        if ":" not in item:
            sys.exit(f"Expected CSV:NAME, got: {item}")
        csv_str, name = item.rsplit(":", 1)
        impl_names.append(name)
        frames.append(load_csv(Path(csv_str), name))
    df = pd.concat(frames, ignore_index=True)

    out_path = Path(args.output)

    impl_colors = {name: IMPL_PALETTE[i % len(IMPL_PALETTE)] for i, name in enumerate(impl_names)}
    dtypes = list(dict.fromkeys(df["dtype"]))  # unique, first-seen order
    fallback_iter = iter(FALLBACK_MARKERS)
    dtype_markers = {}
    for dt in dtypes:
        dtype_markers[dt] = DTYPE_MARKERS.get(dt, next(fallback_iter, "o"))

    devices = list(dict.fromkeys(df["device"]))  # unique, first-seen order
    ls_fallback_iter = iter(FALLBACK_LINESTYLES)
    device_linestyles = {}
    for dev in devices:
        device_linestyles[dev] = DEVICE_LINESTYLES.get(dev, next(ls_fallback_iter, "-"))
    multi_device = len(devices) > 1

    # ------------------------------------------------------------------
    # Plot styling
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "figure.figsize": (9, 5.5),
    })
    if not HAVE_SCIENCEPLOTS:
        plt.rcParams.update({
            "font.family": "serif",
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
        })

    def plot_series(ax, log_y=True):
        for impl in impl_names:
            for dt in dtypes:
                for dev in devices:
                    subset = df[(df["impl"] == impl) & (df["dtype"] == dt) & (df["device"] == dev)].sort_values("size")
                    if subset.empty:
                        continue
                    label = f"{impl} \u2014 {dt}"
                    if multi_device:
                        label += f" ({dev.upper()})"
                    ax.plot(
                        subset["size"],
                        subset["latency_us"],
                        color=impl_colors[impl],
                        marker=dtype_markers[dt],
                        linestyle=device_linestyles[dev],
                        markersize=6,
                        linewidth=2,
                        label=label,
                    )
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log", base=10)
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("Latency (\u00b5s)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    def save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved plot to {path}")

    title = "Matrix Multiplication Benchmark \u2014 " + " vs. ".join(impl_names)

    # Figure 1: log-log
    fig, ax = plt.subplots()
    plot_series(ax, log_y=True)
    ax.set_title(title)
    ax.legend(title="Implementation \u2014 Datatype", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    plt.close(fig)

    # Figure 2: linear y-axis.
    # NOTE: only inspect this one if your latency range doesn't span more
    # than ~1-2 orders of magnitude -- otherwise the small-N points get
    # compressed to the baseline and this figure stops being informative
    # (same caveat as the single-implementation GPU-only script).
    fig, ax = plt.subplots()
    plot_series(ax, log_y=False)
    ax.set_title(title + " (linear scale)")
    ax.legend(title="Implementation \u2014 Datatype", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    lin_out = out_path.with_name(out_path.stem + "_linear" + out_path.suffix)
    save(fig, lin_out)
    plt.close(fig)


if __name__ == "__main__":
    main()
