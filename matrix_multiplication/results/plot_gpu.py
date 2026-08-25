#!/usr/bin/env python3
"""
GPU-only benchmark plot: latency vs. matrix size, one line per datatype.

Uses the SciencePlots matplotlib style if available
(pip install scienceplots) for a clean, paper-ready look; falls back
to manual styling otherwise.

Produces two figures:
  1. Full range, log-log axes: "<output>.png"
  2. Linear y-axis version, saved as "<output>_linear.png", to make
     smaller inter-datatype differences easier to inspect (they get
     visually compressed on a log scale).

Usage:
    python plot_gpu_benchmark.py [path/to/GPUi16fp16bf16Benchmark.csv] [path/to/output.png]

Expects a ';'-separated, quoted CSV with a header, columns in order:
    "size";"datatype";"approach";"latency(us)";
(a trailing ';' per line, creating an empty extra column, is handled.)
All rows are assumed to be the same "approach" (e.g. all "gpu") — if
the file has more than one, every value present is plotted together.
"""

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

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/GPUi16fp16bf16Benchmark.csv")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/GPUi16fp16bf16Benchmark.png")

COLUMN_NAMES = ["size", "dtype", "device", "latency_us"]

# Colours are assigned dynamically (below) so any set of datatypes works,
# but fixed hints are given for datatypes we know to expect, for
# consistency with the other benchmark figures.
COLOR_HINTS = {
    "int16": "#d95f02",
    "half":  "#1b9e77",
    "bf16":  "#7570b3",
    "fp16":  "#1b9e77",
}
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# ----------------------------------------------------------------------
# Load & clean data
# ----------------------------------------------------------------------
if not CSV_PATH.exists():
    sys.exit(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, sep=";", header=0, quotechar='"', engine="python")
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
df.columns = COLUMN_NAMES

for col in ["dtype", "device"]:
    df[col] = df[col].astype(str).str.strip().str.strip('"').str.lower()

df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["latency_us"] = pd.to_numeric(df["latency_us"], errors="coerce")
df = df.dropna(subset=["size", "latency_us"])

dtypes = list(dict.fromkeys(df["dtype"]))  # unique, preserve first-seen order

colors = {}
markers = {}
palette_iter = iter(PALETTE)
marker_iter = iter(MARKERS)
for dt in dtypes:
    colors[dt] = COLOR_HINTS.get(dt, next(palette_iter))
    markers[dt] = next(marker_iter)

# ----------------------------------------------------------------------
# Plot styling
# ----------------------------------------------------------------------
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


def plot_series(ax, data, log_y=True):
    for dt in dtypes:
        subset = data[data["dtype"] == dt].sort_values("size")
        if subset.empty:
            continue
        ax.plot(
            subset["size"],
            subset["latency_us"],
            color=colors[dt],
            marker=markers[dt],
            markersize=6,
            linewidth=2,
            label=dt,
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


# ----------------------------------------------------------------------
# Figure 1: log-log
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
plot_series(ax, df)
ax.set_title("GPU Benchmark \u2014 int16 / half / bf16")
ax.legend(title="Datatype", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
fig.tight_layout()
save(fig, OUT_PATH)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: linear y-axis, to expose smaller differences
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
plot_series(ax, df, log_y=False)
ax.set_title("GPU Benchmark \u2014 int16 / half / bf16 (linear scale)")
ax.legend(title="Datatype", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
fig.tight_layout()
lin_out = OUT_PATH.with_name(OUT_PATH.stem + "_linear" + OUT_PATH.suffix)
save(fig, lin_out)
plt.close(fig)
