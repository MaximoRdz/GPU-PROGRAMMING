#!/usr/bin/env python3
"""
Matrix multiplication benchmark plots: latency vs. matrix size,
grouped by datatype (colour) and device (line style / marker).

Uses the SciencePlots matplotlib style if available
(pip install scienceplots) for a clean, paper-ready look; falls back
to manual styling otherwise.

Produces two figures:
  1. Full comparison: all datatypes x devices, log-log axes.
  2. GPU-only: same datatypes, linear y-axis, saved as "<name>_gpu.png",
     to make the smaller inter-datatype differences on GPU (which get
     visually compressed on a log scale in Fig. 1) easier to inspect.

Usage:
    python plot_benchmark.py [path/to/DatatypeBenchmark.csv] [path/to/output.png]

Expects a ';'-separated, quoted CSV with a header, columns in order:
    "size";"datatype";"approach";"latency(us)";
(a trailing ';' per line, creating an empty extra column, is handled.)
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
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/DatatypeBenchmark.csv")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/DatatypeBenchmark.png")

COLUMN_NAMES = ["size", "dtype", "device", "latency_us"]

# ColorBrewer "Dark2" palette, one colour per datatype.
COLORS = {
    "int8":    "#1b9e77",
    "int16":   "#d95f02",
    "int32":   "#7570b3",
    "int64":   "#e7298a",
    "float32": "#66a61e",
    "float64": "#e6ab02",
}

# Line style + marker per device.
DEVICE_STYLE = {
    "cpu": dict(linestyle="-",  marker="o"),
    "gpu": dict(linestyle="--", marker="s"),
}

DTYPE_ORDER = ["int8", "int16", "int32", "int64", "float32", "float64"]
DEVICE_ORDER = ["cpu", "gpu"]

# ----------------------------------------------------------------------
# Load & clean data
# ----------------------------------------------------------------------
if not CSV_PATH.exists():
    sys.exit(f"CSV not found: {CSV_PATH}")

# File has a quoted header and a trailing ';' on every line (creating a
# stray empty 5th column) — quotechar handles the quoting, and we simply
# drop any extra trailing column after reading.
df = pd.read_csv(
    CSV_PATH,
    sep=";",
    header=0,
    quotechar='"',
    engine="python",
)

# Drop unnamed/empty trailing column(s) caused by the trailing ';'.
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

# Map actual headers -> our fixed internal names, by position.
df.columns = COLUMN_NAMES

# Strip stray whitespace / quotes from string columns and coerce numerics.
for col in ["dtype", "device"]:
    df[col] = df[col].astype(str).str.strip().str.strip('"').str.lower()

df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["latency_us"] = pd.to_numeric(df["latency_us"], errors="coerce")
df = df.dropna(subset=["size", "latency_us"])

# ----------------------------------------------------------------------
# Plot styling
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "figure.figsize": (14, 8.5),
})
if not HAVE_SCIENCEPLOTS:
    # Manual fallback approximating the "science" look.
    plt.rcParams.update({
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })


def plot_series(ax, data, dtypes, devices, log_y=True):
    """Draw one line per (dtype, device) combination present in `data`."""
    for dtype in dtypes:
        for device in devices:
            subset = data[(data["dtype"] == dtype) & (data["device"] == device)].sort_values("size")
            if subset.empty:
                continue
            style = DEVICE_STYLE[device]
            label = dtype if len(devices) == 1 else f"{dtype} \u2014 {device.upper()}"
            ax.plot(
                subset["size"],
                subset["latency_us"],
                color=COLORS[dtype],
                linestyle=style["linestyle"],
                marker=style["marker"],
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {path}")


# ----------------------------------------------------------------------
# Figure 1: full comparison, all datatypes x devices
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
plot_series(ax, df, DTYPE_ORDER, DEVICE_ORDER)
ax.set_title("Matrix Multiplication Benchmark")
ax.legend(title="Datatype / Device", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
fig.tight_layout()
save(fig, OUT_PATH)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: GPU-only, linear y-axis to expose the smaller inter-datatype
# differences that get visually compressed on a log scale in Figure 1.
# ----------------------------------------------------------------------
gpu_df = df[df["device"] == "gpu"]
if gpu_df.empty:
    print("No GPU rows found — skipping GPU-only figure.", file=sys.stderr)
else:
    fig, ax = plt.subplots()
    plot_series(ax, gpu_df, DTYPE_ORDER, ["gpu"], log_y=False)
    ax.set_title("Matrix Multiplication Benchmark \u2014 GPU only")
    ax.legend(title="Datatype", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    gpu_out = OUT_PATH.with_name(OUT_PATH.stem + "_gpu" + OUT_PATH.suffix)
    save(fig, gpu_out)
    plt.close(fig)
