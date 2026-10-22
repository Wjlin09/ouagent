#!/usr/bin/env python3
# python scripts/plot_streamingbench_metrics.py --video-dir results/streamingbench/sample_53

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Query Latency
# ============================================================

def plot_latency(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    required = [
        "question_id",
        "queue_wait_time",
        "generation_time",
        "end_to_end_latency",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing latency columns: {missing}"
        )

    df = df.copy()
    df["question_id"] = df["question_id"].astype(str)

    # seconds -> milliseconds
    df["queue_wait_ms"] = (
        pd.to_numeric(df["queue_wait_time"], errors="coerce") * 1000
    )

    df["generation_ms"] = (
        pd.to_numeric(df["generation_time"], errors="coerce") * 1000
    )

    df["e2e_ms"] = (
        pd.to_numeric(df["end_to_end_latency"], errors="coerce") * 1000
    )

    df = df.dropna(
        subset=[
            "queue_wait_ms",
            "generation_ms",
            "e2e_ms",
        ]
    )

    x = list(range(len(df)))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Queue waiting
    ax.bar(
        x,
        df["queue_wait_ms"],
        label="Queue Wait",
    )

    # Generation time stacked on top
    ax.bar(
        x,
        df["generation_ms"],
        bottom=df["queue_wait_ms"],
        label="Generation",
    )

    # End-to-end latency
    ax.plot(
        x,
        df["e2e_ms"],
        marker="o",
        linewidth=1.8,
        label="End-to-End Latency",
    )

    ax.set_xlabel("Query")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("StreamingBench Query Latency")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Q{i + 1}" for i in range(len(df))]
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = output_dir / "latency_breakdown.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[Saved] {output_path}")


# ============================================================
# 2. KV Cache Growth
# ============================================================

def plot_kv_cache(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    required = [
        "n_pos_added",
        "kv_cache_length_tokens",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing KV cache columns: {missing}"
        )

    df = df.copy()

    # Convert to numeric
    df["n_pos_added"] = pd.to_numeric(
        df["n_pos_added"],
        errors="coerce",
    )

    df["kv_cache_length_tokens"] = pd.to_numeric(
        df["kv_cache_length_tokens"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "n_pos_added",
            "kv_cache_length_tokens",
        ]
    )

    # --------------------------------------------------------
    # Cumulative number of newly added input positions
    # --------------------------------------------------------
    df["cumulative_input_tokens"] = (
        df["n_pos_added"].cumsum()
    )

    # Sort according to actual recording order
    if "wall_time" in df.columns:
        df["wall_time"] = pd.to_numeric(
            df["wall_time"],
            errors="coerce",
        )

        if df["wall_time"].notna().all():
            df = df.sort_values("wall_time")

            # Recalculate after sorting
            df["cumulative_input_tokens"] = (
                df["n_pos_added"].cumsum()
            )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(
        df["cumulative_input_tokens"],
        df["kv_cache_length_tokens"],
        marker=".",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Cumulative Input Tokens Added"
    )

    ax.set_ylabel(
        "KV Cache Length (tokens)"
    )

    ax.set_title(
        "KV Cache Growth with Increasing Input Tokens"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = output_dir / "kv_cache_growth.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[Saved] {output_path}")


# ============================================================
# 3. Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Plot StreamingBench latency and KV-cache growth."
        )
    )

    parser.add_argument(
        "--video-dir",
        required=True,
        help=(
            "Directory containing one video's "
            "StreamingBench results."
        ),
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {video_dir}"
        )

    # --------------------------------------------------------
    # Find query CSV
    # --------------------------------------------------------
    csv_files = list(
        video_dir.glob("*.csv")
    )

    latency_files = [
        p
        for p in csv_files
        if p.name not in {
            "visual_tokens.csv",
            "kv_memory.csv",
        }
    ]

    if not latency_files:
        raise FileNotFoundError(
            f"No query CSV found in {video_dir}"
        )

    if len(latency_files) > 1:
        print(
            "[Warning] Multiple query CSV files found. "
            f"Using: {latency_files[0]}"
        )

    latency_csv = latency_files[0]

    # --------------------------------------------------------
    # KV cache CSV
    # --------------------------------------------------------
    kv_csv = video_dir / "kv_memory.csv"

    if not kv_csv.exists():
        raise FileNotFoundError(
            f"Missing KV cache file: {kv_csv}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    output_dir = video_dir / "plots"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[Input] latency: {latency_csv}")
    print(f"[Input] KV cache: {kv_csv}")

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------
    plot_latency(
        latency_csv,
        output_dir,
    )

    plot_kv_cache(
        kv_csv,
        output_dir,
    )

    print("[Done]")


if __name__ == "__main__":
    main()