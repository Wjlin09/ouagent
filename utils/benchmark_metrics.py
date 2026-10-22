"""Benchmark metrics collection and result saving for StreamingBench.

The CSV/JSON field schemas are FIXED so downstream pandas/matplotlib code can
rely on the column names. This module only collects and saves data; it never
touches the inference path or the KV-cache lifecycle.

Result layout (``<out_dir>`` defaults to ``results/streamingbench``):

    <out_dir>/<video_name>/
        <video_name>.json
        <video_name>.csv
        visual_tokens.csv
        kv_memory.csv
    <out_dir>/summary/
        video_summary.csv
        query_summary.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

# Per-video query results (one row per question).
QUERY_FIELDS = [
    "video_name",
    "question_id",
    "task_type",
    "scheduled_timestamp",
    "actual_trigger_timestamp",
    "question",
    "reference_answer",
    "model_output",
    "extracted_choice",
    "correct",
    "query_arrival_wall_time",
    "query_start_wall_time",
    "query_end_wall_time",
    "queue_wait_time",
    "generation_time",
    "end_to_end_latency",
]

# Per-frame visual tokens (resolution -> tokens/frame -> tokens/second).
VISUAL_TOKEN_FIELDS = [
    "video_timestamp",
    "frame_index",
    "width",
    "height",
    "resolution",
    "visual_tokens",
    "fps",
    "tokens_per_second",
]

# Per-append KV/context growth (X = kv_cache_length_tokens).
KV_MEMORY_FIELDS = [
    "wall_time",
    "video_timestamp",
    "frame_index",
    "n_pos_added",
    "kv_cache_length_tokens",
]

# One row per video.
VIDEO_SUMMARY_FIELDS = [
    "video_name",
    "video_duration",
    "num_questions",
    "num_correct",
    "accuracy",
    "total_wall_time",
    "processing_speed_ratio",
    "rtf",
    "avg_end_to_end_latency",
    "avg_generation_time",
    "avg_queue_wait_time",
]

# One row per query (for latency/accuracy plots).
QUERY_SUMMARY_FIELDS = [
    "video_name",
    "question_id",
    "scheduled_timestamp",
    "actual_trigger_timestamp",
    "queue_wait_time",
    "generation_time",
    "end_to_end_latency",
    "correct",
    "extracted_choice",
    "reference_answer",
]


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {f: ("" if row.get(f) is None else row.get(f)) for f in fields}
            )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


class MetricsCollector:
    """Accumulates one video's metrics from the driver and the native backend."""

    def __init__(self, video_name: str, fps: float) -> None:
        self.video_name = video_name
        self.fps = float(fps)
        self.kv_records: list[dict] = []
        self.visual_token_records: list[dict] = []
        self.query_records: list[dict] = []
        self._frame_meta: list[tuple[int, float]] = []

    # -- native side (called from NativeMTMDBackend._append_to_context) -------

    def record_append(
        self,
        wall_time: float,
        n_pos_added: int,
        kv_cache_length_tokens: int,
        images: list[tuple[int, int, int]],
    ) -> None:
        """Record one chunk append.

        ``images`` is a list of ``(width, height, visual_tokens)`` per frame in
        the append, in order.
        """
        self.kv_records.append(
            {
                "wall_time": _iso(wall_time),
                "video_timestamp": None,
                "frame_index": None,
                "n_pos_added": int(n_pos_added),
                "kv_cache_length_tokens": int(kv_cache_length_tokens),
            }
        )
        for width, height, visual_tokens in images:
            vt = int(visual_tokens)
            self.visual_token_records.append(
                {
                    "video_timestamp": None,
                    "frame_index": None,
                    "width": int(width),
                    "height": int(height),
                    "resolution": f"{width}x{height}",
                    "visual_tokens": vt,
                    "fps": self.fps,
                    "tokens_per_second": vt * self.fps,
                }
            )

    # -- driver side ---------------------------------------------------------

    def note_frame(self, frame_index: int, video_timestamp: float) -> None:
        """Record the video frame index/timestamp for the i-th ingested frame."""
        self._frame_meta.append((int(frame_index), float(video_timestamp)))

    def record_query(self, record: dict) -> None:
        self.query_records.append(record)

    def fill_frame_meta(self) -> None:
        """Back-fill video_timestamp/frame_index into the native-collected rows.

        With ``num_frameproc == 1`` each ingested frame maps 1:1 to one
        visual-token row and one KV append, so both are aligned with
        ``_frame_meta`` by index.
        """
        for i, rec in enumerate(self.visual_token_records):
            if i < len(self._frame_meta):
                rec["frame_index"], rec["video_timestamp"] = self._frame_meta[i]
        for i, rec in enumerate(self.kv_records):
            if i < len(self._frame_meta):
                rec["frame_index"], rec["video_timestamp"] = self._frame_meta[i]

    # -- saving --------------------------------------------------------------

    def save_video(self, out_dir: Path, video_summary: dict) -> tuple[Path, Path]:
        """Write ``<video>.json/.csv`` + ``visual_tokens.csv`` + ``kv_memory.csv``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{self.video_name}.json"
        csv_path = out_dir / f"{self.video_name}.csv"
        _write_json(
            json_path,
            {
                "video_name": self.video_name,
                "video_summary": video_summary,
                "queries": self.query_records,
            },
        )
        _write_csv(csv_path, QUERY_FIELDS, self.query_records)
        _write_csv(
            out_dir / "visual_tokens.csv", VISUAL_TOKEN_FIELDS, self.visual_token_records
        )
        _write_csv(out_dir / "kv_memory.csv", KV_MEMORY_FIELDS, self.kv_records)
        return json_path, csv_path


def save_video_summary(rows: list[dict], out_dir: Path) -> Path:
    path = out_dir / "summary" / "video_summary.csv"
    _write_csv(path, VIDEO_SUMMARY_FIELDS, rows)
    return path


def save_query_summary(rows: list[dict], out_dir: Path) -> Path:
    path = out_dir / "summary" / "query_summary.csv"
    _write_csv(path, QUERY_SUMMARY_FIELDS, rows)
    return path
