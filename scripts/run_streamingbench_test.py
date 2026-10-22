"""StreamingBench real-time test driver for OURS_Agent.

Streams the video on its timeline, triggering QA pairs at their specific
``time_stamp``. Generation and video ingestion are interleaved, allowing 
new frames to be appended to the KV cache mid-query.

Data Mapping:
    The video ID is extracted from its directory name and matched against the ``question_id`` token in the CSV. 
    The script will error out if the match is missing or ambiguous.

Command:

    # 批量
    python scripts/run_streamingbench_test.py --video-dir data/StreamingBench/video

    # 单视频
    python scripts/run_streamingbench_test.py --video data/StreamingBench/video/sample_55/video.mp4

"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from utils.benchmark_metrics import (
    QUERY_SUMMARY_FIELDS,
    MetricsCollector,
    save_query_summary,
    save_video_summary,
)

# Allow running as `python scripts/run_streamingbench_test.py` from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger("streamingbench")


def load_config(path: str | Path) -> dict:
    """Load a YAML config (inlined so parsing/validation stays cv2/model-free)."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# StreamingBench CSV parsing / QA mapping
# ---------------------------------------------------------------------------


def parse_time_stamp(text: str) -> float:
    """Parse a StreamingBench ``time_stamp`` into seconds on the video timeline."""
    parts = (text or "").strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"cannot parse time_stamp {text!r}")


def parse_options(text: str) -> list[str]:
    """Parse the CSV ``options`` field (a Python-list string) into a list."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def find_qa_for_video(rows: list[dict], video_key: str) -> list[dict]:
    """Return the QA rows for ``video_key`` in timestamp order.

    ``video_key`` (the video's containing directory, e.g. ``sample_51``) must
    appear as a ``_<key>_<idx>`` suffix of the CSV ``question_id``. Zero matches
    is a hard error -- the QA correspondence is never guessed.
    """
    pattern = re.compile(rf"_{re.escape(video_key)}_(\d+)$")
    matched: list[tuple[int, dict]] = []
    for row in rows:
        m = pattern.search(row["question_id"])
        if m:
            matched.append((int(m.group(1)), row))
    if not matched:
        raise KeyError(
            f"no CSV rows match video identifier {video_key!r} "
            f"(looked for a '_{video_key}_<n>' suffix in question_id)"
        )
    matched.sort(key=lambda pair: pair[0])
    return [row for _, row in matched]


def build_schedule(rows: list[dict]) -> list[dict]:
    """Turn raw CSV rows into an ordered QA schedule (sorted by time_stamp)."""
    schedule = []
    for row in rows:
        schedule.append(
            {
                "question_id": row["question_id"],
                "task_type": row.get("task_type", ""),
                "question": row["question"],
                "time_stamp": parse_time_stamp(row["time_stamp"]),
                "answer": (row.get("answer") or "").strip(),
                "options": parse_options(row.get("options", "")),
                "frames_required": row.get("frames_required", ""),
                "temporal_clue_type": row.get("temporal_clue_type", ""),
            }
        )
    schedule.sort(key=lambda qa: qa["time_stamp"])
    return schedule


def build_prompt(qa: dict) -> str:
    """Build the model prompt from the question + options (no reference answer)."""
    lines = [qa["question"], "", "Options:"]
    lines.extend(qa["options"])
    lines.append("")
    lines.append("Answer with the letter of the correct option (A, B, C, or D).")
    return "\n".join(lines)


def extract_choice(text: str) -> str | None:
    """Extract a standalone ``A``-``D`` letter from a model answer, else None."""
    match = re.search(r"\b([A-D])\b", text or "")
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Interleaved streaming driver
# ---------------------------------------------------------------------------


def run_streaming(
    agent,
    frame_iter,
    schedule: list[dict],
    *,
    video_name: str,
    collector: MetricsCollector,
    total_duration: float,
) -> tuple[list[dict], float]:
    """Drive the interleaved streaming loop for one video.

    Returns ``(query_records, total_wall_time)``. ``total_wall_time`` is the
    wall-clock time from ``start_session`` to ``end_session``; the KV cache is
    cleared at both boundaries (one MP4 = one session).
    """
    session_start = time.perf_counter()
    agent.start_session()

    pending = list(schedule)
    active: dict | None = None
    video_done = False
    current_ts = 0.0
    query_records: list[dict] = []
    last_progress = -1.0

    def _iso(t: float) -> str:
        return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")

    def finish_query(qa: dict, output: str) -> None:
        wall_end = time.time()
        choice = extract_choice(output)
        arrival = qa["_arrival_wall"]
        start = qa["_start_wall"]
        record = {
            "video_name": video_name,
            "question_id": qa["question_id"],
            "task_type": qa["task_type"],
            "scheduled_timestamp": qa["time_stamp"],
            "actual_trigger_timestamp": qa["_trigger_ts"],
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "model_output": output,
            "extracted_choice": choice,
            "correct": (choice == qa["answer"]) if qa["answer"] else False,
            "query_arrival_wall_time": _iso(arrival),
            "query_start_wall_time": _iso(start),
            "query_end_wall_time": _iso(wall_end),
            "queue_wait_time": start - arrival,
            "generation_time": wall_end - start,
            "end_to_end_latency": wall_end - arrival,
        }
        query_records.append(record)
        collector.record_query(record)
        print(
            f"[Q{len(query_records)}/{len(schedule)}] answer={choice or '-'} | "
            f"correct={record['correct']} | E2E={record['end_to_end_latency']:.2f}s"
        )

    def skipped_record(qa: dict) -> dict:
        return {
            "video_name": video_name,
            "question_id": qa["question_id"],
            "task_type": qa["task_type"],
            "scheduled_timestamp": qa["time_stamp"],
            "actual_trigger_timestamp": None,
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "model_output": "",
            "extracted_choice": None,
            "correct": False,
            "query_arrival_wall_time": "",
            "query_start_wall_time": "",
            "query_end_wall_time": "",
            "queue_wait_time": 0.0,
            "generation_time": 0.0,
            "end_to_end_latency": 0.0,
        }

    try:
        while True:
            # 1. Advance the video timeline by one (sampled) frame.
            if not video_done:
                try:
                    idx, ts, frame = next(frame_iter)
                except StopIteration:
                    video_done = True
                    agent.flush_chunk()
                else:
                    agent.ingest_frame(frame, ts)
                    collector.note_frame(idx, ts)
                    current_ts = ts
                    if ts - last_progress >= 5.0:
                        last_progress = ts
                        if total_duration > 0:
                            pct = min(100.0, ts / total_duration * 100.0)
                            print(f"Progress: {pct:.0f}% | video={ts:.1f}s")
                        else:
                            print(f"Progress: video={ts:.1f}s")

            # 2. Trigger a QA the moment the timeline reaches its time_stamp.
            if active is None and pending and current_ts >= pending[0]["time_stamp"]:
                active = pending.pop(0)
                active["_trigger_ts"] = current_ts
                active["_arrival_wall"] = time.time()
                agent.start_query(build_prompt(active))
                active["_start_wall"] = time.time()

            # 3. Step generation, interleaved with further ingestion above.
            if active is not None:
                text, done = agent.step_generation()
                if done:
                    finish_query(active, text)
                    active = None

            # 4. Termination: video exhausted, no active query, nothing reachable.
            if video_done and active is None:
                if not pending:
                    break
                if pending[0]["time_stamp"] > current_ts:
                    break
    finally:
        agent.end_session()

    # Any QA whose timestamp the video never reached is recorded as skipped.
    for qa in pending:
        rec = skipped_record(qa)
        query_records.append(rec)
        collector.record_query(rec)

    total_wall_time = time.perf_counter() - session_start
    return query_records, total_wall_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_mp4_files(video_dir: Path) -> list[Path]:
    """Recursively find all ``.mp4`` files under ``video_dir``, in sorted order."""
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory not found: {video_dir}")
    return sorted(video_dir.rglob("*.mp4"))


def _iter_video_frames(video_path: Path, fps: float):
    """Fresh frame iterator for one video (a reader is bound to one file)."""
    from video.reader import VideoReader
    from video.sampler import FrameSampler

    reader = VideoReader(video_path=str(video_path))
    sampler = FrameSampler(fps=fps, max_frames=None)
    return sampler.stream(reader.read_frames())


def _video_key_from(video_path: Path) -> str:
    """Video identifier = the directory containing the mp4 (e.g. ``sample_51``)."""
    video_key = video_path.parent.name
    if not video_key or video_key in {".", "..", "video", "StreamingBench"}:
        raise ValueError(
            f"cannot derive a video identifier from {video_path!r}; "
            f"expected a path like .../video/<sample_id>/video.mp4"
        )
    return video_key


def _video_duration_seconds(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return (n / fps) if n > 0 else 0.0
    finally:
        cap.release()


def _load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"StreamingBench CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _resolve_config(args: argparse.Namespace) -> dict:
    config = load_config(_PROJECT_ROOT / args.config)
    if args.fps is not None:
        config.setdefault("video", {})["fps"] = float(args.fps)
    return config


def _build_schedule(rows: list[dict], video_key: str, limit: int | None) -> list[dict]:
    schedule = build_schedule(find_qa_for_video(rows, video_key))
    if limit is not None:
        schedule = schedule[:limit]
    return schedule


def _run_one_video(agent, video_path, video_key, schedule, fps, out_dir):
    """Run one video (its own session), save its metrics, return the summary."""
    collector = MetricsCollector(video_name=video_key, fps=fps)
    # Attach the collector to the native backend: the only place where actual
    # mtmd visual-token counts and per-append KV growth are observable.
    agent._backend._native.set_metrics_recorder(collector)

    total_duration = _video_duration_seconds(video_path)
    if total_duration <= 0:
        total_duration = max((q["time_stamp"] for q in schedule), default=0.0)

    query_records, total_wall_time = run_streaming(
        agent,
        _iter_video_frames(video_path, fps),
        schedule,
        video_name=video_key,
        collector=collector,
        total_duration=total_duration,
    )
    collector.fill_frame_meta()

    num_q = len(query_records)
    num_c = sum(1 for r in query_records if r["correct"])
    e2e = [r["end_to_end_latency"] for r in query_records]
    gen = [r["generation_time"] for r in query_records]
    queue = [r["queue_wait_time"] for r in query_records]
    video_summary = {
        "video_name": video_key,
        "video_duration": total_duration,
        "num_questions": num_q,
        "num_correct": num_c,
        "accuracy": (num_c / num_q) if num_q else 0.0,
        "total_wall_time": total_wall_time,
        "processing_speed_ratio": (total_duration / total_wall_time) if total_wall_time else 0.0,
        "rtf": (total_wall_time / total_duration) if total_duration else 0.0,
        "avg_end_to_end_latency": (sum(e2e) / len(e2e)) if e2e else 0.0,
        "avg_generation_time": (sum(gen) / len(gen)) if gen else 0.0,
        "avg_queue_wait_time": (sum(queue) / len(queue)) if queue else 0.0,
    }

    collector.save_video(out_dir / video_key, video_summary)
    return video_summary, query_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OURS_Agent streaming inference on StreamingBench video(s).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video",
        help="Path to a single video, e.g. data/StreamingBench/video/sample_51/video.mp4.",
    )
    source.add_argument(
        "--video-dir",
        help="Directory to scan recursively for .mp4 files and run them in order.",
    )
    parser.add_argument(
        "--csv",
        default="data/StreamingBench/annotations/Real_Time_Visual_Understanding.csv",
        help="Path to the StreamingBench QA CSV (default relative to project root).",
    )
    parser.add_argument(
        "--config",
        default="configs/gemma4.yaml",
        help="Path to the YAML configuration (default: configs/gemma4.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/streamingbench",
        help="Root output directory (default: results/streamingbench).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video.fps (frames ingested per second of video).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Answer only the first N scheduled questions (smoke test).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Parse the CSV and print the QA schedule, then exit (no model/video).",
    )
    return parser.parse_args(argv)


def _print_schedule(video_path: Path, video_key: str, schedule: list[dict]) -> None:
    print(f"video: {video_path}")
    print(f"video identifier: {video_key}")
    print(f"questions: {len(schedule)}")
    for i, qa in enumerate(schedule, start=1):
        print(f"  [{i}] {qa['question_id']}")
        print(f"      time_stamp={qa['time_stamp']:.2f}s  answer={qa['answer']!r}  "
              f"clue={qa['temporal_clue_type']}  frames={qa['frames_required']}")
        print(f"      Q: {qa['question']}")
        if qa["options"]:
            print(f"      options: {qa['options']}")


def _run_single(args: argparse.Namespace, rows: list[dict]) -> int:
    video_path = Path(args.video)
    try:
        video_key = _video_key_from(video_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    schedule = _build_schedule(rows, video_key, args.limit)
    if args.validate:
        _print_schedule(video_path, video_key, schedule)
        return 0

    config = _resolve_config(args)
    fps = float(config.get("video", {}).get("fps", 2.0))
    out_dir = _PROJECT_ROOT / args.out_dir

    from scripts.run_video import build_agent

    agent = build_agent(config, str(video_path), max_frames=None)

    print(f"[1/1] {video_key}.mp4")
    video_summary, query_records = _run_one_video(
        agent, video_path, video_key, schedule, fps, out_dir
    )
    save_video_summary([video_summary], out_dir)
    save_query_summary(
        [{k: r.get(k) for k in QUERY_SUMMARY_FIELDS} for r in query_records], out_dir
    )
    print(
        f"[Video Done] {video_key} | accuracy={video_summary['num_correct']}/"
        f"{video_summary['num_questions']} ({video_summary['accuracy'] * 100:.1f}%)"
    )
    print(f"[Saved] {out_dir / video_key}/")
    return 0


def _run_batch(args: argparse.Namespace, rows: list[dict]) -> int:
    video_paths = find_mp4_files(Path(args.video_dir))
    if not video_paths:
        print(f"no .mp4 files found under {args.video_dir}")
        return 2

    if args.validate:
        for video_path in video_paths:
            try:
                video_key = _video_key_from(video_path)
                _print_schedule(
                    video_path, video_key, _build_schedule(rows, video_key, args.limit)
                )
            except Exception as exc:
                print(f"{video_path.parent.name}: error: {exc}")
        return 0

    config = _resolve_config(args)
    fps = float(config.get("video", {}).get("fps", 2.0))
    out_dir = _PROJECT_ROOT / args.out_dir

    from scripts.run_video import build_agent

    # One agent / one model load for the whole batch. Each video runs its own
    # start_session/end_session inside run_streaming, clearing the KV at both
    # boundaries, so the KV cache is never reused across videos.
    agent = build_agent(config, str(video_paths[0]), max_frames=None)

    video_summaries: list[dict] = []
    all_query_summary_rows: list[dict] = []
    for i, video_path in enumerate(video_paths, start=1):
        stem = video_path.parent.name
        print(f"[{i}/{len(video_paths)}] {stem}.mp4")
        try:
            video_key = _video_key_from(video_path)
            schedule = _build_schedule(rows, video_key, args.limit)
        except Exception as exc:
            print(f"[Video Failed] {stem}: {exc}")
            continue
        try:
            video_summary, query_records = _run_one_video(
                agent, video_path, video_key, schedule, fps, out_dir
            )
        except Exception as exc:
            log.exception("video failed: %s", video_path)
            print(f"[Video Failed] {stem}: {exc}")
            continue
        video_summaries.append(video_summary)
        all_query_summary_rows.extend(
            {k: r.get(k) for k in QUERY_SUMMARY_FIELDS} for r in query_records
        )
        print(
            f"[Video Done] {video_key} | accuracy={video_summary['num_correct']}/"
            f"{video_summary['num_questions']} ({video_summary['accuracy'] * 100:.1f}%)"
        )
        print(f"[Saved] {out_dir / video_key}/")

    if video_summaries:
        save_video_summary(video_summaries, out_dir)
        save_query_summary(all_query_summary_rows, out_dir)
        total_q = sum(s["num_questions"] for s in video_summaries)
        total_c = sum(s["num_correct"] for s in video_summaries)
        acc = (total_c / total_q * 100.0) if total_q else 0.0
        print(
            f"[Batch Done] {len(video_summaries)}/{len(video_paths)} videos | "
            f"accuracy={total_c}/{total_q} ({acc:.1f}%)"
        )
        print(f"[Summary] {out_dir / 'summary'}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    rows = _load_csv_rows(_PROJECT_ROOT / args.csv)
    if args.video_dir:
        return _run_batch(args, rows)
    return _run_single(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
