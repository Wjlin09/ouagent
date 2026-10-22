"""Entry point: run the streaming video-understanding agent on a video file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/run_video.py` from anywhere by making the
# project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.agent import Agent
from agent.controller import Controller
from agent.state import AgentState
from algorithms.frame_selector import AutoGazeFrameSelector
from algorithms.semantic_pruning import SemanticPruner
from evaluation.latency import LatencyTracker
from runtime.gemma_backend import GemmaBackend
from video.reader import VideoReader
from video.sampler import FrameSampler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OURS_Agent streaming video-understanding agent.",
    )
    parser.add_argument(
        "--config",
        default="configs/gemma4.yaml",
        help="Path to the YAML configuration file (default: configs/gemma4.yaml).",
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Sentinel distinguishing "use the config's ``video.max_frames``" (the default)
# from "explicitly uncapped" (``max_frames=None``) for streaming/interactive
# callers. ``FrameSampler`` treats ``None`` as "no frame cap".
_UNSET_MAX_FRAMES = object()


def build_agent(
    config: dict, video_path: str | Path, max_frames: int | None = _UNSET_MAX_FRAMES
) -> Agent:
    model_cfg = config.get("model", {})
    video_cfg = config.get("video", {})
    algo_cfg = config.get("algorithms", {})
    agent_cfg = config.get("agent", {})
    runtime_cfg = config.get("runtime", {})
    streaming_cfg = config.get("streaming", {})

    # Optional input-frame resize (resolution -> visual tokens/frame study).
    # ``None`` = disabled (baseline, original resolution passes through).
    resize_cfg = video_cfg.get("resize", {})
    resize_max_size = None
    if resize_cfg.get("enabled", False):
        _ms = resize_cfg.get("max_size")
        if _ms is not None and int(_ms) > 0:
            resize_max_size = int(_ms)

    n_threads = model_cfg.get("n_threads")
    backend = GemmaBackend(
        model_path=model_cfg["path"],
        mmproj_path=model_cfg["mmproj"],
        n_ctx=int(model_cfg.get("n_ctx", 8192)),
        n_batch=int(model_cfg.get("n_batch", 2048)),
        n_ubatch=int(model_cfg.get("n_ubatch", 512)),
        n_gpu_layers=int(model_cfg.get("n_gpu_layers", -1)),
        n_threads=int(n_threads) if n_threads is not None else None,
        mmproj_use_gpu=bool(model_cfg.get("mmproj_use_gpu", True)),
        temperature=float(model_cfg.get("temperature", 0.2)),
        top_p=float(model_cfg.get("top_p", 0.9)),
        max_tokens=int(model_cfg.get("max_tokens", 512)),
        seed=int(model_cfg.get("seed", 1234)),
        repeat_penalty=float(model_cfg.get("repeat_penalty", 1.1)),
        libllama_path=runtime_cfg.get("libllama_path"),
        libmtmd_path=runtime_cfg.get("libmtmd_path"),
        lib_dir=runtime_cfg.get("lib_dir"),
        resize_max_size=resize_max_size,
    )

    reader = VideoReader(video_path=video_path)
    if max_frames is _UNSET_MAX_FRAMES:
        max_frames = int(video_cfg.get("max_frames", 16))
    sampler = FrameSampler(
        fps=float(video_cfg.get("fps", 2.0)),
        max_frames=max_frames,
    )
    state = AgentState()
    controller = Controller()

    # AutoGaze (frame-level input filtering) — independent of semantic pruning.
    frame_selector = None
    ag_cfg = algo_cfg.get("autogaze", {})
    if ag_cfg.get("enabled", False):
        frame_selector = AutoGazeFrameSelector(
            model_id=ag_cfg.get("model_id", "nvidia/AutoGaze"),
            gazing_ratio=float(ag_cfg.get("gazing_ratio", 0.75)),
            task_loss_requirement=float(ag_cfg.get("task_loss_requirement", 0.7)),
            device=ag_cfg.get("device"),
        )

    # Query-aware semantic pruning — independent of AutoGaze. There is no
    # concrete scorer yet, so enabling it constructs the unimplemented
    # interface (which fails loudly on first use) rather than a fake no-op.
    semantic_pruner = None
    sp_cfg = algo_cfg.get("semantic_pruning", {})
    if sp_cfg.get("enabled", False):
        semantic_pruner = SemanticPruner()

    latency_tracker = (
        LatencyTracker() if config.get("latency", {}).get("enabled", True) else None
    )

    prompt_template = agent_cfg.get(
        "prompt_template", "What is happening in this image?"
    )

    num_frameproc = 1
    if streaming_cfg.get("enabled", True):
        num_frameproc = max(1, int(streaming_cfg.get("num_frameproc", 1)))

    return Agent(
        backend=backend,
        reader=reader,
        sampler=sampler,
        state=state,
        controller=controller,
        frame_selector=frame_selector,
        semantic_pruner=semantic_pruner,
        latency_tracker=latency_tracker,
        prompt_template=prompt_template,
        num_frameproc=num_frameproc,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    agent = build_agent(config, args.video)
    agent.run()

    if agent.latency_tracker is not None:
        print("Latency summary:", agent.latency_tracker.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
