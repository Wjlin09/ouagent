"""Entry point: analyze a single image with the Gemma 4 backend."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/run_image.py` from anywhere by making the
# project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime.gemma_backend import GemmaBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a single image with the OURS_Agent Gemma 4 backend.",
    )
    parser.add_argument(
        "--config",
        default="configs/gemma4.yaml",
        help="Path to the YAML configuration file (default: configs/gemma4.yaml).",
    )
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt. Defaults to `agent.prompt_template` from the config.",
    )
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_backend(config: dict) -> GemmaBackend:
    """Build a ``GemmaBackend`` from the same ``model`` keys as run_video.py."""
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})
    n_threads = model_cfg.get("n_threads")
    return GemmaBackend(
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
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    prompt = args.prompt or config.get("agent", {}).get(
        "prompt_template", "What is happening in this image?"
    )

    backend = build_backend(config)
    output = backend.infer(args.image, prompt)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
