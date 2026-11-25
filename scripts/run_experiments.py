"""Main CLI entry point for running OpenBMI SSVEP & P300 cross-day experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_full_benchmark
from src.utils import load_config, set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenBMI (Lee2019) SSVEP & P300 Cross-Day Benchmark.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="quick=5 subjects, full=54 subjects")
    parser.add_argument("--subjects", type=int, nargs="*", default=None, help="Explicit subject IDs (1..54)")
    parser.add_argument("--paradigm", choices=["both", "ssvep", "p300"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve relative paths against project root
    for key in ("data_dir", "results_dir"):
        p = Path(cfg["project"][key])
        if not p.is_absolute():
            cfg["project"][key] = str(PROJECT_ROOT / p)

    set_seed(int(cfg["project"]["seed"]))
    log_path = Path(cfg["project"]["results_dir"]) / "run.log"
    logger = setup_logger("openbmi_bci", log_file=log_path)

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = cfg["dataset"]["quick_subjects"] if args.mode == "quick" else cfg["dataset"]["full_subjects"]

    logger.info("Mode: %s | Paradigm: %s | Subjects (%d): %s", args.mode, args.paradigm, len(subjects), subjects)
    run_full_benchmark(
        subjects=subjects,
        cfg=cfg,
        run_ssvep=args.paradigm in ("both", "ssvep"),
        run_p300=args.paradigm in ("both", "p300"),
    )


if __name__ == "__main__":
    main()
