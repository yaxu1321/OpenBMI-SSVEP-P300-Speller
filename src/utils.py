"""Utility functions: reproducibility, logging, ITR calculation, and I/O."""

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "openbmi_bci", log_file: str | Path | None = None) -> logging.Logger:
    """Configure console + optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_itr(n_classes: int, accuracy: float, time_sec: float) -> float:
    """Compute Wolpaw Information Transfer Rate (ITR) in bits/min.

    Args:
        n_classes: Number of stimulus classes (N >= 2).
        accuracy: Classification accuracy P in [0.0, 1.0].
        time_sec: Total selection time per command in seconds (T + T_gaze).

    Returns:
        ITR in bits per minute.
    """
    if n_classes < 2 or time_sec <= 0:
        return 0.0
    p = float(np.clip(accuracy, 0.0, 1.0))
    if p <= 1.0 / n_classes:
        return 0.0
    if p >= 1.0 - 1e-12:
        bits_per_trial = math.log2(n_classes)
    else:
        bits_per_trial = (
            math.log2(n_classes)
            + p * math.log2(p)
            + (1.0 - p) * math.log2((1.0 - p) / (n_classes - 1.0))
        )
    return max(0.0, bits_per_trial * (60.0 / time_sec))


def save_json(data: Any, path: str | Path) -> None:
    """Save dictionary/list to UTF-8 JSON file with clean formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)
