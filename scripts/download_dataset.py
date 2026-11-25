"""Download and cache OpenBMI (Lee2019_SSVEP & Lee2019_ERP) datasets via MOABB.

All files are stored inside `data/mne_data` (or the configured Drive path on Colab),
so subsequent runs reuse the cached files directly without re-downloading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import configure_mne_data_dir
from src.utils import load_config, save_json, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenBMI Lee2019 SSVEP & P300 datasets.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="quick=5 subjects, full=54 subjects")
    parser.add_argument("--subjects", type=int, nargs="*", default=None, help="Explicit subject IDs (1..54)")
    parser.add_argument("--paradigm", choices=["both", "ssvep", "p300"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("openbmi_download")

    data_dir = Path(cfg["project"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    resolved_dir = configure_mne_data_dir(data_dir)

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = cfg["dataset"]["quick_subjects"] if args.mode == "quick" else cfg["dataset"]["full_subjects"]

    logger.info("MNE/MOABB cache directory: %s", resolved_dir)
    logger.info("Downloading OpenBMI (Lee2019) for %d subjects: %s", len(subjects), subjects)

    from moabb.datasets import Lee2019_ERP, Lee2019_SSVEP

    manifest = {
        "data_dir": str(resolved_dir),
        "subjects_requested": list(subjects),
        "ssvep_downloaded": [],
        "p300_downloaded": [],
    }

    if args.paradigm in ("both", "ssvep"):
        ds_ssvep = Lee2019_SSVEP(sessions=(1, 2))
        for subj in subjects:
            logger.info("[Download SSVEP] Subject %02d (Day 1 & Day 2)...", subj)
            ds_ssvep.download(subject_list=[subj])
            manifest["ssvep_downloaded"].append(subj)

    if args.paradigm in ("both", "p300"):
        ds_p300 = Lee2019_ERP(sessions=(1, 2))
        for subj in subjects:
            logger.info("[Download P300] Subject %02d (Day 1 & Day 2)...", subj)
            ds_p300.download(subject_list=[subj])
            manifest["p300_downloaded"].append(subj)

    save_json(manifest, resolved_dir.parent / "dataset_manifest.json")
    logger.info("Dataset download completed. Manifest saved to %s", resolved_dir.parent / "dataset_manifest.json")


if __name__ == "__main__":
    main()
