"""Dual-paradigm OpenBMI (Lee2019) data pipeline using MOABB and MNE-Python.

Handles:
1. Persistent MNE/MOABB cache directory setup (compatible with Google Drive on Colab).
2. Loading Lee2019_SSVEP (4-class occipital EEG) and Lee2019_ERP (P300 parieto-occipital EEG).
3. Preprocessing: powerline notch filtering, bandpass filtering, resampling,
   baseline correction, and event-aligned epoching.
4. Strict Day 1 / Day 2 session separation for cross-day generalization evaluation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("openbmi_bci")


@dataclass
class SessionEpochData:
    """Container for epoched EEG data of a single subject and paradigm."""

    subject: int
    paradigm: str
    X_day1: np.ndarray  # shape: (n_trials_d1, n_channels, n_times)
    y_day1: np.ndarray  # shape: (n_trials_d1,)
    X_day2: np.ndarray  # shape: (n_trials_d2, n_channels, n_times)
    y_day2: np.ndarray  # shape: (n_trials_d2,)
    ch_names: List[str]
    sfreq: float
    tmin: float
    class_names: List[str]


def configure_mne_data_dir(data_dir: str | Path) -> Path:
    """Point MNE and MOABB download cache to a persistent local or Drive folder."""
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    for key in ("MNE_DATA", "MNE_DATASETS_LEE2019_SSVEP_PATH", "MNE_DATASETS_LEE2019_ERP_PATH"):
        os.environ[key] = str(data_path)
    try:
        import mne

        for key in ("MNE_DATA", "MNE_DATASETS_LEE2019_SSVEP_PATH", "MNE_DATASETS_LEE2019_ERP_PATH"):
            mne.set_config(key, str(data_path), set_env=True)
    except Exception as exc:
        logger.warning("Could not persist MNE_DATA via mne.set_config: %s", exc)
    return data_path


def _resolve_session_keys(subject_dict: Dict[str, dict]) -> Tuple[str, str]:
    """Resolve Day 1 and Day 2 session keys robustly across MOABB versions.

    MOABB <1.0 uses 'session_1' / 'session_2'; MOABB >=1.0 uses '0' / '1'.
    """
    keys = sorted(subject_dict.keys())
    if len(keys) < 2:
        raise ValueError(f"Expected 2 sessions (Day 1 & Day 2), found: {keys}")
    if "session_1" in keys and "session_2" in keys:
        return "session_1", "session_2"
    if "0" in keys and "1" in keys:
        return "0", "1"
    return keys[0], keys[1]


def _select_channels(raw, target_channels: Sequence[str]) -> List[str]:
    """Pick available channels matching target_channels while preserving order."""
    available = set(raw.ch_names)
    selected = [ch for ch in target_channels if ch in available]
    if not selected:
        raise ValueError(
            f"None of requested channels {list(target_channels)} found in raw channels {raw.ch_names[:15]}..."
        )
    return selected


def _preprocess_raw_and_epoch(
    raw,
    target_channels: Sequence[str],
    event_id_map: Dict[str, int],
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    sfreq: float,
    tmin: float,
    tmax: float,
    baseline: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Preprocess a single MNE Raw run and extract baseline-corrected epochs."""
    import mne

    raw = raw.copy().load_data()

    # Locate stimulus channel before picking EEG channels
    stim_picks = mne.pick_types(raw.info, stim=True)
    stim_names = [raw.ch_names[i] for i in stim_picks]

    # Extract events from stim channel or annotations
    if stim_names:
        events = mne.find_events(raw, stim_channel=stim_names[0], shortest_event=1, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, verbose=False)

    # Pick target EEG channels
    selected_chs = _select_channels(raw, target_channels)
    raw.pick(selected_chs)

    # Powerline notch filter (60 Hz in Korea OpenBMI) + harmonic if below Nyquist
    nyq = raw.info["sfreq"] / 2.0
    notch_freqs = [f for f in [notch_freq, notch_freq * 2.0] if f < nyq]
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, method="fir", verbose=False)

    # Bandpass filter (zero-phase Butterworth IIR)
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)

    # Filter event_id_map to events actually present in this run
    present_codes = set(np.unique(events[:, 2]).tolist())
    valid_event_id = {k: v for k, v in event_id_map.items() if v in present_codes}
    if not valid_event_id:
        raise RuntimeError(
            f"No matching events found. Expected codes {event_id_map}, present codes {sorted(present_codes)}"
        )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=valid_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    if abs(epochs.info["sfreq"] - sfreq) > 1e-3:
        epochs.resample(sfreq, npad="auto", verbose=False)

    X = epochs.get_data(copy=True).astype(np.float64)
    y = epochs.events[:, 2].astype(np.int64)
    return X, y, selected_chs


def _concat_session_runs(
    session_runs: Dict[str, object],
    target_channels: Sequence[str],
    event_id_map: Dict[str, int],
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    sfreq: float,
    tmin: float,
    tmax: float,
    baseline: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Preprocess and concatenate all runs within a single session."""
    X_list, y_list = [], []
    ch_names: List[str] = []
    for run_key in sorted(session_runs.keys()):
        raw = session_runs[run_key]
        X_run, y_run, ch_names = _preprocess_raw_and_epoch(
            raw=raw,
            target_channels=target_channels,
            event_id_map=event_id_map,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
        )
        X_list.append(X_run)
        y_list.append(y_run)

    # Align time axis length across runs if resampling caused +/- 1 sample difference
    min_len = min(x.shape[-1] for x in X_list)
    X_cat = np.concatenate([x[:, :, :min_len] for x in X_list], axis=0)
    y_cat = np.concatenate(y_list, axis=0)
    return X_cat, y_cat, ch_names


def load_openbmi_ssvep_subject(subject: int, cfg: Dict) -> SessionEpochData:
    """Load and preprocess Day 1 & Day 2 SSVEP data for one OpenBMI subject."""
    from moabb.datasets import Lee2019_SSVEP

    configure_mne_data_dir(cfg["project"]["data_dir"])
    dataset = Lee2019_SSVEP(sessions=(1, 2))
    raw_data = dataset.get_data(subjects=[subject])[subject]
    day1_key, day2_key = _resolve_session_keys(raw_data)

    # Map MOABB dataset.event_id keys to ordered frequency indices 0..3
    freqs = [float(f) for f in cfg["ssvep"]["frequencies"]]
    # dataset.event_id maps e.g. {'5.45': 4, '6.67': 3, '8.57': 2, '12.0': 1}
    code_to_class_idx: Dict[int, int] = {}
    event_id_map: Dict[str, int] = {}
    for ev_name, ev_code in dataset.event_id.items():
        try:
            ev_freq = float(ev_name)
        except ValueError:
            continue
        # Match nearest target frequency
        best_idx = int(np.argmin([abs(ev_freq - f) for f in freqs]))
        code_to_class_idx[int(ev_code)] = best_idx
        event_id_map[ev_name] = int(ev_code)

    scfg = cfg["ssvep"]
    notch = float(cfg["dataset"]["notch_freq"])
    baseline = (float(scfg["baseline"][0]), float(scfg["baseline"][1]))

    X_d1, y_d1_raw, ch_names = _concat_session_runs(
        raw_data[day1_key],
        target_channels=scfg["channels"],
        event_id_map=event_id_map,
        l_freq=float(scfg["l_freq"]),
        h_freq=float(scfg["h_freq"]),
        notch_freq=notch,
        sfreq=float(scfg["sfreq"]),
        tmin=float(scfg["tmin"]),
        tmax=float(scfg["tmax"]),
        baseline=baseline,
    )
    X_d2, y_d2_raw, _ = _concat_session_runs(
        raw_data[day2_key],
        target_channels=scfg["channels"],
        event_id_map=event_id_map,
        l_freq=float(scfg["l_freq"]),
        h_freq=float(scfg["h_freq"]),
        notch_freq=notch,
        sfreq=float(scfg["sfreq"]),
        tmin=float(scfg["tmin"]),
        tmax=float(scfg["tmax"]),
        baseline=baseline,
    )

    y_d1 = np.array([code_to_class_idx[int(c)] for c in y_d1_raw], dtype=np.int64)
    y_d2 = np.array([code_to_class_idx[int(c)] for c in y_d2_raw], dtype=np.int64)

    return SessionEpochData(
        subject=subject,
        paradigm="SSVEP",
        X_day1=X_d1,
        y_day1=y_d1,
        X_day2=X_d2,
        y_day2=y_d2,
        ch_names=ch_names,
        sfreq=float(scfg["sfreq"]),
        tmin=float(scfg["tmin"]),
        class_names=[f"{f:.2f}Hz" for f in freqs],
    )


def load_openbmi_p300_subject(subject: int, cfg: Dict) -> SessionEpochData:
    """Load and preprocess Day 1 & Day 2 P300 ERP data for one OpenBMI subject."""
    from moabb.datasets import Lee2019_ERP

    configure_mne_data_dir(cfg["project"]["data_dir"])
    dataset = Lee2019_ERP(sessions=(1, 2))
    raw_data = dataset.get_data(subjects=[subject])[subject]
    day1_key, day2_key = _resolve_session_keys(raw_data)

    # dataset.event_id is typically {'Target': 1, 'NonTarget': 2}
    event_id_map = {k: int(v) for k, v in dataset.event_id.items()}
    target_code = int(event_id_map["Target"])

    pcfg = cfg["p300"]
    notch = float(cfg["dataset"]["notch_freq"])
    baseline = (float(pcfg["baseline"][0]), float(pcfg["baseline"][1]))

    X_d1, y_d1_raw, ch_names = _concat_session_runs(
        raw_data[day1_key],
        target_channels=pcfg["channels"],
        event_id_map=event_id_map,
        l_freq=float(pcfg["l_freq"]),
        h_freq=float(pcfg["h_freq"]),
        notch_freq=notch,
        sfreq=float(pcfg["sfreq"]),
        tmin=float(pcfg["tmin"]),
        tmax=float(pcfg["tmax"]),
        baseline=baseline,
    )
    X_d2, y_d2_raw, _ = _concat_session_runs(
        raw_data[day2_key],
        target_channels=pcfg["channels"],
        event_id_map=event_id_map,
        l_freq=float(pcfg["l_freq"]),
        h_freq=float(pcfg["h_freq"]),
        notch_freq=notch,
        sfreq=float(pcfg["sfreq"]),
        tmin=float(pcfg["tmin"]),
        tmax=float(pcfg["tmax"]),
        baseline=baseline,
    )

    # Map to binary labels: 1 = Target, 0 = NonTarget
    y_d1 = (y_d1_raw == target_code).astype(np.int64)
    y_d2 = (y_d2_raw == target_code).astype(np.int64)

    return SessionEpochData(
        subject=subject,
        paradigm="P300",
        X_day1=X_d1,
        y_day1=y_d1,
        X_day2=X_d2,
        y_day2=y_d2,
        ch_names=ch_names,
        sfreq=float(pcfg["sfreq"]),
        tmin=float(pcfg["tmin"]),
        class_names=["NonTarget", "Target"],
    )


def slice_ssvep_window(
    X: np.ndarray,
    sfreq: float,
    tmin: float,
    window_sec: float,
    visual_delay: float = 0.14,
) -> np.ndarray:
    """Slice [visual_delay, visual_delay + window_sec] from SSVEP epochs."""
    start_idx = int(round((visual_delay - tmin) * sfreq))
    n_samples = int(round(window_sec * sfreq))
    end_idx = start_idx + n_samples
    if start_idx < 0 or end_idx > X.shape[-1]:
        raise ValueError(
            f"Requested window [{start_idx}:{end_idx}] out of bounds for epoch length {X.shape[-1]}."
        )
    return X[:, :, start_idx:end_idx]
