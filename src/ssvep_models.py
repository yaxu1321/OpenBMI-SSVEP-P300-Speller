"""SSVEP spatial-spectral decoding models: Chebyshev I Filter Bank, CCA, FBCCA, eTRCA.

References:
- Lin et al. (2006), Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs.
- Chen et al. (2015), Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based BCI.
- Nakanishi et al. (2018), Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh, qr, svd
from scipy.signal import cheby1, sosfiltfilt, welch


class ChebyshevFilterBank:
    """Chebyshev Type I Filter Bank for capturing SSVEP fundamental & harmonic components."""

    def __init__(
        self,
        sfreq: float,
        subband_lows: Sequence[float] = (4.5, 9.0, 13.5, 18.0, 22.5),
        subband_high: float = 45.0,
        order: int = 4,
        rp: float = 1.0,
        weight_a: float = 1.25,
        weight_b: float = 0.25,
    ) -> None:
        self.sfreq = float(sfreq)
        self.subband_lows = [float(f) for f in subband_lows]
        self.subband_high = float(subband_high)
        self.order = int(order)
        self.rp = float(rp)
        self.n_subbands = len(self.subband_lows)

        # Sub-band weights w(m) = m^(-a) + b (Chen et al., 2015)
        m_idx = np.arange(1, self.n_subbands + 1, dtype=np.float64)
        self.weights = np.power(m_idx, -weight_a) + weight_b

        nyq = self.sfreq / 2.0
        self.sos_list = []
        for low in self.subband_lows:
            wn_low = max(low / nyq, 1e-4)
            wn_high = min(self.subband_high / nyq, 0.99)
            if wn_low >= wn_high:
                raise ValueError(f"Invalid sub-band [{low}, {self.subband_high}] Hz for sfreq={self.sfreq}")
            sos = cheby1(N=self.order, rp=self.rp, Wn=[wn_low, wn_high], btype="bandpass", output="sos")
            self.sos_list.append(sos)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply filter bank to EEG epochs.

        Args:
            X: shape (n_trials, n_channels, n_times)

        Returns:
            X_fb: shape (n_subbands, n_trials, n_channels, n_times)
        """
        out = np.empty((self.n_subbands,) + X.shape, dtype=np.float64)
        for m, sos in enumerate(self.sos_list):
            out[m] = sosfiltfilt(sos, X, axis=-1)
        return out


def _generate_sinusoidal_reference(freq: float, n_samples: int, sfreq: float, n_harmonics: int) -> np.ndarray:
    """Generate sinusoidal reference matrix Y of shape (n_samples, 2 * n_harmonics)."""
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    cols = []
    for h in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * h * freq * t
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def _max_canonical_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute maximum canonical correlation between X (T, Cx) and Y (T, Cy) via QR + SVD."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Qx, _ = qr(Xc, mode="economic")
    Qy, _ = qr(Yc, mode="economic")
    s = svd(Qx.T @ Qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0)) if s.size > 0 else 0.0


class CCADecoder:
    """Training-free Canonical Correlation Analysis (CCA) SSVEP decoder."""

    def __init__(self, frequencies: Sequence[float], sfreq: float, n_harmonics: int = 5) -> None:
        self.frequencies = [float(f) for f in frequencies]
        self.sfreq = float(sfreq)
        self.n_harmonics = int(n_harmonics)
        self._ref_cache: Dict[int, List[np.ndarray]] = {}

    def _get_references(self, n_samples: int) -> List[np.ndarray]:
        if n_samples not in self._ref_cache:
            self._ref_cache[n_samples] = [
                _generate_sinusoidal_reference(f, n_samples, self.sfreq, self.n_harmonics)
                for f in self.frequencies
            ]
        return self._ref_cache[n_samples]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CCADecoder":
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return canonical correlation scores of shape (n_trials, n_classes)."""
        n_trials, _, n_samples = X.shape
        refs = self._get_references(n_samples)
        scores = np.zeros((n_trials, len(self.frequencies)), dtype=np.float64)
        for i in range(n_trials):
            xi = X[i].T  # (n_samples, n_channels)
            for k, Yk in enumerate(refs):
                scores[i, k] = _max_canonical_corr(xi, Yk)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)


class FBCCADecoder:
    """Training-free Filter Bank CCA (FBCCA) SSVEP decoder using Chebyshev Type I Filter Bank."""

    def __init__(
        self,
        frequencies: Sequence[float],
        sfreq: float,
        filter_bank: ChebyshevFilterBank,
        n_harmonics: int = 5,
    ) -> None:
        self.frequencies = [float(f) for f in frequencies]
        self.sfreq = float(sfreq)
        self.filter_bank = filter_bank
        self.cca = CCADecoder(frequencies=frequencies, sfreq=sfreq, n_harmonics=n_harmonics)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCCADecoder":
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return weighted sub-band squared correlation scores of shape (n_trials, n_classes)."""
        X_fb = self.filter_bank.transform(X)  # (n_subbands, n_trials, C, T)
        n_trials = X.shape[0]
        total_scores = np.zeros((n_trials, len(self.frequencies)), dtype=np.float64)
        for m in range(self.filter_bank.n_subbands):
            rho_m = self.cca.predict_scores(X_fb[m])
            total_scores += self.filter_bank.weights[m] * (rho_m ** 2)
        return total_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)


class eTRCADecoder:
    """Supervised Ensemble Task-Related Component Analysis (eTRCA) with Filter Bank.

    Trained on Day 1 calibration trials and evaluated on Day 2 unseen trials.
    """

    def __init__(
        self,
        n_classes: int,
        filter_bank: ChebyshevFilterBank,
        regularization: float = 0.01,
    ) -> None:
        self.n_classes = int(n_classes)
        self.filter_bank = filter_bank
        self.regularization = float(regularization)
        self.W_: List[np.ndarray] = []  # list of (C, n_classes) per subband
        self.templates_: List[np.ndarray] = []  # list of (n_classes, C, T) per subband

    def _compute_trca_filter(self, X_cls: np.ndarray) -> np.ndarray:
        """Solve generalized eigenvalue problem S w = lambda Q w for one class.

        Args:
            X_cls: shape (n_trials_k, C, T)
        Returns:
            w: unit-norm spatial filter of shape (C,)
        """
        n_trials, n_ch, _ = X_cls.shape
        # Center each trial along time axis
        Xc = X_cls - X_cls.mean(axis=-1, keepdims=True)

        # Inter-trial covariance S = sum_{i != j} X_i X_j^T = (sum X_i)(sum X_i)^T - sum (X_i X_i^T)
        X_sum = Xc.sum(axis=0)  # (C, T)
        S_total = X_sum @ X_sum.T
        Q = np.einsum("nct,ndt->cd", Xc, Xc)
        S = S_total - Q

        # Symmetrize and apply mild shrinkage regularization to Q for cross-day stability
        S = 0.5 * (S + S.T)
        Q = 0.5 * (Q + Q.T)
        trace_scale = np.trace(Q) / max(n_ch, 1)
        Q_reg = Q + self.regularization * trace_scale * np.eye(n_ch, dtype=np.float64)

        eigvals, eigvecs = eigh(S, Q_reg)
        w = eigvecs[:, -1]
        norm = np.linalg.norm(w)
        return w / norm if norm > 1e-12 else w

    def fit(self, X: np.ndarray, y: np.ndarray) -> "eTRCADecoder":
        """Fit per-subband templates and ensemble spatial filters on Day 1 training data."""
        X_fb = self.filter_bank.transform(X)  # (M, N, C, T)
        n_ch, n_times = X.shape[1], X.shape[2]
        self.W_ = []
        self.templates_ = []

        for m in range(self.filter_bank.n_subbands):
            Xm = X_fb[m]
            Wm = np.zeros((n_ch, self.n_classes), dtype=np.float64)
            Tm = np.zeros((self.n_classes, n_ch, n_times), dtype=np.float64)
            for k in range(self.n_classes):
                Xk = Xm[y == k]
                if Xk.shape[0] < 2:
                    raise ValueError(f"eTRCA requires >= 2 training trials for class {k}, got {Xk.shape[0]}")
                Tm[k] = Xk.mean(axis=0)
                Wm[:, k] = self._compute_trca_filter(Xk)
            self.W_.append(Wm)
            self.templates_.append(Tm)
        return self

    @staticmethod
    def _pearson_corr_2d(A: np.ndarray, B: np.ndarray) -> float:
        """Flatten 2D projected matrices (K, T) and compute Pearson correlation."""
        a = A.ravel()
        b = B.ravel()
        a = a - a.mean()
        b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Score test trials via ensemble spatial projection correlation across sub-bands."""
        X_fb = self.filter_bank.transform(X)
        n_trials = X.shape[0]
        total_scores = np.zeros((n_trials, self.n_classes), dtype=np.float64)

        for m in range(self.filter_bank.n_subbands):
            Xm = X_fb[m]
            Wm = self.W_[m]  # (C, K)
            Tm = self.templates_[m]  # (K, C, T)
            # Pre-project templates: proj_T[k] has shape (K, T)
            proj_templates = [Wm.T @ Tm[k] for k in range(self.n_classes)]
            weight_m = self.filter_bank.weights[m]

            for i in range(n_trials):
                proj_x = Wm.T @ Xm[i]  # (K, T)
                for k in range(self.n_classes):
                    r = self._pearson_corr_2d(proj_x, proj_templates[k])
                    total_scores[i, k] += weight_m * r
        return total_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)


def extract_ssvep_harmonic_stats(
    X: np.ndarray,
    y: np.ndarray,
    frequencies: Sequence[float],
    sfreq: float,
    ch_names: Sequence[str],
) -> List[Dict[str, float | str]]:
    """Compute numerical PSD & SNR at fundamental (1f) and 2nd/3rd harmonics on Oz/occipital average."""
    oz_idx = ch_names.index("Oz") if "Oz" in ch_names else 0
    rows: List[Dict[str, float | str]] = []

    for k, freq in enumerate(frequencies):
        Xk = X[y == k, oz_idx, :]  # (n_trials_k, T)
        if Xk.shape[0] == 0:
            continue
        freqs_psd, psd = welch(Xk, fs=sfreq, nperseg=min(Xk.shape[-1], int(sfreq * 2)), axis=-1)
        mean_psd = psd.mean(axis=0)

        entry: Dict[str, float | str] = {
            "target_freq_hz": float(freq),
            "channel": ch_names[oz_idx],
            "n_trials": int(Xk.shape[0]),
        }
        for h in (1, 2, 3):
            fh = freq * h
            if fh >= sfreq / 2.0 - 1.0:
                continue
            idx = int(np.argmin(np.abs(freqs_psd - fh)))
            sig_power = float(mean_psd[idx])
            # Neighboring noise band: +/- [0.5, 2.0] Hz excluding target bin
            noise_mask = (np.abs(freqs_psd - fh) >= 0.5) & (np.abs(freqs_psd - fh) <= 2.0)
            noise_power = float(mean_psd[noise_mask].mean()) if np.any(noise_mask) else 1e-12
            snr_db = 10.0 * np.log10(max(sig_power, 1e-18) / max(noise_power, 1e-18))
            entry[f"h{h}_freq_hz"] = round(fh, 2)
            entry[f"h{h}_snr_db"] = round(float(snr_db), 3)
        rows.append(entry)
    return rows
