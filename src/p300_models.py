"""P300 target detection and multi-repetition character speller decoding.

Implements:
1. xDAWN spatial filtering + Riemannian Tangent Space + Class-Balanced Logistic Regression (pyRiemann).
2. EEGNet-8,2 deep learning baseline in PyTorch with class-weighted Cross-Entropy loss (handling 1:5 imbalance).
3. Multi-repetition (1..5 rounds) probability fusion decoder for 6x6 (36-character) row-column speller.
4. Numerical ERP waveform statistics around P300 latency (250-450 ms) without plotting.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class XdawnRiemannianDecoder:
    """xDAWN spatial filtering + Riemannian Tangent Space mapping + balanced classifier."""

    def __init__(
        self,
        n_filters: int = 4,
        cov_estimator: str = "lwf",
        xdawn_estimator: str = "scm",
        classifier_c: float = 1.0,
    ) -> None:
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace

        self.pipeline = Pipeline(
            steps=[
                (
                    "xdawn_cov",
                    XdawnCovariances(
                        nfilter=n_filters,
                        estimator=cov_estimator,
                        xdawn_estimator=xdawn_estimator,
                        classes=[1],  # Target class ERP spatial filters
                    ),
                ),
                ("tangent_space", TangentSpace(metric="riemann")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=classifier_c,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XdawnRiemannianDecoder":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return target probability P(y=1 | x) of shape (n_trials,)."""
        return self.pipeline.predict_proba(X)[:, 1]


def _build_eegnet_module(
    n_channels: int,
    n_times: int,
    n_classes: int = 2,
    f1: int = 8,
    d: int = 2,
    f2: int = 16,
    kernel_length: int = 64,
    dropout: float = 0.5,
):
    """Construct Lawhern et al. (2018) EEGNet PyTorch nn.Module."""
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f1 * d),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
                nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                feat_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]
            self.classifier = nn.Linear(feat_dim, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            return self.classifier(x)

    return EEGNet()


class EEGNetP300Decoder:
    """PyTorch EEGNet with class-weighted Cross-Entropy loss for 1:5 imbalanced P300 detection."""

    def __init__(
        self,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_ratio: float = 0.15,
        patience: int = 10,
        seed: int = 42,
    ) -> None:
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.kernel_length = kernel_length
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_ratio = val_ratio
        self.patience = patience
        self.seed = seed

        self.model = None
        self.ch_mean_: np.ndarray | None = None
        self.ch_std_: np.ndarray | None = None
        self.device_ = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Channel-wise z-score normalization fitted strictly on Day 1 training data."""
        if fit:
            self.ch_mean_ = X.mean(axis=(0, 2), keepdims=True)
            self.ch_std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return (X - self.ch_mean_) / self.ch_std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetP300Decoder":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_norm = self._standardize(X, fit=True).astype(np.float32)

        # Stratified Day 1 train/val split for early stopping (zero Day 2 leakage)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
        train_idx, val_idx = next(splitter.split(X_norm, y))

        X_tr, y_tr = X_norm[train_idx], y[train_idx]
        X_va, y_va = X_norm[val_idx], y[val_idx]

        # Inverse-frequency class weights to counter 1:5 Target/NonTarget imbalance
        n_neg = max(int(np.sum(y_tr == 0)), 1)
        n_pos = max(int(np.sum(y_tr == 1)), 1)
        total = n_neg + n_pos
        class_weights = torch.tensor(
            [total / (2.0 * n_neg), total / (2.0 * n_pos)],
            dtype=torch.float32,
            device=self.device_,
        )

        n_channels, n_times = X.shape[1], X.shape[2]
        self.model = _build_eegnet_module(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=2,
            f1=self.f1,
            d=self.d,
            f2=self.f2,
            kernel_length=min(self.kernel_length, max(8, n_times // 2)),
            dropout=self.dropout,
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr[:, None, :, :]), torch.from_numpy(y_tr.astype(np.int64))),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_auc = -1.0
        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        X_va_t = torch.from_numpy(X_va[:, None, :, :]).to(self.device_)

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_probs = torch.softmax(self.model(X_va_t), dim=1)[:, 1].cpu().numpy()
            val_auc = roc_auc_score(y_va, val_probs) if len(np.unique(y_va)) == 2 else 0.5
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return target probability P(y=1 | x) of shape (n_trials,)."""
        import torch

        self.model.eval()
        X_norm = self._standardize(X, fit=False).astype(np.float32)
        probs_list = []
        with torch.no_grad():
            for start in range(0, len(X_norm), 256):
                xb = torch.from_numpy(X_norm[start : start + 256, None, :, :]).to(self.device_)
                p = torch.softmax(self.model(xb), dim=1)[:, 1].cpu().numpy()
                probs_list.append(p)
        return np.concatenate(probs_list, axis=0)


def evaluate_p300_flash_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute single-flash binary detection metrics suited for imbalanced P300."""
    y_pred = (y_prob >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.5
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "roc_auc": auc,
        "balanced_acc": bal_acc,
        "f1_target": f1,
    }


def evaluate_speller_character_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_rows: int = 6,
    n_cols: int = 6,
    n_repetitions: int = 5,
) -> Dict[int, float]:
    """Multi-repetition probability fusion for 6x6 P300 character speller.

    In OpenBMI (Lee2019_ERP), each character is spelled over `n_repetitions` (5)
    rounds of `n_rows + n_cols` (12) flashes. Within every 12-flash round, exactly
    2 flashes are Targets (1 target row + 1 target column) and 10 are NonTargets
    (5 non-target rows + 5 non-target columns).

    We accumulate log-odds log(p / (1 - p)) across repetitions 1..R for the 6 row
    candidates and 6 column candidates; a character is decoded correctly iff both
    the true target row and true target column attain the highest accumulated score.

    Returns:
        Dict mapping repetition count R (1..n_repetitions) -> character accuracy in [0, 1].
    """
    flashes_per_rep = n_rows + n_cols
    flashes_per_char = flashes_per_rep * n_repetitions
    n_chars = len(y_true) // flashes_per_char
    if n_chars == 0:
        return {r: 0.0 for r in range(1, n_repetitions + 1)}

    eps = 1e-6
    log_odds = np.log(np.clip(y_prob, eps, 1.0 - eps) / np.clip(1.0 - y_prob, eps, 1.0 - eps))

    correct_by_rep = {r: 0 for r in range(1, n_repetitions + 1)}
    valid_chars = 0

    for c_idx in range(n_chars):
        char_start = c_idx * flashes_per_char
        row_cum = np.zeros(n_rows, dtype=np.float64)
        col_cum = np.zeros(n_cols, dtype=np.float64)

        # Verify each repetition in this character block has 2 targets + 10 non-targets
        block_valid = True
        rep_scores: List[Tuple[np.ndarray, np.ndarray]] = []
        for r in range(n_repetitions):
            s = char_start + r * flashes_per_rep
            e = s + flashes_per_rep
            yr = y_true[s:e]
            lr = log_odds[s:e]
            tgt_scores = lr[yr == 1]
            ntgt_scores = lr[yr == 0]
            if len(tgt_scores) < 2 or len(ntgt_scores) < (flashes_per_rep - 2):
                block_valid = False
                break
            # Assign index 0 as true target row / col, indices 1..5 as non-target rows / cols
            row_r = np.concatenate([[tgt_scores[0]], ntgt_scores[: n_rows - 1]])
            col_r = np.concatenate([[tgt_scores[1]], ntgt_scores[n_rows - 1 : n_rows + n_cols - 2]])
            rep_scores.append((row_r, col_r))

        if not block_valid:
            continue

        valid_chars += 1
        for r_idx, (row_r, col_r) in enumerate(rep_scores, start=1):
            row_cum += row_r
            col_cum += col_r
            if int(np.argmax(row_cum)) == 0 and int(np.argmax(col_cum)) == 0:
                correct_by_rep[r_idx] += 1

    if valid_chars == 0:
        return {r: 0.0 for r in range(1, n_repetitions + 1)}
    return {r: correct_by_rep[r] / valid_chars for r in range(1, n_repetitions + 1)}


def extract_p300_erp_stats(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    tmin: float,
    ch_names: Sequence[str],
    p300_window: Tuple[float, float] = (0.25, 0.45),
) -> List[Dict[str, float | str]]:
    """Compute numerical P300 ERP waveform statistics (250-450 ms) on key midline/parietal channels."""
    times = tmin + np.arange(X.shape[-1], dtype=np.float64) / sfreq
    win_mask = (times >= p300_window[0]) & (times <= p300_window[1])
    win_times = times[win_mask]

    key_chs = [ch for ch in ("Pz", "Cz", "POz", "Oz") if ch in ch_names]
    if not key_chs:
        key_chs = [ch_names[0]]

    rows: List[Dict[str, float | str]] = []
    for ch in key_chs:
        idx = ch_names.index(ch)
        erp_target = X[y == 1, idx, :].mean(axis=0)
        erp_nontarget = X[y == 0, idx, :].mean(axis=0)
        diff_wave = erp_target - erp_nontarget

        peak_i = int(np.argmax(diff_wave[win_mask]))
        rows.append(
            {
                "channel": ch,
                "target_mean_uv": round(float(erp_target[win_mask].mean() * 1e6), 3),
                "nontarget_mean_uv": round(float(erp_nontarget[win_mask].mean() * 1e6), 3),
                "p300_peak_diff_uv": round(float(diff_wave[win_mask][peak_i] * 1e6), 3),
                "p300_peak_latency_ms": round(float(win_times[peak_i] * 1000.0), 1),
            }
        )
    return rows
