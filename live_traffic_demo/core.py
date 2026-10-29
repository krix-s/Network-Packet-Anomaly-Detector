"""Signal processing and 1D CNN autoencoder for packet-rate windows."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW = 32
ALPHA = 0.22
CHANNELS = 6


class SignalAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(CHANNELS, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, CHANNELS, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def signal_channels(raw: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Apply log compression, causal EWMA low-pass, and high-pass residual."""
    logged = np.log1p(np.maximum(np.asarray(raw, dtype=np.float32), 0))
    smooth = np.empty_like(logged)
    if not len(logged):
        return np.empty((0, CHANNELS), dtype=np.float32), smooth
    smooth[0] = logged[0]
    for i in range(1, len(logged)):
        smooth[i] = alpha * logged[i] + (1 - alpha) * smooth[i - 1]
    residual = logged - smooth
    return np.concatenate([smooth, residual], axis=1), smooth


def make_windows(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(x) < window:
        return np.empty((0, x.shape[1], window), dtype=np.float32)
    return np.stack([x[i:i + window].T for i in range(len(x) - window + 1)]).astype(np.float32)


def synthetic_baseline(n_seconds: int = 720, seed: int = 43) -> np.ndarray:
    """Create a labeled normal-traffic emulation, not a real network trace."""
    rng = np.random.default_rng(seed)
    result = np.zeros((n_seconds, 3), dtype=np.float32)
    for t in range(n_seconds):
        rate = max(2.0, 14 + 3 * np.sin(2 * np.pi * t / 180) + rng.normal(0, 1.8))
        count = int(rng.poisson(rate))
        sizes = np.clip(rng.lognormal(np.log(650), 0.45, count), 48, 1450)
        times = np.sort(rng.uniform(0, 1, count))
        mean_iat_ms = float(np.diff(times).mean() * 1000) if count > 1 else 0.0
        result[t] = (count, sizes.sum(), mean_iat_ms)
    return result


def synthetic_bursts(n_seconds: int = 300, seed: int = 44) -> tuple[np.ndarray, np.ndarray]:
    """Create a test trace with two deliberately labeled burst intervals."""
    rng = np.random.default_rng(seed)
    anomaly_bins = np.zeros(n_seconds, dtype=bool)
    anomaly_bins[70:92] = True
    anomaly_bins[190:215] = True
    result = np.zeros((n_seconds, 3), dtype=np.float32)
    for t in range(n_seconds):
        if anomaly_bins[t]:
            count = int(rng.poisson(110))
            sizes = np.clip(rng.lognormal(np.log(180), 0.35, count), 48, 1450)
        else:
            rate = max(2.0, 14 + 3 * np.sin(2 * np.pi * t / 180) + rng.normal(0, 1.8))
            count = int(rng.poisson(rate))
            sizes = np.clip(rng.lognormal(np.log(650), 0.45, count), 48, 1450)
        times = np.sort(rng.uniform(0, 1, count))
        mean_iat_ms = float(np.diff(times).mean() * 1000) if count > 1 else 0.0
        result[t] = (count, sizes.sum(), mean_iat_ms)
    return result, anomaly_bins


def train_detector(
    baseline_raw: np.ndarray,
    *,
    seed: int = 43,
    epochs: int = 14,
    window: int = WINDOW,
) -> dict:
    """Fit on the first 80% of a normal baseline; calibrate threshold on the rest."""
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    features, _ = signal_channels(baseline_raw)
    split = int(len(features) * 0.8)
    train_features, calibration_features = features[:split], features[split:]
    if len(train_features) < window * 3 or len(calibration_features) < window:
        raise ValueError(f"Need at least {window * 4} baseline seconds; received {len(features)}")
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train_features - mean) / scale
    calibration_scaled = (calibration_features - mean) / scale
    train_x = make_windows(train_scaled, window)
    calibration_x = make_windows(calibration_scaled, window)
    if len(train_x) < 2 or len(calibration_x) < 1:
        raise ValueError("Not enough normal windows to train and calibrate")

    dataset = TensorDataset(torch.from_numpy(train_x))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    model = SignalAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(calibration_x)).numpy()
        errors = np.mean((recon - calibration_x) ** 2, axis=(1, 2))
    # The calibration segment is short in a classroom demo. Keep a margin over
    # its 99th percentile so isolated baseline spikes do not dominate the display.
    threshold = float(np.quantile(errors, 0.99) * 2.0)
    return {
        "model": model,
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "threshold": threshold,
        "threshold_rule": "2 × 99th percentile of held-out baseline reconstruction error",
        "calibration_errors": errors.astype(np.float32),
        "train_loss": history,
        "window": window,
        "baseline_seconds": int(len(baseline_raw)),
    }


def score_trace(detector: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, smooth = signal_channels(raw)
    scaled = (features - detector["mean"]) / detector["scale"]
    windows = make_windows(scaled, detector["window"])
    if not len(windows):
        return np.empty(0), smooth, np.empty(0, dtype=bool)
    model = detector["model"]
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(windows)).numpy()
    errors = np.mean((predicted - windows) ** 2, axis=(1, 2))
    return errors, smooth, errors > detector["threshold"]


def bin_packets(packet_events: list[tuple[float, int]]) -> tuple[float, float, float]:
    count = len(packet_events)
    total = int(sum(size for _, size in packet_events))
    times = np.asarray(sorted(timestamp for timestamp, _ in packet_events), dtype=np.float64)
    mean_iat_ms = float(np.diff(times).mean() * 1000) if count > 1 else 0.0
    return float(count), float(total), mean_iat_ms
