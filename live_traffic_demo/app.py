"""Local packet-signal demo. Captures metadata only; never stores packet payloads."""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from core import bin_packets, score_trace, synthetic_baseline, synthetic_bursts, train_detector

ROOT = Path(__file__).resolve().parent
STATE: dict = {"phase": "starting", "mode": "", "message": "Starting demo", "trace": []}
LOCK = threading.Lock()


def update(**values) -> None:
    with LOCK:
        STATE.update(values)


def add_trace(item: dict) -> None:
    with LOCK:
        STATE["trace"].append(item)
        STATE["trace"] = STATE["trace"][-300:]
        STATE["latest"] = item
        STATE["packets_total"] = STATE.get("packets_total", 0) + int(item.get("packets", 0))
        STATE["anomalies"] = STATE.get("anomalies", 0) + int(item.get("alert", False))


def stream_simulation() -> None:
    update(phase="calibration", mode="simulation", message="Training on emulated baseline traffic", trace=[], packets_total=0, anomalies=0)
    baseline = synthetic_baseline()
    detector = train_detector(baseline)
    test, anomaly_bins = synthetic_bursts()
    errors, smooth, flags = score_trace(detector, test)
    # Each score corresponds to the end of a 32-second window.
    all_features = np.log1p(test)
    smoothed_rates = np.expm1(smooth[:, 0])
    residuals = all_features[:, 0] - smooth[:, 0]
    labels = []
    for j in range(len(errors)):
        start = j
        end = j + detector["window"]
        labels.append(bool(anomaly_bins[start:end].any()))
    truth = np.asarray(labels, dtype=bool)
    tp = int(np.sum(flags & truth)); fp = int(np.sum(flags & ~truth)); fn = int(np.sum(~flags & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    update(
        phase="simulation", message="Synthetic burst-pattern emulation. These are not live packets and do not measure attack-detection accuracy.",
        threshold=detector["threshold"], simulation_metrics={"precision": precision, "recall": recall, "true_positives": tp, "false_positives": fp, "false_negatives": fn},
        training_seconds=detector["baseline_seconds"],
    )
    time.sleep(0.4)
    for i in range(len(test)):
        if i >= detector["window"] - 1:
            j = i - detector["window"] + 1
            score = float(errors[j]); alert = bool(flags[j])
        else:
            score = 0.0; alert = False
        add_trace({"t": i + 1, "packets": int(test[i, 0]), "pps": float(test[i, 0]),
                   "smooth": float(smoothed_rates[i]), "residual": float(abs(residuals[i])),
                   "score": score, "threshold": detector["threshold"], "alert": alert,
                   "known_burst": bool(anomaly_bins[i])})
        time.sleep(0.035)
    update(phase="complete", message="Simulation complete. Switch to --mode live to monitor an authorized interface.")


def stream_live(interface: str, baseline_seconds: int, monitor_seconds: int) -> None:
    try:
        from scapy.all import sniff
    except ImportError as exc:
        update(phase="error", message="Scapy is missing. Install live_traffic_demo/requirements.txt.")
        return

    def one_second() -> tuple[float, float, float]:
        packet_events: list[tuple[float, int]] = []
        def record(pkt) -> None:
            # Keep timestamp and frame length for this bin; discard payload immediately.
            packet_events.append((float(pkt.time), int(len(pkt))))
        try:
            sniff(iface=interface, prn=record, store=False, timeout=1)
        except (PermissionError, OSError) as exc:
            raise RuntimeError(
                f"Cannot capture on {interface}: {exc}. Check the interface name and packet-capture permission."
            ) from exc
        return bin_packets(packet_events)

    update(phase="baseline", mode="live capture", message=f"Capturing {baseline_seconds}s baseline on {interface}", trace=[], packets_total=0, anomalies=0)
    baseline = []
    try:
        for i in range(baseline_seconds):
            row = one_second(); baseline.append(row)
            if i % 5 == 0:
                update(progress=f"Baseline capture: {i + 1}/{baseline_seconds}s", packets_total=sum(int(r[0]) for r in baseline))
    except RuntimeError as exc:
        update(phase="error", message=str(exc))
        return
    try:
        detector = train_detector(np.asarray(baseline, dtype=np.float32))
    except Exception as exc:
        update(phase="error", message=f"Could not calibrate baseline: {exc}")
        return
    update(phase="monitoring", message=f"CNN calibrated on {baseline_seconds}s of local baseline traffic; monitoring {interface}", threshold=detector["threshold"], training_seconds=baseline_seconds)
    history = []
    for i in range(monitor_seconds):
        try:
            row = one_second()
        except RuntimeError as exc:
            update(phase="error", message=str(exc))
            return
        history.append(row)
        raw = np.asarray(history, dtype=np.float32)
        errors, smooth, flags = score_trace(detector, raw)
        score = float(errors[-1]) if len(errors) else 0.0
        alert = bool(flags[-1]) if len(flags) else False
        features = np.log1p(raw)
        smooth_pps = float(np.expm1(smooth[-1, 0]))
        residual = float(abs(features[-1, 0] - smooth[-1, 0]))
        add_trace({"t": i + 1, "packets": int(row[0]), "pps": row[0], "smooth": smooth_pps,
                   "residual": residual, "score": score, "threshold": detector["threshold"],
                   "alert": alert, "known_burst": None})
        update(progress=f"Monitoring: {i + 1}/{monitor_seconds}s")
    update(phase="complete", message="Live capture finished. Alerts mean unusual compared with this capture's baseline, not confirmed attacks.")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/state":
            with LOCK:
                payload = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload); return
        if self.path in ("/", "/index.html"):
            payload = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload); return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("simulate", "live"), default="simulate")
    parser.add_argument("--interface", help="Capture interface, e.g. wlan0 or eth0 (live mode only)")
    parser.add_argument("--baseline-seconds", type=int, default=300)
    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.mode == "live" and not args.interface:
        parser.error("--interface is required in live mode")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    worker = stream_simulation if args.mode == "simulate" else lambda: stream_live(args.interface, args.baseline_seconds, args.monitor_seconds)
    threading.Thread(target=worker, daemon=True).start()
    print(f"Open http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    if args.mode == "live":
        print(f"Capturing packet headers and lengths from {args.interface}; payloads are not stored.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
