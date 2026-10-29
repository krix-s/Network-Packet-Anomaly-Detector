# Packet Traffic Signal Lab

A separate local experiment that demonstrates Signals and Systems concepts on packet metadata. It is distinct from the original supervised flow classifier and the GitHub Pages demo.

## Run the safe simulation

From the project root:

```bash
.venv/bin/pip install -r live_traffic_demo/requirements.txt
.venv/bin/python live_traffic_demo/app.py --mode simulate
```

Open <http://127.0.0.1:8765>. The generated traffic includes a normal-like baseline and two labeled bursts. The app trains a 1D convolutional autoencoder on the baseline and flags windows with high reconstruction error. Results apply only to the generated simulation.

If port 8765 is occupied, add `--port 8766` to the command and open <http://127.0.0.1:8766> instead.

## Capture live packets

First list interfaces with `ip link` (Linux) and choose one you are authorized to monitor. Then start the app with packet-capture permission:

```bash
sudo .venv/bin/python live_traffic_demo/app.py --mode live --interface wlan0 --baseline-seconds 300 --monitor-seconds 300
```

Replace `wlan0` with the chosen interface. The first five minutes form a local baseline; the following five minutes are scored. The capture callback reads packet lengths and timestamps and discards packets immediately; it does not save payloads or a PCAP. Use the simulation mode if the lab PC does not grant capture permission.

The live score means “unusual relative to the calibration period,” not “confirmed attack.” It is a prototype, not a validated network IDS. The 1D CNN is trained to reconstruct normal signal windows; it does not predict one of the original 15 attack categories.

## Signal pipeline

1. Aggregate captured packets into one-second bins: packet count, bytes, and mean inter-arrival time from packet timestamps.
2. Apply `log1p` to compress skewed magnitudes.
3. Apply a causal EWMA low-pass filter; use the residual as a high-frequency/change channel.
4. Standardize using the baseline period only.
5. Feed 32-second, six-channel windows to the convolutional autoencoder.
6. Compare reconstruction error with a threshold calibrated on later baseline windows.

A causal filter is used so the live estimate does not depend on future packets. The original labeled dataset remains the source for the binary and multiclass flow-classification models; this live experiment is a separate anomaly-scoring extension.
