# FlowWatch demo guide

FlowWatch is a small, static browser app for presenting the flow classifier to a general audience. The GitHub Pages deployment is designed to open as a web link, so the lab PC does not need the Parquet dataset, Python, or model files. The app does not capture live network traffic.

## Start the demo

After Pages has been enabled for the repository, open <https://krix-s.github.io/Network-Packet-Anomaly-Detector/> from any computer. To run the same demo locally from a checkout:

```bash
python3 scripts/demo_server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The app reads precomputed predictions and four labeled examples from `web/demo_bundle.json`. It does not need the raw flow feature vectors at run time.

To regenerate that bundle after training new models, use this computer where the dataset and model files are available:

```bash
.venv/bin/python scripts/train.py --task binary --seed 43
.venv/bin/python scripts/train.py --task multiclass --training-weighting source-prior --seed 43
.venv/bin/python scripts/build_demo_bundle.py
```

Stop the server with Ctrl+C. It binds to `127.0.0.1` so the demo is available only on the same computer.

If startup says the port is already in use, FlowWatch may already be running; open the URL instead of starting a second copy. Otherwise stop the process using that port, or launch the demo on another local port with `python3 scripts/demo_server.py --port 8001` and open `http://127.0.0.1:8001`.

## Three-minute walkthrough

1. Start with the receipt analogy: each flow row is a short summary of a computer connection—its duration and traffic counts—not the message content.
2. Select **Everyday web browsing** and analyze it. Show that the binary model calls it benign and that the stored dataset label agrees.
3. Select **Traffic surge (DDoS)**. Show the binary screening flag, the attack category guess, and the top category scores.
4. Select **Rare SQL Injection example**. The model predicts a different attack category. Use this to show that machine learning can be confidently wrong, particularly for a rare class.
5. Close with the metrics above the demo and explain that they come from a sampled test set from this dataset, not a live network or an independent organization.

## Plain-language script

> Network connections leave behind measurable clues, like how long a connection lasts and how much data it moves. I trained two models on labeled flow summaries: one asks “does this look like an attack?” and the other asks “which attack category is most likely?” This page lets you try examples from the dataset. The model can recognize common patterns, but it can miss unusual or rare attacks, so this is a research prototype and not a replacement for a security analyst.

## What the page does and does not show

- The four scenarios are real flow rows sampled from the local dataset; the page does not stream or capture traffic.
- The Pages deployment copies only `index.html` and `demo_bundle.json`. It does not publish the raw Parquet dataset, trained model binaries, or the full feature vectors in `web/demo_examples.json`.
- Binary screening uses the seed-43 binary HGB artifact. Attack-category prediction uses the seed-43 source-prior HGB artifact.
- The model score shown as confidence is not calibrated as a real-world probability.
- The summary metrics are source-weighted sample estimates. The high multiclass accuracy is driven by the dataset's class distribution; macro-F1 and rare-class examples provide additional context.
- The example rows are presentation aids, not a benchmark. Do not use their predictions to claim production readiness.
