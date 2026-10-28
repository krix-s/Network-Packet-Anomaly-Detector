# Lab-PC deployment

The presentation demo is packaged as a static site for GitHub Pages. Its URL is:

<https://krix-s.github.io/Network-Packet-Anomaly-Detector/>

## One-time GitHub Pages setup

The repository is public and the deployment workflow is in `.github/workflows/pages.yml`. A repository maintainer must enable the Pages publishing source once:

1. Open the repository on GitHub and choose **Settings**.
2. Open **Pages** under **Code and automation**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Open **Actions**, choose **Deploy FlowWatch demo**, and run the workflow (or rerun the latest push if it failed before Pages was enabled).
5. Wait for the `deploy` job to finish. Then open the URL at the top of this page from the lab PC.

GitHub's current setup instructions require choosing GitHub Actions as the Pages source before the custom workflow can publish. See [GitHub's Pages publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) and [custom workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## What the deployment contains

The workflow publishes only `web/index.html` and `web/demo_bundle.json`. The bundle has precomputed output for four examples and the published metrics. It does not contain the 642 MB Parquet file, trained model binaries, or complete feature rows. The lab PC needs only a browser and internet access; it does not need Python, project dependencies, or access to this development computer.

This is an interactive presentation of fixed examples. It does not accept arbitrary flow records, retrain models, or inspect live traffic. Rebuilding the bundle after changing models requires the dataset and trained artifacts on a machine that has them:

```bash
.venv/bin/python scripts/build_demo_bundle.py
```

Commit and push the updated `web/demo_bundle.json`; the Pages workflow will publish it.

## Local fallback

If Pages is not enabled yet or the lab has no internet access, copy the repository checkout (including `web/demo_bundle.json`) to the lab PC and run:

```bash
python3 scripts/demo_server.py
```

Open <http://127.0.0.1:8000>. This local static server uses Python's standard library; it does not need the dataset or model files.

## Possible live-flow phase

Live flow analysis is possible, but it is not part of this static demo. A later version could receive NetFlow/IPFIX records from an authorized router or switch exporter, or derive flows with a packet-to-flow tool such as NFStream. The collector would need to normalize each record to the model's 36 predictor columns, using the same feature definitions and units as training. `Label` and `Attack` are training targets and are not known for unlabeled live records. Live performance would need separate analyst-confirmed labels and testing on traffic captured in the intended environment. Keep any collector on an authorized, private lab network; the Pages site does not receive flow data.

References: [IETF RFC 7011, IPFIX flow export and collection](https://datatracker.ietf.org/doc/rfc7011/) and [NFStream documentation](https://www.nfstream.org/docs/).
