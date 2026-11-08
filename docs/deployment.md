# GitHub and free deployment

The simplest supported route for this project is **GitHub + Streamlit Community Cloud**. The app is a Streamlit application with a root-level `app.py` and `requirements.txt`.

## 1. Create the GitHub repository

1. Sign in to GitHub.
2. Create a new repository named `tether-telemetry-integrity`.
3. Choose **Public** if you want a portfolio-ready public repository. Do not upload private telemetry or credentials.
4. Do not initialize another README/license if you are uploading this project as-is; this ZIP already contains them.

## 2. Push the project from your computer

Extract the ZIP, open a terminal in the extracted `telemetry-integrity-gate` folder, then run:

```bash
git init
git add .
git commit -m "Build TETHER telemetry integrity prototype"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/tether-telemetry-integrity.git
git push -u origin main
```

Replace `<YOUR_USERNAME>` with your GitHub username. Git may ask you to authenticate through your browser or credential manager. Do not paste a personal access token into source code or commit it to the repository.

If `git remote add origin` reports that `origin` already exists, use:

```bash
git remote set-url origin https://github.com/<YOUR_USERNAME>/tether-telemetry-integrity.git
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Authorize access to the repository if prompted.
3. Select **Create app**.
4. Choose repository `YOUR_USERNAME/tether-telemetry-integrity`, branch `main`, and main file path `app.py`.
5. Choose Python 3.11 if the runtime selector is shown.
6. Optionally choose an app subdomain such as `tether-telemetry-integrity`.
7. Select **Deploy** and wait for the build to finish.

The live URL will be similar to:

`https://<chosen-subdomain>.streamlit.app/`

Use the URL shown in the Streamlit app settings; availability and subdomain uniqueness are controlled by the provider.

## 4. Verify the live demo

- Select **Clean synthetic lap** and run analysis. It should show no escalated `ALERT` findings.
- Select **Fault-injected synthetic lap** and run analysis. It should show `DROPOUT`, `STUCK`, `SPIKE` and `NOISE` alerts, plus trend candidates under review.
- Upload only a small, non-sensitive UTF-8 CSV if you need to test the input path.

## 5. Add it to your portfolio

Suggested project card:

**TETHER — Telemetry Health & Event Reliability Engine**

A CPU-first Python ML prototype that screens motorsport-style telemetry for sensor data integrity issues. Combines robust time-series features, a Random Forest classifier, deterministic fault rules, defensive CSV validation and uncertainty-aware triage.

Links: `GitHub repository` · `Live demo` · `Case study`

## Troubleshooting

- **Build fails:** open app logs in Streamlit Community Cloud and check dependency installation. `requirements.txt` is required.
- **Module not found:** make sure the `src/tether` package and `pyproject.toml` were pushed and that the entrypoint is the root `app.py`.
- **App sleeps or cold-starts:** free services may pause inactive apps; allow the app to wake up before testing.
- **CSV rejected:** confirm UTF-8, comma delimiter, a header, unique column names, and the documented 5 MB / 20,000 row / 40 column limits.
- **Unexpected REVIEW:** inspect the time series manually. Drift and low-score classes are intentionally not presented as confirmed failures.
