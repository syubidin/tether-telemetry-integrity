# Contributing

1. Create a focused branch for each change.
2. Keep business logic in `src/tether`; keep Streamlit-specific code in `app.py`.
3. Add a test for new behaviour and relevant edge cases.
4. Run `pytest` and `ruff check app.py src tests scripts` before opening a pull request.
5. Do not commit credentials, proprietary telemetry, personally identifiable data, generated model binaries or private datasets.
6. Keep claims about model performance tied to reproducible evaluation data; never describe synthetic metrics as real-track validation.
