# Contributing

1. Keep channel / model contracts explicit and versioned.
2. Add tests for new signal relationships, missing values, timing gaps and fault modes.
3. Do not promote `REVIEW` to `ALERT` without documented evidence and tests for the expected operating envelope.
4. Keep the default path CPU-first and preserve the NumPy fallback when adding native code.
5. Do not commit downloaded third-party datasets, user telemetry, credentials, model dumps or build artifacts.
6. Add dataset attribution and verify reuse rights before introducing external data.

Run locally:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check app.py src tests scripts
bandit -q -r src scripts app.py
pip-audit -r requirements.txt
pip-audit -r requirements-api.txt
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```
