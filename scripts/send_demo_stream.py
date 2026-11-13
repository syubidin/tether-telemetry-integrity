"""Send bounded demo packets to an authenticated TETHER API using stdlib only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/telemetry/packet")
    parser.add_argument("--input", default=str(ROOT / "data" / "demo_clean.csv"))
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("--url must use http or https with a valid host.")

    token = os.getenv("TETHER_API_TOKEN", "")
    if len(token) < 32:
        parser.error("Set TETHER_API_TOKEN to a random secret of at least 32 characters.")
    if not 1 <= args.limit <= 2000:
        parser.error("--limit must be between 1 and 2,000")

    frame = pd.read_csv(args.input, nrows=args.limit)
    excluded = {"packet_seq", "time_s"}
    health_url = args.url.replace("/v1/telemetry/packet", "/health")
    with urlopen(Request(health_url), timeout=5) as response:  # nosec B310
        print("API health:", response.status)
    for row in frame.to_dict(orient="records"):
        values = {name: (None if pd.isna(value) else float(value)) for name, value in row.items() if name not in excluded}
        body = json.dumps({
            "contract_version": "1.0.0",
            "packet_seq": int(row["packet_seq"]),
            "timestamp_s": float(row["time_s"]),
            "values": values,
        }).encode("utf-8")
        request = Request(args.url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }, method="POST")
        with urlopen(request, timeout=5) as response:  # nosec B310
            result = json.loads(response.read().decode("utf-8"))
        if result.get("diagnostics"):
            print(result["diagnostics"])
    print(f"Sent {len(frame)} packets.")


if __name__ == "__main__":
    main()