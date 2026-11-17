#!/usr/bin/env python3
import time
import os
import requests
import argparse
from pathlib import Path

def send_to_discord(webhook_url: str, text: str):
    if not text.strip():
        return

    data = {
        "content": f"```\n{text}\n```"  # Wrap in code blocks for readability
    }

    try:
        resp = requests.post(webhook_url, json=data, timeout=10)
        # Discord webhooks usually return 204 No Content on success
        if resp.status_code not in (200, 204):
            print(f"[WARN] Discord webhook returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Error sending to Discord: {e}")

def follow(thefile, flush_interval=2.0, max_chunk_chars=1800):
    """Generator that yields buffered chunks of new lines in a file."""
    # Skip existing content; start from the end
    thefile.seek(0, os.SEEK_END)

    buffer = []
    last_send = time.time()

    while True:
        line = thefile.readline()
        if not line:
            # No new line, see if we should flush based on time
            if buffer and (time.time() - last_send > flush_interval):
                yield "".join(buffer)
                buffer = []
                last_send = time.time()
            time.sleep(0.1)
            continue

        buffer.append(line)

        # If buffer gets close to Discord limit, flush immediately
        if len("".join(buffer)) > max_chunk_chars:
            yield "".join(buffer)
            buffer = []
            last_send = time.time()

def main():
    parser = argparse.ArgumentParser(
        description="Stream a log file to a Discord webhook in near real time."
    )
    parser.add_argument(
        "--webhook-url",
        required=True,
        help="Discord webhook URL (from Channel → Integrations → Webhooks).",
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to the log file to watch.",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=2.0,
        help="Seconds between forced sends when there is buffered data (default: 2.0).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)

    if not log_path.exists():
        print(f"[INFO] Waiting for log file to be created: {log_path}")
        while not log_path.exists():
            time.sleep(1)

    print(f"[INFO] Streaming {log_path} to Discord...")
    send_to_discord(args.webhook_url, f"🚀 Log Stream Started for `{log_path}`")

    with log_path.open("r", encoding="utf-8", errors="replace") as logfile:
        for chunk in follow(logfile, flush_interval=args.flush_interval):
            send_to_discord(args.webhook_url, chunk)

if __name__ == "__main__":
    main()
