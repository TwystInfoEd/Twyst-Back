import argparse
import asyncio
import re
import time

import httpx
import serial

BASE_URL = "http://localhost:8000"
BAUD = 115200


def parse_line(line: str, buf: dict) -> bool:
    # returns true when all fields are filled
    line = line.strip()

    # generic key:value parser for single-line firmware output like:
    # acc_x:-1.770 acc_y:-0.240 acc_z:10.181 gyro_x:3.427 gyro_y:-0.924 gyro_z:-0.145 roll:-1.35 pitch:9.86 yaw:0.00
    for m in re.finditer(r"([a-zA-Z_]+)\s*[:=]\s*([\-\d.]+)", line):
        key = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if key in ("acc_x", "acc_y", "acc_z",
                   "gyro_x", "gyro_y", "gyro_z",
                   "roll", "pitch", "yaw"):
            buf[key] = val

    # fallback: also accept the older multi-line / labelled formats
    m = re.search(r"Proc Acc\[g\]:\s*([\-\d.]+),\s*([\-\d.]+),\s*([\-\d.]+)", line)
    if m:
        buf["acc_x"], buf["acc_y"], buf["acc_z"] = map(float, m.groups())

    m = re.search(r"Proc Gyro\[deg/s\]:\s*([\-\d.]+),\s*([\-\d.]+),\s*([\-\d.]+)", line)
    if m:
        buf["gyro_x"], buf["gyro_y"], buf["gyro_z"] = map(float, m.groups())

    m = re.search(r"Angles \[deg\] R/P/Y:\s*([\-\d.]+),\s*([\-\d.]+),\s*([\-\d.]+)", line)
    if m:
        buf["roll"], buf["pitch"], buf["yaw"] = map(float, m.groups())

    required = {"acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "roll", "pitch", "yaw"}
    return required.issubset(buf)


async def post_frame(buf: dict, endpoint: str, client: httpx.AsyncClient):
    payload = dict(buf)
    payload["timestamp"] = time.time()
    try:
        r = await client.post(endpoint, json=payload, timeout=1.0)
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        print(f"  [warn] {e}")
        return {}


async def run(port: str, name: str, bezier_order: int = 8,
    host: str | None = None, endpoint_path: str | None = None,
    baud: int | None = None, debug: bool = False):

    baud = baud or BAUD
    base_url = host or BASE_URL
    endpoint_path = endpoint_path or "/frame"
    endpoint_url = f"{base_url.rstrip('/')}{endpoint_path}"

    print(f"Opening {port} at {baud} baud. Posting frames to {endpoint_url}. Press Ctrl+C to stop.\n")
    buf: dict = {}
    frame_count = 0

    async with httpx.AsyncClient(timeout=1.0) as client:
        try:
            with serial.Serial(port, baud, timeout=2) as ser:
                while True:
                    line = ser.readline().decode("utf-8", errors="ignore")
                    if not line:
                        continue
                    if debug:
                        print(f"[{frame_count}] {line.rstrip()}")
                    ready = parse_line(line, buf)
                    if ready:
                        resp = await post_frame(buf, endpoint_url, client)
                        if debug:
                            print(f"  -> posted frame #{frame_count}; server replied: {resp}")
                        frame_count += 1
                        buf = {}  # reset for next triplet
        except KeyboardInterrupt:
            print("\nStopping…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Twyst ESP32 serial bridge")
    ap.add_argument("--port", default="COM3", help="Serial port, e.g. /dev/ttyUSB0")
    ap.add_argument("--name", required=True,
                    help="Motion name to save (record) or compare against")
    ap.add_argument("--order", type=int, default=8,
                    help="Bézier order (default 8)")
    ap.add_argument("--host", default=None, help="Base URL of backend, e.g. http://192.168.1.136:8000")
    ap.add_argument("--endpoint", default=None, help="Endpoint path to POST frames to (default /frame)")
    ap.add_argument("--baud", type=int, default=None, help="Serial baud rate override (default 115200)")
    ap.add_argument("--debug", action="store_true", help="Print debug serial lines and server responses")
    args = ap.parse_args()
    asyncio.run(run(args.port, args.name, args.order, host=args.host, endpoint_path=args.endpoint, baud=args.baud, debug=args.debug))
