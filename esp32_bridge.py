import argparse
import asyncio
import re
import time

import httpx
import serial

BASE_URL = "http://localhost:8000"
BAUD = 115200

NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

LINK_RE = re.compile(r"^LINK\s+secondary_connected=(\d)\s+state=(\w+)")

REQUIRED_FRAME_KEYS = {
    "ts",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "roll",
    "pitch",
    "yaw",
}


def parse_link_line(line: str) -> dict | None:
    m = LINK_RE.search(line.strip())
    if not m:
        return None
    return {
        "main_connected": True,  
        "secondary_connected": bool(int(m.group(1))),
        "state": m.group(2),
    }

def parse_frame_line(line: str) -> dict | None:
    """
    parses a single line, complete frame from either band:
      SEC acc_x=-1.770 acc_y=-0.240 acc_z=10.181 gyro_x=3.427 gyro_y=-0.924 gyro_z=-0.145 roll=-1.35 pitch=9.86 yaw=0.00
      MAIN imu2 acc_x=... acc_y=... ... yaw=0.00
    """
    fields: dict = {}
    for m in re.finditer(rf"([a-zA-Z_]+)\s*[:=]\s*({NUM_RE})", line):
        key = m.group(1)
        if key in REQUIRED_FRAME_KEYS:
            try:
                fields[key] = float(m.group(2))
            except ValueError:
                continue
    return fields if REQUIRED_FRAME_KEYS.issubset(fields) else None


async def post_link_status(status: dict, base_url: str, client: httpx.AsyncClient):
    payload = dict(status)
    payload["host_timestamp"] = time.time()
    try:
        await client.post(f"{base_url.rstrip('/')}/link/status", json=payload, timeout=1.0)
    except Exception as e:
        print(f"  [warn] link status post failed: {e}")


async def post_frame(frame: dict, endpoint: str, client: httpx.AsyncClient):
    payload = dict(frame)
    payload["host_timestamp"] = time.time()
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
    main_endpoint_path: str | None = None,
    baud: int | None = None, debug: bool = False):

    baud = baud or BAUD
    base_url = host or BASE_URL
    endpoint_path = endpoint_path or "/frame"
    main_endpoint_path = main_endpoint_path or "/frame/main"
    endpoint_url = f"{base_url.rstrip('/')}{endpoint_path}"
    main_endpoint_url = f"{base_url.rstrip('/')}{main_endpoint_path}"

    print(f"Opening {port} at {baud} baud.")
    print(f"  Secondary-band frames -> {endpoint_url}")
    print(f"  Main-band frames      -> {main_endpoint_url}")
    print("Press Ctrl+C to stop.\n")

    sec_count = 0
    main_count = 0

    async with httpx.AsyncClient(timeout=1.0) as client:
        try:
            with serial.Serial(port, baud, timeout=2) as ser:
                # As soon as the port opens we know the secondary band is talking to us,
                # but we don't yet know the state of its BLE link to the main band.
                await post_link_status(
                    {"secondary_connected": True, "main_connected": False, "state": "unknown"},
                    base_url, client,
                )
                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore")
                    if not raw:
                        continue
                    line = raw.rstrip()
                    if debug:
                        print(line)

                    link = parse_link_line(line)
                    if link is not None:
                        await post_link_status(link, base_url, client)
                        if debug:
                            print(f"  [link] {link}")
                        continue

                    if line.startswith("SEC "):
                        frame = parse_frame_line(line)
                        if frame is not None:
                            resp = await post_frame(frame, endpoint_url, client)
                            sec_count += 1
                            if debug:
                                print(f"  [sec #{sec_count}] posted; server replied: {resp}")
                        continue

                    if line.startswith("MAIN imu2"):
                        frame = parse_frame_line(line)
                        if frame is not None:
                            resp = await post_frame(frame, main_endpoint_url, client)
                            main_count += 1
                            if debug:
                                print(f"  [main #{main_count}] posted; server replied: {resp}")
                        continue

                    # anything else is not forwarded
        except KeyboardInterrupt:
            print("\nStopping…")
        except serial.SerialException as e:
            print(f"\nSerial error: {e}")
        finally:
         
            await post_link_status(
                {"secondary_connected": False, "main_connected": False, "state": "bridge_stopped"},
                base_url, client,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Twyst ESP32 serial bridge")
    ap.add_argument("--port", default="COM3", help="Serial port for the SECONDARY band, e.g. COM13")
    ap.add_argument("--name", required=True,
                    help="Motion name to save (record) or compare against")
    ap.add_argument("--order", type=int, default=8,
                    help="Bézier order (default 8)")
    ap.add_argument("--host", default=None, help="Base URL of backend, e.g. http://192.168.1.136:8000")
    ap.add_argument("--endpoint", default=None, help="Endpoint path for secondary-band frames (default /frame)")
    ap.add_argument("--main-endpoint", default=None, help="Endpoint path for main-band frames (default /frame/main)")
    ap.add_argument("--baud", type=int, default=None, help="Serial baud rate override (default 115200)")
    ap.add_argument("--debug", action="store_true", help="Print debug serial lines and server responses")
    args = ap.parse_args()
    asyncio.run(run(
        args.port, args.name, args.order,
        host=args.host, endpoint_path=args.endpoint, main_endpoint_path=args.main_endpoint,
        baud=args.baud, debug=args.debug,
    ))