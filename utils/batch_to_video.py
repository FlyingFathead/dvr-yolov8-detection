#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL_DEFAULT = "http://127.0.0.1:5000"
CANVAS_DEFAULT = "1280x720"

DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
]

SUMMARY_FIRST_RE = re.compile(
    r"First seen:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2})"
)
SUMMARY_LATEST_RE = re.compile(
    r"Latest:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2})"
)
CANVAS_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")


def http_get_json(url: str, timeout: int = 10):
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def http_get_bytes(url: str, timeout: int = 30):
    req = Request(url, headers={"Accept": "*/*"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def ensure_ffmpeg_or_exit():
    if shutil.which("ffmpeg") is None:
        print('ERROR: "ffmpeg" not found in PATH -- cannot render video.', file=sys.stderr)
        print('Install ffmpeg (e.g. "sudo apt-get install ffmpeg") or adjust PATH.', file=sys.stderr)
        sys.exit(2)


def project_root_from_utils() -> str:
    """
    This script lives in <repo>/utils/batch_to_video.py
    So project root is one level up from this file's directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def data_dir(project_root: str) -> str:
    return os.path.join(project_root, "data")


def parse_canvas(s: str):
    m = CANVAS_RE.match(s or "")
    if not m:
        raise ValueError(f'bad --canvas {s!r}, expected WxH like "{CANVAS_DEFAULT}"')
    w = int(m.group(1))
    h = int(m.group(2))
    if w < 2 or h < 2:
        raise ValueError("canvas too small")
    return w, h


def parse_dt(s: str) -> datetime:
    """
    Returns a naive datetime in UTC-ish terms (aware -> UTC -> drop tzinfo).
    Supports:
      - HTTP-date style: "Sat, 14 Feb 2026 15:06:14 GMT"
      - ISO / "YYYY-MM-DD HH:MM:SS"
    """
    if not isinstance(s, str):
        raise ValueError(f"timestamp is not a string: {type(s)}")
    s = s.strip()
    if not s:
        raise ValueError("empty timestamp")

    # Try RFC 2822 / HTTP-date
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
    except Exception:
        pass

    # Trim common trailing Z / offsets for naive parsing
    s2 = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", s)

    for fmt in DT_FORMATS:
        try:
            return datetime.strptime(s2, fmt)
        except ValueError:
            pass

    if len(s2) >= 19 and s2[4] == "-" and s2[7] == "-":
        try:
            return datetime.strptime(s2[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    raise ValueError(f"unparseable datetime: {s!r}")


def parse_first_latest_from_item(item: dict):
    """
    Returns (first_dt, latest_dt) as datetimes.
    Tries first_timestamp/latest_timestamp; if those fail, tries summary field.
    """
    first_raw = item.get("first_timestamp", "")
    latest_raw = item.get("latest_timestamp", "")

    try:
        first_dt = parse_dt(first_raw)
        latest_dt = parse_dt(latest_raw)
        return first_dt, latest_dt
    except Exception:
        pass

    summary = item.get("summary") or ""
    m1 = SUMMARY_FIRST_RE.search(summary)
    m2 = SUMMARY_LATEST_RE.search(summary)
    if m1 and m2:
        first_dt = parse_dt(m1.group(1).replace("T", " "))
        latest_dt = parse_dt(m2.group(1).replace("T", " "))
        return first_dt, latest_dt

    raise ValueError("could not parse first/latest timestamps")


def pick_filename(entry: dict, prefer: str):
    if prefer == "detection_area":
        return entry.get("detection_area") or entry.get("full_frame")
    return entry.get("full_frame") or entry.get("detection_area")


def run_ffmpeg_concat(frames_dir: str, out_path: str, fps: float, canvas: str):
    ensure_ffmpeg_or_exit()

    if fps <= 0:
        raise ValueError("--fps must be > 0")

    w, h = parse_canvas(canvas)

    ffconcat_path = os.path.join(frames_dir, "frames.ffconcat")
    files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    if not files:
        raise RuntimeError("No frames downloaded (frame_* files missing).")

    dur = 1.0 / fps
    with open(ffconcat_path, "w", encoding="utf-8") as fp:
        fp.write("ffconcat version 1.0\n")
        for f in files[:-1]:
            fp.write(f"file {f}\n")
            fp.write(f"duration {dur:.6f}\n")
        fp.write(f"file {files[-1]}\n")
        fp.write(f"file {files[-1]}\n")

    # Normalize variable-size crops to a fixed canvas with scale+pad.
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
        f"format=yuv420p"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", ffconcat_path,
        "-vf", vf,
        "-c:v", "libx264",
        out_path,
    ]
    subprocess.check_call(cmd)


def build_index_rows(detections, server_day, today_only: bool, finalized_only: bool):
    rows = []
    skipped_parse = 0

    for item in detections:
        uuid_str = item.get("uuid")
        if not uuid_str:
            continue

        finalized = bool(item.get("finalized", False))
        if finalized_only and not finalized:
            continue

        try:
            first_dt, latest_dt = parse_first_latest_from_item(item)
        except Exception:
            skipped_parse += 1
            continue

        if today_only and first_dt.date() != server_day:
            continue

        count = item.get("count", 0)
        try:
            count = int(count)
        except Exception:
            pass

        rows.append((latest_dt, first_dt, count, finalized, uuid_str, item))

    rows.sort(key=lambda t: t[0], reverse=True)
    return rows, skipped_parse


def print_index(rows, limit: int, as_json: bool):
    rows = rows[:limit] if (limit and limit > 0) else rows

    if as_json:
        for latest_dt, first_dt, count, finalized, uuid_str, _ in rows:
            out = {
                "uuid": uuid_str,
                "first_timestamp": first_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "latest_timestamp": latest_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "count": count,
                "finalized": finalized,
            }
            print(json.dumps(out, ensure_ascii=False))
        return

    print(f"{'latest':19}  {'first':19}  {'count':5}  {'F/O':3}  uuid")
    print("-" * 90)
    for latest_dt, first_dt, count, finalized, uuid_str, _ in rows:
        flag = "F" if finalized else "O"
        latest_s = latest_dt.strftime("%Y-%m-%d %H:%M:%S")
        first_s = first_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{latest_s:19}  {first_s:19}  {str(count):>5}  {flag:>3}  {uuid_str}")


def resolve_out_path(project_root: str, uuid_str: str, out_arg: str | None) -> str:
    """
    Rule you asked for:
      - default output always goes under <project_root>/data/
      - if --out is relative, it is also treated as relative to <project_root>/data/
      - if --out is absolute, we respect it
      - if --out points to a directory, we drop <uuid>_out.mp4 inside it
    """
    ddir = data_dir(project_root)
    os.makedirs(ddir, exist_ok=True)

    default_name = f"{uuid_str}_out.mp4"

    if not out_arg:
        return os.path.abspath(os.path.join(ddir, default_name))

    out_arg = os.path.expanduser(out_arg)

    # Absolute path: respect it
    if os.path.isabs(out_arg):
        if os.path.isdir(out_arg):
            return os.path.abspath(os.path.join(out_arg, default_name))
        os.makedirs(os.path.dirname(out_arg) or ".", exist_ok=True)
        return os.path.abspath(out_arg)

    # Relative path: treat relative to data/
    # Special case: user already prefixes "data/..." -> treat relative to project root
    if out_arg == "data" or out_arg.startswith("data" + os.sep):
        p = os.path.join(project_root, out_arg)
    else:
        p = os.path.join(ddir, out_arg)

    if p.endswith(os.sep) or os.path.isdir(p):
        os.makedirs(p, exist_ok=True)
        return os.path.abspath(os.path.join(p, default_name))

    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return os.path.abspath(p)


def resolve_frames_dir(project_root: str, uuid_str: str, workdir_arg: str | None) -> str:
    """
    Default frames dir goes under <project_root>/data/frames_<uuid>
    If --workdir is relative, treat it as relative to <project_root>/data/
    If --workdir is absolute, respect it.
    """
    ddir = data_dir(project_root)
    os.makedirs(ddir, exist_ok=True)

    if not workdir_arg:
        p = os.path.join(ddir, f"frames_{uuid_str}")
        os.makedirs(p, exist_ok=True)
        return os.path.abspath(p)

    workdir_arg = os.path.expanduser(workdir_arg)

    if os.path.isabs(workdir_arg):
        os.makedirs(workdir_arg, exist_ok=True)
        return os.path.abspath(workdir_arg)

    # Relative -> data/
    p = os.path.join(ddir, workdir_arg)
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)


def main():
    argv = sys.argv[1:]
    base_url_given = any(a == "--base-url" or a.startswith("--base-url=") for a in argv)
    help_requested = ("-h" in argv) or ("--help" in argv)

    ap = argparse.ArgumentParser(
        description="Fetch a YOLO detection aggregation batch and render a stop-motion MP4, or list batches."
    )

    ap.add_argument(
        "--base-url",
        default=BASE_URL_DEFAULT,
        help=f'Example: "http://127.0.0.1:5000" (default: {BASE_URL_DEFAULT})',
    )

    ap.add_argument("--list", action="store_true", help="List batch UUIDs and timestamps (no downloads, no ffmpeg).")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,  # default ALL for today
        help="Max rows to print in --list mode. Default 0 = all. (Use e.g. 30 to limit output.)",
    )
    ap.add_argument("--all-days", action="store_true", help="In --list mode, show batches across all days.")
    ap.add_argument("--json", action="store_true", help="In --list mode, output one JSON object per line.")

    ap.add_argument("--uuid", default=None, help="If set, use this batch UUID directly (skips today-filter).")
    ap.add_argument("--prefer", choices=["detection_area", "full_frame"], default="detection_area")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument(
        "--canvas",
        default=CANVAS_DEFAULT,
        help=f'Output canvas WxH for video (default "{CANVAS_DEFAULT}"). Example: "640x360".',
    )
    ap.add_argument(
        "--finalized-only",
        action="store_true",
        help="Only consider finalized batches (works for list and auto-pick).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help='Output MP4 path. If omitted, defaults to "<project_root>/data/<uuid>_out.mp4". '
             'If relative, it is treated as relative to "<project_root>/data/".',
    )
    ap.add_argument(
        "--workdir",
        default=None,
        help='Directory to store frames. Default: "<project_root>/data/frames_<uuid>/". '
             'If relative, it is treated as relative to "<project_root>/data/".',
    )
    args = ap.parse_args()

    if (not help_requested) and (not base_url_given):
        print(f'NOTE: no --base-url given, trying default ("{BASE_URL_DEFAULT}")', file=sys.stderr)

    base = args.base_url.rstrip("/")

    project_root = project_root_from_utils()
    ddir = data_dir(project_root)
    os.makedirs(ddir, exist_ok=True)

    server_time = http_get_json(f"{base}/api/current_time").get("current_time")
    if not server_time:
        raise RuntimeError("Server did not return current_time.")
    server_dt = parse_dt(server_time)
    server_day = server_dt.date()

    detections = http_get_json(f"{base}/api/detections")

    if args.list:
        today_only = not args.all_days
        rows, skipped_parse = build_index_rows(
            detections=detections,
            server_day=server_day,
            today_only=today_only,
            finalized_only=args.finalized_only,
        )

        if today_only:
            print(f"# Server day: {server_day} (from /api/current_time = {server_time})")
        else:
            print(f"# Server now: {server_time} (listing all days)")

        if skipped_parse:
            print(f"# NOTE: skipped {skipped_parse} entries due to unparsable timestamps", file=sys.stderr)

        print_index(rows, limit=args.limit, as_json=args.json)
        return

    chosen = None

    if args.uuid:
        for item in detections:
            if item.get("uuid") == args.uuid:
                if args.finalized_only and not item.get("finalized", False):
                    raise RuntimeError(
                        f"UUID {args.uuid} exists but is not finalized (use without --finalized-only)."
                    )
                chosen = item
                break
        if not chosen:
            raise RuntimeError(f"UUID not found in /api/detections: {args.uuid}")
    else:
        candidates = []
        skipped_parse = 0

        for item in detections:
            uuid_str = item.get("uuid")
            if not uuid_str:
                continue
            if args.finalized_only and not item.get("finalized", False):
                continue

            try:
                first_dt, latest_dt = parse_first_latest_from_item(item)
            except Exception:
                skipped_parse += 1
                continue

            if first_dt.date() == server_day:
                candidates.append((latest_dt, item))

        if not candidates:
            raise RuntimeError(f"No 'today' batches found for server-day {server_day}.")

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]

        if skipped_parse:
            print(f"NOTE: skipped {skipped_parse} entries due to unparsable timestamps", file=sys.stderr)

    uuid_str = chosen["uuid"]
    count = chosen.get("count", "?")
    first_ts = chosen.get("first_timestamp", "?")
    latest_ts = chosen.get("latest_timestamp", "?")
    finalized = chosen.get("finalized", False)

    print(f"Picked batch UUID: {uuid_str}")
    print(f"  count={count}, finalized={finalized}")
    print(f"  first={first_ts}, latest={latest_ts}")

    images = http_get_json(f"{base}/api/detection_images/{uuid_str}")
    if not isinstance(images, list) or not images:
        raise RuntimeError("Batch has no images (empty /api/detection_images response).")

    frames_dir = resolve_frames_dir(project_root, uuid_str, args.workdir)

    saved = 0
    manifest = {
        "base_url": base,
        "uuid": uuid_str,
        "prefer": args.prefer,
        "fps": args.fps,
        "canvas": args.canvas,
        "picked": {
            "count": count,
            "first_timestamp": first_ts,
            "latest_timestamp": latest_ts,
            "finalized": finalized,
        },
        "frames": [],
    }

    for i, entry in enumerate(images, start=1):
        if not isinstance(entry, dict):
            continue
        fname = pick_filename(entry, args.prefer)
        if not fname:
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
            ext = ".jpg"

        local_name = f"frame_{i:06d}{ext}"
        local_path = os.path.join(frames_dir, local_name)

        quoted = quote(fname, safe="/")
        url = f"{base}/api/detections/{quoted}"

        try:
            blob = http_get_bytes(url)
        except Exception as e:
            print(f"WARNING: failed to fetch {fname}: {e}", file=sys.stderr)
            continue

        with open(local_path, "wb") as fp:
            fp.write(blob)

        manifest["frames"].append({"index": i, "source": fname, "saved_as": local_name})
        saved += 1

    if saved < 2:
        raise RuntimeError(f"Downloaded only {saved} frames -- not enough to make a video.")

    out_path = resolve_out_path(project_root, uuid_str, args.out)
    manifest["out"] = os.path.relpath(out_path, project_root)

    with open(os.path.join(frames_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    run_ffmpeg_concat(frames_dir, out_path, args.fps, args.canvas)

    print(f"OK: wrote {out_path}")
    print(f"Frames + manifest in: {frames_dir}")


if __name__ == "__main__":
    main()
