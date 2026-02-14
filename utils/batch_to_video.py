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
CANVAS_PORTRAIT_DEFAULT = "720x1280"
CANVAS_LANDSCAPE_DEFAULT = "1280x720"
CANVAS_DEFAULT = CANVAS_PORTRAIT_DEFAULT
FIT_DEFAULT = "blur"  # "blur" = fills portrait canvas without cropping content

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


def _evenize(n: int) -> int:
    return n if (n % 2 == 0) else (n + 1)


# --- Image size parsing (stdlib-only) ----------------------------------------

def _png_size(path: str):
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return None
        # PNG: after signature, first chunk should be IHDR
        _len = f.read(4)
        ctype = f.read(4)
        if ctype != b"IHDR":
            return None
        w = int.from_bytes(f.read(4), "big")
        h = int.from_bytes(f.read(4), "big")
        if w > 0 and h > 0:
            return w, h
    return None


def _jpeg_size(path: str):
    # Parse JPEG SOF marker for width/height
    with open(path, "rb") as f:
        if f.read(2) != b"\xFF\xD8":
            return None

        def _read1():
            b = f.read(1)
            return b[0] if b else None

        while True:
            b = _read1()
            if b is None:
                return None
            if b != 0xFF:
                continue

            # skip fill 0xFF bytes
            marker = _read1()
            while marker == 0xFF:
                marker = _read1()
            if marker is None:
                return None

            # stand-alone markers
            if marker in (0xD8, 0xD9):  # SOI/EOI
                continue

            # read segment length
            seglen_bytes = f.read(2)
            if len(seglen_bytes) != 2:
                return None
            seglen = int.from_bytes(seglen_bytes, "big")
            if seglen < 2:
                return None

            # SOF markers that contain size
            if marker in (
                0xC0, 0xC1, 0xC2, 0xC3,
                0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB,
                0xCD, 0xCE, 0xCF
            ):
                # precision (1), height (2), width (2)
                data = f.read(5)
                if len(data) != 5:
                    return None
                h = int.from_bytes(data[1:3], "big")
                w = int.from_bytes(data[3:5], "big")
                if w > 0 and h > 0:
                    return w, h
                return None

            # skip segment payload (we already consumed 2 length bytes)
            f.seek(seglen - 2, os.SEEK_CUR)


def _webp_size(path: str):
    # WebP RIFF container: VP8 / VP8L / VP8X
    with open(path, "rb") as f:
        hdr = f.read(30)
        if len(hdr) < 16:
            return None
        if hdr[0:4] != b"RIFF" or hdr[8:12] != b"WEBP":
            return None

        chunk = hdr[12:16]
        # chunk size at 16:20, data starts at 20
        # We may need more bytes depending on chunk type
        f.seek(12)
        riff = f.read(4096)  # enough for headers in practice
        if len(riff) < 32:
            return None

        chunk = riff[12:16]
        if chunk == b"VP8X":
            # Extended WebP: width-1 at bytes 24..26, height-1 at 27..29
            if len(riff) < 30:
                return None
            w = 1 + (riff[24] | (riff[25] << 8) | (riff[26] << 16))
            h = 1 + (riff[27] | (riff[28] << 8) | (riff[29] << 16))
            return (w, h) if (w > 0 and h > 0) else None

        if chunk == b"VP8L":
            # Lossless WebP: signature 0x2f then 4 bytes with packed dims
            # data begins at 20; signature at 20
            if len(riff) < 25:
                return None
            if riff[20] != 0x2F:
                return None
            bits = int.from_bytes(riff[21:25], "little")
            w = 1 + (bits & 0x3FFF)
            h = 1 + ((bits >> 14) & 0x3FFF)
            return (w, h) if (w > 0 and h > 0) else None

        if chunk == b"VP8 ":
            # Lossy VP8: frame tag(3), start code(3) at data[3:6] == 0x9d 0x01 0x2a
            # then 2 bytes width, 2 bytes height with scaling bits
            if len(riff) < 30:
                return None
            data = riff[20:]  # chunk data start
            if len(data) < 10:
                return None
            if data[3:6] != b"\x9d\x01\x2a":
                return None
            w = data[6] | ((data[7] & 0x3F) << 8)
            h = data[8] | ((data[9] & 0x3F) << 8)
            return (w, h) if (w > 0 and h > 0) else None

    return None


def image_size(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".png":
            return _png_size(path)
        if ext in (".jpg", ".jpeg"):
            return _jpeg_size(path)
        if ext == ".webp":
            return _webp_size(path)
    except Exception:
        return None
    return None


def choose_canvas_from_frames(frames_dir: str, fallback_canvas: str):
    """
    Find the largest downloaded frame by pixel area and return its WxH (evenized).
    If parsing fails for all frames, return fallback_canvas.
    Returns: (canvas_str, chosen_file_or_none)
    """
    files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    best = None  # (area, w, h, filename)

    for fn in files:
        p = os.path.join(frames_dir, fn)
        sz = image_size(p)
        if not sz:
            continue
        w, h = sz
        area = w * h
        if best is None or area > best[0]:
            best = (area, w, h, fn)

    if not best:
        return fallback_canvas, None

    _, w, h, fn = best
    w = _evenize(w)
    h = _evenize(h)
    return f"{w}x{h}", fn


# --- Timestamp parsing --------------------------------------------------------

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

    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
    except Exception:
        pass

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


# --- ffmpeg render ------------------------------------------------------------

def run_ffmpeg_concat(frames_dir: str, out_path: str, fps: float, canvas: str, fit_mode: str):
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

    # # Fit-to-screen: preserve aspect, no stretch; pad to canvas.
    # vf = (
    #     f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
    #     f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
    #     f"format=yuv420p"
    # )

    if fit_mode == "contain":
        # No crop, no stretch; may produce bars.
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,format=yuv420p"
        )
    elif fit_mode == "blur":
        # No crop of foreground. Background is stretched+blurred to fill canvas.
        vf = (
            f"split=2[bg][fg];"
            f"[bg]scale={w}:{h},gblur=sigma=24[bg];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,"
            f"setsar=1,format=yuv420p"
        )
    else:
        raise ValueError(f"bad --fit {fit_mode!r}")

    # Add a silent audio track for better messenger compatibility.
    # (Helps some clients classify it as "video" and generate previews consistently.)
    base_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", ffconcat_path,
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", vf,

        # “Boring MP4” defaults for broad compatibility:
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",

        "-c:a", "aac",
        "-b:a", "96k",
        "-shortest",

        # Put moov atom at the front (important for previews/streaming).
        "-movflags", "+faststart",
    ]

    # Prefer CFR to avoid weird “tbr 25” style metadata confusing clients.
    cmd1 = base_cmd + ["-r", str(fps), "-fps_mode", "cfr", out_path]
    p = subprocess.run(cmd1, capture_output=True, text=True)

    if p.returncode != 0:
        # Older ffmpeg: -fps_mode might not exist -> fallback
        stderr = (p.stderr or "")
        if "Unrecognized option" in stderr and "fps_mode" in stderr:
            cmd2 = base_cmd + ["-r", str(fps), "-vsync", "cfr", out_path]
            p2 = subprocess.run(cmd2, capture_output=True, text=True)
            if p2.returncode != 0:
                raise RuntimeError(f"ffmpeg failed:\n{p2.stderr or p2.stdout}")
        else:
            raise RuntimeError(f"ffmpeg failed:\n{stderr or p.stdout}")

# --- Listing helpers ----------------------------------------------------------

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


# --- Path resolution ----------------------------------------------------------

def resolve_out_path(project_root: str, uuid_str: str, out_arg: str | None) -> str:
    ddir = data_dir(project_root)
    os.makedirs(ddir, exist_ok=True)

    default_name = f"{uuid_str}_out.mp4"

    if not out_arg:
        return os.path.abspath(os.path.join(ddir, default_name))

    out_arg = os.path.expanduser(out_arg)

    if os.path.isabs(out_arg):
        if os.path.isdir(out_arg):
            return os.path.abspath(os.path.join(out_arg, default_name))
        os.makedirs(os.path.dirname(out_arg) or ".", exist_ok=True)
        return os.path.abspath(out_arg)

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

    p = os.path.join(ddir, workdir_arg)
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)

def main():
    argv = sys.argv[1:]
    base_url_given = any(a == "--base-url" or a.startswith("--base-url=") for a in argv)
    canvas_given = any(a == "--canvas" or a.startswith("--canvas=") for a in argv)
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
        default=0,
        help="Max rows to print in --list mode. Default 0 = all. (Use e.g. 30 to limit output.)",
    )
    ap.add_argument("--all-days", action="store_true", help="In --list mode, show batches across all days.")
    ap.add_argument("--json", action="store_true", help="In --list mode, output one JSON object per line.")

    ap.add_argument("--uuid", default=None, help="If set, use this batch UUID directly (skips today-filter).")
    ap.add_argument("--prefer", choices=["detection_area", "full_frame"], default="detection_area")
    ap.add_argument("--fps", type=float, default=8.0)

    # Canvas / orientation presets
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--portrait",
        action="store_true",
        help=f'Shortcut for --canvas "{CANVAS_PORTRAIT_DEFAULT}" (default).',
    )
    g.add_argument(
        "--landscape",
        action="store_true",
        help=f'Shortcut for --canvas "{CANVAS_LANDSCAPE_DEFAULT}".',
    )

    ap.add_argument(
        "--canvas",
        default=CANVAS_DEFAULT,
        help=f'Output canvas WxH for video (default "{CANVAS_DEFAULT}"). Example: "1080x1920".',
    )

    ap.add_argument(
        "--fit",
        choices=["contain", "blur"],
        default=FIT_DEFAULT,
        help='Fitting mode. "contain" = no crop, may add bars. '
             '"blur" = no crop of the real frame; fills canvas using a blurred background copy (default).',
    )

    # Auto-canvas from largest frame (optional)
    ap.add_argument(
        "--fit-largest",
        action="store_true",
        dest="fit_largest",
        help='Auto-pick canvas from the largest downloaded frame (overrides default canvas unless --canvas/--portrait/--landscape was explicitly set).',
    )
    ap.add_argument(
        "--no-fit-largest",
        action="store_false",
        dest="fit_largest",
        help="Disable auto-canvas from largest frame.",
    )
    ap.set_defaults(fit_largest=False)

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

    # Apply portrait/landscape shortcuts. These count as "canvas explicitly set".
    if args.portrait:
        args.canvas = CANVAS_PORTRAIT_DEFAULT
        canvas_given = True
    elif args.landscape:
        args.canvas = CANVAS_LANDSCAPE_DEFAULT
        canvas_given = True

    if (not help_requested) and (not base_url_given):
        print(f'NOTE: no --base-url given, trying default ("{BASE_URL_DEFAULT}")', file=sys.stderr)

    base = args.base_url.rstrip("/")

    project_root = project_root_from_utils()
    os.makedirs(data_dir(project_root), exist_ok=True)

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
        "fit": args.fit,
        "uuid": uuid_str,
        "prefer": args.prefer,
        "fps": args.fps,
        "canvas": args.canvas,  # will be overwritten by auto-canvas if enabled
        "fit_largest": bool(args.fit_largest),
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

    effective_canvas = args.canvas
    if args.fit_largest and (not canvas_given):
        auto_canvas, chosen_frame = choose_canvas_from_frames(frames_dir, fallback_canvas=args.canvas)
        effective_canvas = auto_canvas
        manifest["canvas"] = effective_canvas
        manifest["auto_canvas_from"] = chosen_frame or None
        if chosen_frame:
            print(f"Auto-canvas: {effective_canvas} (from largest frame: {chosen_frame})", file=sys.stderr)
        else:
            print(f"Auto-canvas: could not parse frame sizes; using fallback canvas {effective_canvas}", file=sys.stderr)
    else:
        manifest["canvas"] = effective_canvas
        manifest["auto_canvas_from"] = None

    out_path = resolve_out_path(project_root, uuid_str, args.out)
    manifest["out"] = os.path.relpath(out_path, project_root)

    with open(os.path.join(frames_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    run_ffmpeg_concat(frames_dir, out_path, args.fps, effective_canvas, args.fit)

    print(f"OK: wrote {out_path}")
    print(f"Frames + manifest in: {frames_dir}")


if __name__ == "__main__":
    main()
