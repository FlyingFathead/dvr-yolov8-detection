# dvr_recorder.py
"""
Light‑weight DVR module for dvr‑yolov8‑detection.

Records the raw RTMP stream (video **and** audio) into timestamped files
whenever people are being detected, and automatically closes the file
once no detections have happened for `dvr_record_duration` seconds.

Designed to run in its **own** thread so it never blocks frame capture,
image saving, remote sync, or the web‑server.

Requirements
------------
* FFmpeg available in `$PATH` (or override with `[dvr] ffmpeg_path`).
* Works with any input FFmpeg can open (RTMP/RTSP/…); the command uses
  `-c copy` so **no re‑encoding** – cheap on CPU and it preserves the
  original audio if present.

Usage
-----
```python
from dvr_recorder import DVRRecorder

# existing Event from the detector – stays set while detections exist
from yolov8_live_rtmp_stream_detection import detection_ongoing

rec = DVRRecorder(config, stream_url, detection_ongoing, base_save_dir, logger)
rec.start()   # starts internal thread – returns immediately
...
rec.stop()    # clean shutdown (SIGINT to ffmpeg, joins thread)
```
"""
from __future__ import annotations

import os
import threading
import subprocess
import time
from datetime import datetime
import signal
import logging
import shutil
import queue
from pathlib import Path

__all__ = ["DVRRecorder"]

# ---------------------------------------------------------------------------
class DVRRecorder:
    """High‑level wrapper around one FFmpeg process.

    Parameters
    ----------
    cfg : configparser.ConfigParser
        Parsed *global* config – we only read the `[dvr]` section in here.
    stream_url : str
        URL that FFmpeg should grab (typically the same RTMP URL the
        detection pipeline consumes).
    detection_event : threading.Event
        Set while at least one detection is active, cleared after the
        detector’s *cool‑down* considers the scene clear.
    base_save_dir : str | Path
        Points to the directory that already contains the still‑image
        hierarchy (`yolo_detections/`). A sub‑directory `video/` will be
        created next to it using the same date‑based layout.
    logger : logging.Logger
        Use the project‑wide logger so everything lands in the same files.
    """

    def __init__(
        self,
        cfg,
        stream_url: str,
        detection_event: threading.Event,
        base_save_dir: str | os.PathLike,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.stream_url = stream_url
        self.detection_event = detection_event
        self.base_save_dir = Path(base_save_dir)
        self.log = logger or logging.getLogger("dvr")

        # ---- [dvr] section -------------------------------------------------
        self.enabled = cfg.getboolean("dvr", "dvr_enabled", fallback=False)
        self.gap = cfg.getint("dvr", "dvr_record_duration", fallback=30)
        self.ffmpeg = cfg.get("dvr", "ffmpeg_path", fallback="ffmpeg")

        # Directories --------------------------------------------------------
        self.default_dir = Path(cfg.get("dvr", "default_video_save_dir", fallback=str(self.base_save_dir / "video")))
        self.fallback_dir = Path(cfg.get("dvr", "fallback_video_save_dir", fallback=str(Path.cwd() / "yolo_detections" / "video")))

        env_var = cfg.get("dvr", "env_video_dir_var", fallback="YOLO_VIDEO_SAVE_DIR")
        self.use_env = cfg.getboolean("dvr", "use_env_save_dir", fallback=True)
        env_val = os.getenv(env_var) if self.use_env else None

        self.save_root = self._pick_save_root(env_val)
        self.log.info(f"DVR save root: {self.save_root}")

        # Thread / state -----------------------------------------------------
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="DVRRecorder", daemon=True)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API
    def start(self):
        if not self.enabled:
            self.log.info("[dvr] disabled – not starting DVR thread")
            return
        if not self._thread.is_alive():
            self._thread.start()
            self.log.info("[dvr] thread started")

    def stop(self):
        self._stop.set()
        self.detection_event.set()  # wake the worker if it’s sitting in wait()
        if self._thread.is_alive():
            self._thread.join()
        self._terminate_ffmpeg()
        self.log.info("[dvr] stopped cleanly")

    # ---------------------------------------------------------------- utils
    def _pick_save_root(self, env_val):
        candidates = []
        if env_val:
            candidates.append(Path(env_val))
        candidates.append(self.default_dir)
        candidates.append(self.fallback_dir)
        for p in candidates:
            try:
                p.mkdir(parents=True, exist_ok=True)
                if os.access(p, os.W_OK):
                    return p.resolve()
            except Exception:
                continue
        raise RuntimeError("No writable DVR save directory found")

    @staticmethod
    def _date_subdir(root: Path) -> Path:
        today = datetime.now()
        sub = root / today.strftime("%Y") / today.strftime("%m") / today.strftime("%d")
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def _current_filename(self):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{ts}.mp4"  # container – copy‑codec so extension doesn’t matter much

    # --------------------------------------------------------------- thread
    def _worker(self):
        """Main loop – waits for `detection_event`, starts/stops FFmpeg."""
        idle_since: float | None = None
        while not self._stop.is_set():
            # Wait until a detection begins, but wake every 0.5 s to check stop flag
            if not self.detection_event.wait(0.5):
                continue

            if self._proc is None:
                try:
                    self._start_ffmpeg()
                except Exception as e:
                    self.log.error(f"[dvr] failed to start ffmpeg: {e}")
                    idle_since = None
                    # give up until next detection burst
                    self._drain_event()
                    continue

            # We’re recording right now; track inactivity
            if self.detection_event.is_set():
                idle_since = time.time()  # reset timer every loop while event is set
            else:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since >= self.gap:
                    # Enough silence – cut file
                    self._terminate_ffmpeg()
                    idle_since = None
            # Small sleep to avoid busy‑looping
            time.sleep(0.25)

        # got stop flag – close ffmpeg before exiting
        self._terminate_ffmpeg()

    def _drain_event(self):
        """Clear spurious set() calls so `wait()` blocks again."""
        while self.detection_event.is_set():
            time.sleep(0.1)

    # ------------------------------------------------------------ ffmpeg
    def _start_ffmpeg(self):
        out_dir = self._date_subdir(self.save_root)
        filename = self._current_filename()
        out_path = out_dir / filename

        cmd = [
            self.ffmpeg,
            "-y",                      # overwrite if ever collides (unlikely)
            "-hide_banner", "-loglevel", "error",
            "-i", self.stream_url,
            "-c", "copy",              # no CPU cost – keep original audio & video
            "-movflags", "+faststart", # good for progressive download in browsers
            str(out_path)
        ]
        # Start process in its own process‑group so we can SIGINT the whole thing
        self.log.info("[dvr] ⏺  Recording → %s", out_path)
        self._proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self._current_file = str(out_path)

    def _terminate_ffmpeg(self):
        with self._lock:
            if self._proc is None:
                return
            self.log.info("[dvr] ⏹  Closing file …")
            try:
                # Send SIGINT to the *group* so muxer flushes index cleanly
                os.killpg(self._proc.pid, signal.SIGINT)
                # Wait max 5 s
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log.warning("[dvr] ffmpeg didn’t exit – killing")
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            finally:
                self._proc = None
                self.log.info("[dvr] file closed")
