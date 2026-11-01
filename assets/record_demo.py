# -*- coding: utf-8 -*-
"""录一段完整的演示：写到第 N 个 token 时插话改成散文，全程录屏。

    python assets/record_demo.py --port 8101 --inject-at 45 --seconds 32
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import threading
import time
import urllib.request

FFMPEG = os.environ.get(
    "FFMPEG",
    r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe")


def focus_window(title_part: str = "interject") -> bool:
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(hwnd, _l):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 2)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 2)
            if title_part in buf.value:
                found.append(hwnd)
        return True

    ctypes.windll.user32.EnumWindows(cb, 0)
    if not found:
        return False
    ctypes.windll.user32.ShowWindow(found[0], 3)
    ctypes.windll.user32.SetForegroundWindow(found[0])
    return True


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--prompt", default="写一首关于秋天的诗")
    ap.add_argument("--text", default="停一下，不要写诗了，直接写一篇关于秋天的散文，现在就写，别问我")
    ap.add_argument("--inject-at", type=int, default=45)
    ap.add_argument("--seconds", type=int, default=32)
    ap.add_argument("--out", default="assets/demo.mp4")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    print("focus:", focus_window())
    time.sleep(0.5)
    post(base, "/api/chat/reset", {})
    time.sleep(0.5)

    ff = subprocess.Popen(
        [FFMPEG, "-y", "-f", "gdigrab", "-framerate", "12", "-i", "desktop",
         "-t", str(args.seconds), "-vf", "scale=1600:-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
         args.out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("recording…", flush=True)
    time.sleep(1.5)

    injected = {"done": False, "at": None}

    def listen():
        with urllib.request.urlopen(base + "/api/chat/stream", timeout=args.seconds + 20) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "token" and not injected["done"] and ev.get("index", 0) >= args.inject_at:
                    injected["done"] = True
                    injected["at"] = ev.get("index")
                    post(base, "/api/chat/send", {"text": args.text, "max_tokens": 700})
                    print(f"injected at token {injected['at']}", flush=True)
                if t == "supplement_taken":
                    print(f"absorbed at token {ev.get('step')} ({ev.get('boundary')})", flush=True)
                if t == "idle" and injected["done"]:
                    return

    threading.Thread(target=listen, daemon=True).start()
    time.sleep(0.3)
    post(base, "/api/chat/send", {"text": args.prompt, "max_tokens": 700})
    print("prompt sent", flush=True)

    ff.wait(timeout=args.seconds + 20)
    print("recording finished:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
