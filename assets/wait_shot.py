# -*- coding: utf-8 -*-
"""拍「插话已投递、正在等句子边界」的那一瞬间。

把前置窗口、连拍、投递放进同一个进程，时间才卡得准。

    python assets/wait_shot.py --port 8101 --out assets/_wait
"""
import argparse
import asyncio
import ctypes
import ctypes.wintypes as wt
import json
import pathlib
import threading
import time
import urllib.request

import scapkit_computer_use as cu


def focus_window(title_part: str = "interject") -> bool:
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def enum_proc(hwnd, _lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 2)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 2)
            if title_part in buf.value:
                found.append(hwnd)
        return True

    ctypes.windll.user32.EnumWindows(enum_proc, 0)
    if not found:
        return False
    hwnd = found[0]
    ctypes.windll.user32.ShowWindow(hwnd, 3)      # maximize + activate
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    return True


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


async def capture_loop(handle, stem: pathlib.Path, count: int, interval: float):
    for i in range(count):
        jpg = await cu.current_frame_jpg(handle, quality=92)
        if jpg:
            (stem.parent / f"{stem.name}-{i:02d}.jpg").write_bytes(jpg)
        await asyncio.sleep(interval)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--out", default="assets/_wait")
    ap.add_argument("--count", type=int, default=14)
    ap.add_argument("--interval", type=float, default=0.22)
    ap.add_argument("--delay-before-inject", type=float, default=0.9,
                    help="开拍后多久投递插话")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    stem = pathlib.Path(args.out)

    print("focus:", focus_window())
    time.sleep(0.5)

    post(base, "/api/chat/reset", {})
    time.sleep(0.3)
    post(base, "/api/chat/send", {"text": "写一首关于秋天的诗", "max_tokens": 700})
    time.sleep(2.0)                       # 让它先写几行

    displays = cu.list_displays()
    target = next((d for d in displays if d.get("is_main")), displays[0])
    handle = await cu.start_capture(target["id"])
    try:
        for _ in range(25):
            if await cu.current_frame_jpg(handle, quality=92):
                break
            await asyncio.sleep(0.15)

        task = asyncio.create_task(capture_loop(handle, stem, args.count, args.interval))
        await asyncio.sleep(args.delay_before_inject)
        post(base, "/api/chat/send",
             {"text": "停一下，不要写诗了，直接写一篇关于秋天的散文，现在就写，别问我",
              "max_tokens": 700})
        print("injected")
        await task
    finally:
        await cu.stop_capture(handle)
    print(f"saved {args.count} frames")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
