# -*- coding: utf-8 -*-
"""在「句子正中间」投递插话，才能截到等待状态。

监听 SSE 的 token 事件，只有当最后一个字符是普通汉字（既不是句号也不是逗号）时才投递，
这样模型必然要等到下一个断句点才插入，界面上会停在「等 AI 写完这句再插」。
"""
import argparse
import asyncio
import json
import pathlib
import threading
import time
import urllib.request

import scapkit_computer_use as cu

STOP_CHARS = set("。！？!?…；;.，、：,:：\n ")


def focus_window(title_part: str = "interject") -> bool:
    import ctypes
    import ctypes.wintypes as wt

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
    ap.add_argument("--prompt", default="写一段关于秋天景色的描写")
    ap.add_argument("--text", default="顺便说一下，改用更口语、更随意的方式写")
    ap.add_argument("--out", default="assets/_mid")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    print("focus:", focus_window())
    time.sleep(0.4)
    post(base, "/api/chat/reset", {})
    time.sleep(0.3)

    latest = {"text": "", "hit": False}

    def listen():
        with urllib.request.urlopen(base + "/api/chat/stream", timeout=120) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "token":
                    t = ev.get("text") or ""
                    latest["text"] = t
                    if t and t[-1] not in STOP_CHARS and len(t) > 20:
                        latest["hit"] = True
                if ev.get("type") == "idle":
                    return

    threading.Thread(target=listen, daemon=True).start()
    time.sleep(1.0)
    post(base, "/api/chat/send", {"text": args.prompt, "max_tokens": 700})

    # 等模型写在句子中间的那一刻
    deadline = time.time() + 30
    while time.time() < deadline and not latest["hit"]:
        time.sleep(0.05)
    print("mid-sentence at:", repr(latest["text"][-30:]))

    async def shoot():
        d = cu.list_displays()
        t = next((x for x in d if x.get("is_main")), d[0])
        h = await cu.start_capture(t["id"])
        try:
            for _ in range(25):
                if await cu.current_frame_jpg(h, quality=92):
                    break
                await asyncio.sleep(0.1)
            post(base, "/api/chat/send", {"text": args.text, "max_tokens": 700})
            print("injected")
            stem = pathlib.Path(args.out)
            for i in range(14):
                jpg = await cu.current_frame_jpg(h, quality=92)
                if jpg:
                    (stem.parent / f"{stem.name}-{i:02d}.jpg").write_bytes(jpg)
                await asyncio.sleep(0.2)
        finally:
            await cu.stop_capture(h)

    asyncio.run(shoot())
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
