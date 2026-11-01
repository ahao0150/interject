# -*- coding: utf-8 -*-
"""演示驱动：给截图和录屏造一段「写诗途中改成散文」的完整会话。

用法：
    python assets/drive_demo.py --port 8101 --mode full
    python assets/drive_demo.py --port 8101 --mode waiting   # 停在「等句子边界」
"""
import argparse
import json
import threading
import time
import urllib.request


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def wait_idle(base, timeout=180):
    """等这一轮生成结束。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(base + "/api/chat/state", timeout=10) as r:
            if not json.loads(r.read().decode())["busy"]:
                return True
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--mode", default="full", choices=["full", "waiting"])
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--prompt", default="写一首关于秋天的诗")
    ap.add_argument("--text", default="停一下，不要写诗了，直接写一篇关于秋天的散文，现在就写，别问我")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    post(base, "/api/chat/reset", {})
    time.sleep(0.4)
    post(base, "/api/chat/send", {"text": args.prompt, "max_tokens": 700})
    print(f"[demo] 发送：{args.prompt}", flush=True)

    time.sleep(args.delay)
    post(base, "/api/chat/send", {"text": args.text, "max_tokens": 700})
    print(f"[demo] 插入：{args.text}", flush=True)

    if args.mode == "waiting":
        time.sleep(0.35)          # 停在「等句子边界」的那一小段
        print("[demo] 保持在等待状态", flush=True)
        return

    wait_idle(base)
    print("[demo] 本轮结束", flush=True)


if __name__ == "__main__":
    main()
