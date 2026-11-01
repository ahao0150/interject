# -*- coding: utf-8 -*-
"""连拍：在「插话已投递、正在等句子边界」的那一两秒里抓几张。

需要 workspace 里的 computer-use 依赖（scapkit_computer_use）：
    python assets/burst.py --out assets/_burst --count 8 --interval 0.25
"""
import argparse
import asyncio
import pathlib

import scapkit_computer_use as cu


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/_burst")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--interval", type=float, default=0.25)
    args = ap.parse_args()

    displays = cu.list_displays()
    target = next((d for d in displays if d.get("is_main")), displays[0])
    handle = await cu.start_capture(target["id"])
    stem = pathlib.Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        first = None
        for _ in range(25):
            first = await cu.current_frame_jpg(handle, quality=92)
            if first:
                break
            await asyncio.sleep(0.2)
        if not first:
            print("FAIL: no frame")
            return 1
        for i in range(args.count):
            jpg = await cu.current_frame_jpg(handle, quality=92)
            if jpg:
                (stem.parent / f"{stem.name}-{i:02d}.jpg").write_bytes(jpg)
            await asyncio.sleep(args.interval)
    finally:
        await cu.stop_capture(handle)
    print(f"saved {args.count} frames to {stem.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
