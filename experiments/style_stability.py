# -*- coding: utf-8 -*-
"""稳定性验证：极晚插入时，plain 与 brief 谁能让模型交代。

    python experiments/style_stability.py --model ./models/qwen2.5-1.5b --seeds 1234 777 2026

单次采样可能是运气，看多个 seed 的分布。
"""
from __future__ import annotations

import argparse
import json
import queue
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interject.runtime import generate_chat_turn

PROMPT = "写一首关于秋天的诗"
SWITCH = "停一下，不要写诗了，改成写一篇关于秋天的散文"
ACK = re.compile(r"(好的|明白|收到|了解|改为|改成|切换|换成|根据.{0,6}要求)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-1.5b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 777, 2026])
    ap.add_argument("--inject-at", type=int, default=100)
    ap.add_argument("--out", default="style_stability.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

    rows = []
    for style in ("plain", "brief"):
        for seed in args.seeds:
            inbox: queue.Queue = queue.Queue()
            fired = {"done": False}

            def cb(event, at=args.inject_at, box=inbox, state=fired):
                if state["done"]:
                    return
                if event.get("type") == "token" and event.get("index", 0) >= at:
                    state["done"] = True
                    box.put({"id": 0, "text": SWITCH, "submitted_rel": 0.0})

            r = generate_chat_turn(model, tok, [{"role": "user", "content": PROMPT}], inbox,
                                   max_new_tokens=240, seed=seed, event_cb=cb,
                                   supplement_style=style)
            chars = r["supplements"][0]["chars"] if r["supplements"] else 0
            after = r["text"][chars:]
            ack = bool(ACK.search(after[:60]))
            print(f"{style:<6} seed={seed:<5} 交代={'✓' if ack else '✗'} | "
                  f"{after[:50].replace(chr(10), ' ')}", flush=True)
            rows.append({"style": style, "seed": seed, "ack": ack,
                         "head": after[:80].replace("\n", " ")})
        print("", flush=True)

    print("===== 汇总 =====")
    for style in ("plain", "brief"):
        sub = [r for r in rows if r["style"] == style]
        hit = sum(1 for r in sub if r["ack"])
        print(f"{style:<6} 交代 {hit}/{len(sub)}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
