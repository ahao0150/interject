# -*- coding: utf-8 -*-
"""注入措辞对照：哪种写法能让模型「先交代、再照做、不停下」。

    python experiments/inject_style.py --model ./models/qwen2.5-1.5b

三种写法都放在插话里（紧贴用户补充），两个场景：早插（第 8 token）、极晚（第 100 token）。
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interject.runtime import generate_chat_turn

PROMPT = "写一首关于秋天的诗"
SWITCH = "停一下，不要写诗了，改成写一篇关于秋天的散文"
STYLES = ["plain", "brief", "strict"]
SCENES = [{"name": "早插·第8", "at": 8}, {"name": "极晚·第100", "at": 100}]
ACK = re.compile(r"(好的|明白|收到|了解|稍等|这就|改为|改成|切换|换成|根据.{0,6}要求)")
STALL = re.compile(r"(请稍等|我将为您|请告诉我你希望)")


def shape(text: str) -> dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lens = [len(ln) for ln in lines]
    return {"换行": text.count("\n"), "非空行": len(lines),
            "平均行长": round(statistics.mean(lens), 1) if lens else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-1.5b")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="inject_style.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

    rows = []
    for scene in SCENES:
        for style in STYLES:
            inbox: queue.Queue = queue.Queue()
            fired = {"done": False}

            def cb(event, at=scene["at"], box=inbox, state=fired):
                if state["done"]:
                    return
                if event.get("type") == "token" and event.get("index", 0) >= at:
                    state["done"] = True
                    box.put({"id": 0, "text": SWITCH, "submitted_rel": 0.0})

            r = generate_chat_turn(model, tok, [{"role": "user", "content": PROMPT}], inbox,
                                   max_new_tokens=240, seed=args.seed, event_cb=cb,
                                   supplement_style=style)
            chars = r["supplements"][0]["chars"] if r["supplements"] else 0
            after = r["text"][chars:]
            row = {
                "scene": scene["name"], "style": style,
                "step": r["supplements"][0]["step"] if r["supplements"] else None,
                "ack": bool(ACK.search(after[:100])),
                "stalled": bool(STALL.search(after[:100])),
                "shape": shape(r["text"]),
                "head": after[:60].replace("\n", " "),
            }
            print(f"{scene['name']:<12}{style:<8}step={row['step']} 交代={row['ack']} "
                  f"停在提问={row['stalled']} | {row['head']}", flush=True)
            rows.append(row)
        print("", flush=True)

    print("===== 摘要 =====")
    for sc in SCENES:
        for st in STYLES:
            r = next(x for x in rows if x["scene"] == sc["name"] and x["style"] == st)
            print(f"{sc['name']:<12}{st:<8}交代={'✓' if r['ack'] else '✗':<4}"
                  f"停在提问={'是' if r['stalled'] else '否':<6}平均行长={r['shape']['平均行长']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
