# -*- coding: utf-8 -*-
"""任务切换实验：写诗途中插话改成散文 / 代码，看它会不会真的换活。

    python experiments/task_switch.py --model ./models/qwen2.5-1.5b

五组对照，输出「平均行长」作为文体指标：诗是短行（约 10 字/行），散文是长句（约 40）。
"""
from __future__ import annotations

import argparse
import json
import queue
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interject.runtime import generate_chat_turn

PROMPT = "写一首关于秋天的诗"
SWITCH_PROSE = "停一下，不要写诗了，改成写一篇关于秋天的散文"
SWITCH_CODE = "停一下，不要写诗了，写一段 Python 快速排序的代码"

GROUPS = [
    {"name": "A 基线·不插话", "at": None},
    {"name": "B 早插·第8token", "at": 8},
    {"name": "C 晚插·第45token", "at": 45},
    {"name": "D 极晚·第100token", "at": 100},
    {"name": "E 换领域·改代码", "at": 20, "text": SWITCH_CODE},
]


def shape(text: str) -> dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lens = [len(ln) for ln in lines]
    return {"换行数": text.count("\n"), "非空行数": len(lines),
            "平均行长": round(statistics.mean(lens), 1) if lens else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-1.5b")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--out", default="task_switch.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

    results = []
    for g in GROUPS:
        inbox: queue.Queue = queue.Queue()
        fired = {"done": False}
        switch_text = g.get("text", SWITCH_PROSE)

        def cb(event, at=g["at"], box=inbox, state=fired, sw=switch_text):
            if at is None or state["done"]:
                return
            if event.get("type") == "token" and event.get("index", 0) >= at:
                state["done"] = True
                box.put({"id": 0, "text": sw, "submitted_rel": 0.0})
                print(f"  [投递] 第 {event.get('index')} token：{sw[:20]}…", flush=True)

        print(f"===== {g['name']} =====", flush=True)
        r = generate_chat_turn(model, tok, [{"role": "user", "content": PROMPT}], inbox,
                               max_new_tokens=args.max_new_tokens, seed=args.seed, event_cb=cb)
        print(r["text"].strip(), flush=True)
        stats = shape(r["text"])
        print(f"\n  [形状] {json.dumps(stats, ensure_ascii=False)}", flush=True)
        print(f"  [插入] {[(s['step'], s['boundary']) for s in r['supplements']]}\n", flush=True)
        results.append({"group": g["name"], "text": r["text"], "shape": stats,
                        "supplements": r["supplements"]})

    print("===== 摘要 =====")
    for r in results:
        s = r["shape"]
        print(f"{r['group']:<18} 换行 {s['换行数']:>3}  非空行 {s['非空行数']:>3}  "
              f"平均行长 {s['平均行长']:>5}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
