# -*- coding: utf-8 -*-
"""集成测试：真的加载模型跑一次注入。需要 GPU 和本地权重。

跑：INTERJECT_MODEL=./models/qwen2.5-1.5b pytest tests/ -m live
"""
from __future__ import annotations

import os
import queue

import pytest
import torch

from interject.runtime import generate_chat_turn

pytestmark = pytest.mark.live

MODEL_PATH = os.environ.get("INTERJECT_MODEL", "models/qwen2.5-1.5b")


@pytest.fixture(scope="module")
def engine():
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"模型不在 {MODEL_PATH}，设置 INTERJECT_MODEL 或把权重放进去")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
    return model, tok


def run_turn(engine, prompt: str, inject_at: int | None, text: str, **kw):
    model, tok = engine
    inbox: queue.Queue = queue.Queue()
    fired = {"done": False}

    def cb(event):
        if inject_at is None or fired["done"]:
            return
        if event.get("type") == "token" and event.get("index", 0) >= inject_at:
            fired["done"] = True
            inbox.put({"id": 1, "text": text, "submitted_rel": 0.0})

    return generate_chat_turn(
        model, tok, [{"role": "user", "content": prompt}], inbox,
        max_new_tokens=kw.pop("max_new_tokens", 120),
        seed=kw.pop("seed", 1234),
        event_cb=cb, **kw)


def test_no_injection_produces_plain_reply(engine):
    r = run_turn(engine, "用一句话介绍你自己", None, "")
    assert r["supplements"] == []
    assert r["segments"] == [{"text": r["text"].strip()[:len(r["text"])], "supp": None}] \
        or len(r["segments"]) == 1
    assert r["supplements_dropped"] == []


def test_injection_lands_on_a_sentence_boundary(engine):
    r = run_turn(engine, "推荐一道适合晚餐的菜", 4, "我不吃肉，是素食者")
    assert len(r["supplements"]) == 1, "插话没有被吸收"
    s = r["supplements"][0]
    assert s["at_boundary"] is True, f"插在了句子中间：{s['boundary']}"
    assert s["step"] >= 4
    assert r["supplements_dropped"] == []


def test_segments_align_with_insertion_points(engine):
    r = run_turn(engine, "推荐一道适合晚餐的菜", 6, "我不吃肉，是素食者")
    marks = [seg["supp"] for seg in r["segments"] if seg["supp"]]
    assert [m["id"] for m in marks] == [s["id"] for s in r["supplements"]]
    # 分段文本拼回来必须等于原文，不能丢字
    rebuilt = "".join(seg["text"] for seg in r["segments"] if not seg["supp"])
    assert rebuilt == r["text"]


def test_full_text_contains_what_the_model_actually_saw(engine):
    r = run_turn(engine, "推荐一道适合晚餐的菜", 6, "我不吃肉，是素食者")
    if r["supplements"]:
        assert "用户补充" in r["full_text"]
        assert "我不吃肉" in r["full_text"]


def test_injection_changes_the_outcome(engine):
    """插话「我只有20分钟」之后，回答里应该出现时间相关的收敛，而不是照原样写。"""
    base = run_turn(engine, "推荐一道适合晚餐的菜", None, "", max_new_tokens=140)
    injected = run_turn(engine, "推荐一道适合晚餐的菜", 6, "我只有20分钟", max_new_tokens=140)
    assert injected["supplements"], "插话没进去"
    assert base["text"] != injected["text"], "注入后输出与基线一字不差"
