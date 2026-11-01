# -*- coding: utf-8 -*-
"""纯逻辑测试：不需要模型、不需要 GPU、不需要网络。

跑：pytest tests/ -m "not live"
"""
from __future__ import annotations

import torch
import pytest

from interject.runtime import (
    SUPPLEMENT_STYLES,
    _boundary_level,
    _sample,
    format_supplement,
)


class FakeTokenizer:
    """只实现 decode，用来喂 _boundary_level。"""

    def __init__(self, text: str):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return self.text


# --------------------------------------------------------------------------- #
# 插话措辞
# --------------------------------------------------------------------------- #
def test_format_uses_fullwidth_parens_not_brackets():
    """方括号会被小模型当成模板复读（[用户说] [用户问]），必须用全角括号。"""
    out = format_supplement("我不吃肉", "plain")
    assert "（用户补充：我不吃肉）" in out
    assert "[" not in out and "]" not in out


def test_format_wraps_in_newlines_so_it_starts_a_new_line():
    out = format_supplement("x", "plain")
    assert out.startswith("\n") and out.endswith("\n")


@pytest.mark.parametrize("style", list(SUPPLEMENT_STYLES))
def test_every_style_keeps_the_original_text(style):
    assert "我不吃肉" in format_supplement("我不吃肉", style)


def test_brief_and_strict_carry_an_instruction_plain_does_not():
    assert "先说明你收到这条补充" in format_supplement("x", "brief")
    assert "不要停下来提问" in format_supplement("x", "strict")
    assert "先说明" not in format_supplement("x", "plain")


def test_unknown_style_falls_back_to_plain():
    assert format_supplement("x", "no-such-style") == format_supplement("x", "plain")


# --------------------------------------------------------------------------- #
# 断句判定
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tail,level,why", [
    ("秋天到了。", 2, "句号是完整断点"),
    ("为什么？", 2, "问号是完整断点"),
    ("太好了！", 2, "感叹号是完整断点"),
    ("这是第一行\n", 2, "换行本身就是断点"),
    ("秋天到了，", 1, "逗号是软断点"),
    ("主要包括：", 1, "冒号是软断点"),
    ("秋风起落叶", 0, "句子中间"),
    ("由阿里云开发", 0, "句子中间"),
])
def test_boundary_level(tail, level, why):
    ids = torch.zeros((1, 8), dtype=torch.long)
    assert _boundary_level(ids, FakeTokenizer(tail)) == level, why


def test_boundary_ignores_trailing_spaces():
    """末尾有空格时，看的是最后一个非空白字符。"""
    ids = torch.zeros((1, 8), dtype=torch.long)
    assert _boundary_level(ids, FakeTokenizer("秋天到了。   ")) == 2


# --------------------------------------------------------------------------- #
# 采样
# --------------------------------------------------------------------------- #
def test_top1_sampling_returns_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    out = _sample(logits, temperature=1.0, top_k=1, top_p=1.0,
                  repetition_penalty=1.0, seen_ids=None)
    assert out.item() == 1


def test_temperature_does_not_change_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    for temp in (0.2, 0.75, 1.5):
        out = _sample(logits, temperature=temp, top_k=1, top_p=1.0,
                      repetition_penalty=1.0, seen_ids=None)
        assert out.item() == 1


def test_repetition_penalty_demotes_seen_token():
    """已出现过的 token 被惩罚后，不该再被选中（top_k=1 下可判定）。"""
    logits = torch.tensor([[0.0, 3.0, 2.9]])
    seen = torch.tensor([0, 1])          # token 1 已出现过
    out = _sample(logits, temperature=1.0, top_k=1, top_p=1.0,
                  repetition_penalty=10.0, seen_ids=seen)
    assert out.item() == 2
