# -*- coding: utf-8 -*-
"""Mid-stream context injection for autoregressive decoders.

The idea: a user can speak while the model is still writing. Instead of stopping
the generation, restarting it, or recomputing the KV cache, we append the new
tokens to the running sequence. The next forward pass folds them into the cache
naturally, so the model keeps thinking — it just thinks with the extra context.

Everything here runs on a plain HuggingFace causal LM. No custom kernels.
"""
from __future__ import annotations

import queue
import time
from typing import Callable

import torch

# --------------------------------------------------------------------------- #
# Injection wording
# --------------------------------------------------------------------------- #
SUPPLEMENT_OPEN = "用户补充"

# Placement matters much more than wording.
# Measured on Qwen2.5-1.5B, insertion at token 100 of a poem-writing task:
#   plain  -> the model acknowledged the switch in 1 of 3 seeds
#   brief  -> 3 of 3
# Putting the same instruction in a system message failed completely: the model
# learned only to reply "please wait, I will write for you" and then emitted EOS.
SUPPLEMENT_STYLES = {
    "plain": "（{open}：{text}）",
    "brief": "（{open}：{text}。先说明你收到这条补充，再直接照做。）",
    "strict": "（{open}：{text}。收到后先用一句话说明，然后立刻按新要求继续，不要停下来提问。）",
}

# Optional system prompt. Left off by default: a 1.5B model follows instructions
# far better when they sit next to the inserted text than at the top of the context.
SYSTEM_PROMPT = (
    "你是一个可以随时接收用户中途补充的助手。\n"
    "对话里出现「（用户补充：…）」时，表示用户在你说到一半时插话了。收到后按这个顺序做：\n"
    "第一，用一句话复述用户补充了什么，并说明你是否因此改变正在做的事；\n"
    "第二，如果补充改变了任务，明确说出你正在切换任务，并且停止原来的事；\n"
    "第三，接着完成补充要求的内容，不要回到原来的任务。\n"
    "补充信息已经出现在你面前，不存在没听到或需要重复一遍的情况。"
)

# --------------------------------------------------------------------------- #
# Where in the stream to insert
# --------------------------------------------------------------------------- #
# Inserting the moment a message arrives slices sentences in half — a 1.5B model
# then writes "I am Qwen, made by Alibaba I am a pretrained language model".
# So we wait for a sentence boundary, with two fallbacks.
_TERMINALS = set("。！？!?…；;.")
_SOFT_BREAKS = set("，、：,:")
_SOFT_AFTER = 8      # tokens to wait before accepting a comma as a break point
_MAX_WAIT = 24       # tokens to wait before forcing a full stop and inserting


def _cache_len(past) -> int:
    """Number of tokens already in the KV cache.

    DynamicCache stores (batch, heads, seq, dim), so `past[0][0].shape[1]` is the
    head count — not the sequence length. That mistake silently feeds the model
    the wrong slice and produces garbage output.
    """
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return past.get_seq_length()
    return past[0][0].shape[1]


def _sample(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float,
            repetition_penalty: float, seen_ids: torch.Tensor | None) -> torch.Tensor:
    """Temperature -> repetition penalty -> top-k -> top-p -> multinomial."""
    logits = logits / max(temperature, 1e-5)
    if repetition_penalty != 1.0 and seen_ids is not None:
        seen = torch.unique(seen_ids)
        score = logits[:, seen]
        logits[:, seen] = torch.where(score > 0, score / repetition_penalty,
                                      score * repetition_penalty)
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        drop = cumulative > top_p
        drop[..., 1:], drop[..., 0] = drop[..., :-1].clone(), False
        logits = logits.masked_fill(drop.scatter(1, sorted_indices, drop), float("-inf"))
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def format_supplement(text: str, style: str = "brief") -> str:
    """How an injected message looks in the context.

    Square brackets such as `[user note]` are deliberately avoided: small models
    treat the bracket structure as a template to imitate and start echoing
    `[user says]` / `[user asks]`. Full-width parentheses read as prose.
    """
    tpl = SUPPLEMENT_STYLES.get(style) or SUPPLEMENT_STYLES["plain"]
    return "\n" + tpl.format(open=SUPPLEMENT_OPEN, text=text) + "\n"


def _boundary_level(input_ids: torch.Tensor, tokenizer, window: int = 6) -> int:
    """2 = end of sentence, 1 = comma-like break, 0 = mid-sentence."""
    tail = tokenizer.decode(input_ids[0, -window:], skip_special_tokens=True)
    for ch in reversed(tail):
        if ch.isspace():
            return 2
        if ch in _TERMINALS:
            return 2
        if ch in _SOFT_BREAKS:
            return 1
        return 0
    return 0


# --------------------------------------------------------------------------- #
# The decoding loop
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_chat_turn(
    model,
    tokenizer,
    history: list[dict],
    inbox: "queue.Queue",
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.75,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    seed: int | None = None,
    event_cb: Callable[[dict], None] | None = None,
    system_prompt: str | None = None,
    supplement_style: str = "brief",
    max_wait_tokens: int = _MAX_WAIT,
    soft_after: int = _SOFT_AFTER,
):
    """Run one assistant turn; messages arriving in `inbox` are spliced in mid-flight.

    `inbox` accepts either plain strings or dicts carrying an id / submit time.
    Returns a dict describing the turn, including `segments` — the output split
    at every insertion point, so a UI can show which message changed which part.

    Note: `@torch.inference_mode()` is not optional. Without it every step keeps
    its autograd graph and a 12 GB card hits OOM after a few hundred tokens.
    """
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = list(history)
    if system_prompt and not (messages and messages[0].get("role") == "system"):
        messages = [{"role": "system", "content": system_prompt}] + messages

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    past = None
    started = time.perf_counter()
    new_tokens: list[torch.Tensor] = []
    supplements: list[dict] = []
    insertions: list[torch.Tensor] = []   # token pieces that were spliced in
    insert_positions: list[int] = []      # after how many generated tokens
    stopped_by_eos = False
    pending: list[dict] = []
    pending_since = 0

    for step in range(max_new_tokens):
        # ---- collect messages (never blocks) ----
        while True:
            try:
                item = inbox.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, dict):
                message = str(item.get("text") or "")
            else:
                message = str(item)
                item = {"text": message}
            if not message:
                continue
            if not pending:
                pending_since = step
            pending.append(item)
            if event_cb:
                event_cb({"type": "supplement_held", **item, "step": step,
                          "at": round(time.perf_counter() - started, 3)})

        # ---- splice in at a boundary, not mid-sentence ----
        if pending:
            level = _boundary_level(input_ids, tokenizer)
            waited = step - pending_since
            ready = (level == 2
                     or (level == 1 and waited >= soft_after)
                     or waited >= max_wait_tokens)
            if ready:
                at_boundary = level > 0
                if not at_boundary:
                    # Sentence is half-written: close it first so two clauses
                    # do not fuse into one ungrammatical run-on.
                    stop_piece = tokenizer("。", return_tensors="pt",
                                           add_special_tokens=False)["input_ids"].to(device)
                    input_ids = torch.cat([input_ids, stop_piece], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((1, stop_piece.shape[1]))], dim=-1)
                    insertions.append(stop_piece[0])
                    insert_positions.append(len(new_tokens))

                joined = "；".join(p["text"] for p in pending)
                # Character offset at insertion: lets the UI split the reply into
                # "before this message" / "after this message" segments.
                chars = len(tokenizer.decode(torch.cat(new_tokens, dim=-1)[0],
                                             skip_special_tokens=True)) if new_tokens else 0
                piece = tokenizer(format_supplement(joined, supplement_style),
                                  return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                input_ids = torch.cat([input_ids, piece], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, piece.shape[1]))], dim=-1)

                record = dict(pending[0])
                record.update({
                    "text": joined,
                    "ids": [p.get("id") for p in pending],
                    "items": [{k: p.get(k) for k in
                               ("id", "text", "submitted_wall", "submitted_rel")
                               if p.get(k) is not None} for p in pending],
                    "step": step,
                    "at": round(time.perf_counter() - started, 3),
                    "chars": chars,
                    "waited_tokens": waited,
                    "at_boundary": at_boundary,
                    "boundary": {2: "sentence", 1: "comma"}.get(level, "forced"),
                })
                supplements.append(record)
                insertions.append(piece[0])
                insert_positions.append(len(new_tokens))
                if event_cb:
                    event_cb({"type": "supplement_taken", **record})
                pending = []

        # ---- one decoding step ----
        past_len = _cache_len(past)
        outputs = model(
            input_ids[:, past_len:],
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        nxt = _sample(outputs.logits[:, -1, :], temperature=temperature, top_k=top_k,
                      top_p=top_p, repetition_penalty=repetition_penalty,
                      seen_ids=input_ids[0])
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim=-1)
        new_tokens.append(nxt)

        if event_cb:
            shown = tokenizer.decode(torch.cat(new_tokens, dim=-1)[0], skip_special_tokens=True)
            event_cb({"type": "token", "text": shown, "index": step})

        if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
            stopped_by_eos = True
            break

    wall = time.perf_counter() - started
    clean = tokenizer.decode(torch.cat(new_tokens, dim=-1)[0], skip_special_tokens=True)

    dropped_records = [dict(p) for p in pending]
    for msg in dropped_records:
        if event_cb:
            event_cb({"type": "supplement_dropped", **msg,
                      "reason": "本轮结束前没等到句子边界"})

    # Split the reply at every insertion so the UI can draw "this message shaped
    # that paragraph" without re-deriving offsets.
    segments: list[dict] = []
    cursor = 0
    for s in supplements:
        c = int(s.get("chars") or 0)
        if c > cursor:
            segments.append({"text": clean[cursor:c], "supp": None})
        segments.append({"text": "", "supp": dict(s)})
        cursor = c
    if cursor < len(clean):
        segments.append({"text": clean[cursor:], "supp": None})

    # What the model actually saw, in order: generated tokens and injected pieces.
    if not insertions:
        full_text = clean
    else:
        chunks: list[torch.Tensor] = []
        cursor = 0
        for tokens, pos in zip(insertions, insert_positions):
            if pos > cursor:
                chunks.append(torch.cat(new_tokens[cursor:pos], dim=-1)[0])
            chunks.append(tokens)
            cursor = pos
        if cursor < len(new_tokens):
            chunks.append(torch.cat(new_tokens[cursor:], dim=-1)[0])
        full_text = tokenizer.decode(torch.cat(chunks, dim=-1), skip_special_tokens=True)

    return {
        "text": clean,
        "full_text": full_text,
        "segments": segments,
        "supplements": supplements,
        "supplements_dropped": dropped_records,
        "n_new_tokens": len(new_tokens),
        "wall_seconds": round(wall, 3),
        "tokens_per_second": round(len(new_tokens) / wall, 2) if wall > 0 else 0.0,
        "truncated": (not stopped_by_eos) and len(new_tokens) >= max_new_tokens,
        "max_new_tokens": max_new_tokens,
    }
