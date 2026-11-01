# -*- coding: utf-8 -*-
"""Single-session chat server with mid-stream injection.

Run:
    python -m interject.server --port 8100 --model ./models/qwen2.5-1.5b

Then open http://127.0.0.1:8100

Design notes:
- The model stays resident; generation runs on a worker thread.
- The browser holds one SSE connection; every token and every injection is
  pushed as an event, so the injection is visible the moment it happens.
- One session, one GPU, one generation at a time. This is a demo harness, not a
  multi-tenant service.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from .runtime import generate_chat_turn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STATE: dict = {}
app = FastAPI(title="interject — mid-stream context injection")


def load_model(path: str) -> None:
    """Load a causal LM once and keep it on the GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
    STATE.update(model=model, tokenizer=tokenizer, device=device, model_path=path)
    print(f"[interject] model ready: {path} on {device}", flush=True)


@app.on_event("startup")
def startup() -> None:
    load_model(os.environ.get("INTERJECT_MODEL", str(ROOT / "models" / "qwen2.5-1.5b")))


# --------------------------------------------------------------------------- #
# Chat session
# --------------------------------------------------------------------------- #
# Messages injected mid-flight live in the assistant's content, because that is
# what the model actually saw. `segments` keeps the cut points for the UI.
CHAT: dict = {
    "history": [],
    "inbox": queue.Queue(),
    "busy": False,
    "subscribers": [],
    "lock": threading.Lock(),
    "supp_seq": 0,
    "supp_log": [],
    "turn_seq": 0,
    "turn_started": 0.0,
    "supp_file": ROOT / "logs" / "supplements.jsonl",
}


def _log_supplement(record: dict) -> None:
    try:
        CHAT["supp_file"].parent.mkdir(parents=True, exist_ok=True)
        with open(CHAT["supp_file"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def broadcast(event: dict) -> None:
    for subscriber in list(CHAT["subscribers"]):
        try:
            subscriber.put_nowait(event)
        except Exception:  # noqa: BLE001
            pass


def _new_supplement(text: str) -> dict:
    """Assign an id and record the submit time (wall clock + seconds into the turn)."""
    with CHAT["lock"]:
        CHAT["supp_seq"] += 1
        now = time.time()
        rel = (time.perf_counter() - CHAT["turn_started"]) if CHAT["turn_started"] else 0.0
        record = {
            "id": CHAT["supp_seq"],
            "text": text,
            "submitted_at": round(now, 3),
            "submitted_wall": time.strftime("%H:%M:%S", time.localtime(now)),
            "submitted_rel": round(rel, 3),
            "turn": CHAT.get("turn_seq", 0),
        }
        CHAT["supp_log"].append(record)
    _log_supplement(record)
    return record


def chat_worker(max_new_tokens: int) -> None:
    try:
        broadcast({"type": "assistant_start"})
        result = generate_chat_turn(
            STATE["model"], STATE["tokenizer"], CHAT["history"], CHAT["inbox"],
            max_new_tokens=max_new_tokens,
            event_cb=broadcast,
            max_wait_tokens=int(os.environ.get("INTERJECT_MAX_WAIT", 24)),
        )
        CHAT["history"].append({
            "role": "assistant",
            "content": result["full_text"],
            "segments": result.get("segments", []),
            "supplements": result.get("supplements", []),
        })
        taken = {s.get("id"): s for s in result.get("supplements", [])}
        for rec in CHAT["supp_log"]:
            s = taken.get(rec["id"])
            if s and "taken_step" not in rec:
                rec.update(taken_step=s.get("step"), taken_at=s.get("at"),
                           waited_tokens=s.get("waited_tokens"),
                           at_boundary=s.get("at_boundary"))
                _log_supplement({**rec, "event": "taken"})
        broadcast({"type": "supp_log", "items": CHAT["supp_log"]})
        broadcast({"type": "assistant_done", **result})
    except Exception as exc:  # noqa: BLE001
        broadcast({"type": "error", "message": str(exc)})
    finally:
        CHAT["busy"] = False
        CHAT["turn_started"] = 0.0
        broadcast({"type": "idle"})


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/chat")


@app.get("/chat", response_class=HTMLResponse)
def chat_page() -> str:
    return (HERE / "webui" / "chat.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info() -> dict:
    return {
        "device": str(STATE.get("device")),
        "model": STATE.get("model_path"),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "busy": CHAT["busy"],
    }


@app.get("/api/chat/state")
def chat_state() -> dict:
    return {"history": CHAT["history"], "busy": CHAT["busy"],
            "supp_log": CHAT["supp_log"]}


@app.post("/api/chat/reset")
def chat_reset() -> dict:
    with CHAT["lock"]:
        CHAT["history"].clear()
        while not CHAT["inbox"].empty():
            try:
                CHAT["inbox"].get_nowait()
            except queue.Empty:
                break
        broadcast({"type": "reset"})
    return {"ok": True}


@app.post("/api/chat/send")
def chat_send(payload: dict) -> dict:
    text = (payload.get("text") or "").strip()
    if not text:
        return {"ok": False, "reason": "empty"}
    max_new_tokens = int(payload.get("max_tokens", 1024))

    if CHAT["busy"]:
        # Model is talking: hand the message to the decoding loop. No restart.
        record = _new_supplement(text)
        CHAT["inbox"].put(record)
        broadcast({"type": "queued", **record})
        return {"ok": True, "mode": "injected", "supplement": record}

    with CHAT["lock"]:
        if CHAT["busy"]:
            record = _new_supplement(text)
            CHAT["inbox"].put(record)
            broadcast({"type": "queued", **record})
            return {"ok": True, "mode": "injected", "supplement": record}
        CHAT["turn_seq"] = CHAT.get("turn_seq", 0) + 1
        CHAT["turn_started"] = time.perf_counter()
        CHAT["history"].append({"role": "user", "content": text})
        CHAT["busy"] = True
    broadcast({"type": "user", "text": text})
    threading.Thread(target=chat_worker, args=(max_new_tokens,), daemon=True).start()
    return {"ok": True, "mode": "new_turn"}


@app.get("/api/chat/stream")
def chat_stream():
    def event_stream():
        subscriber: queue.Queue = queue.Queue()
        CHAT["subscribers"].append(subscriber)
        try:
            yield ("data: " + json.dumps(
                {"type": "hello", "busy": CHAT["busy"],
                 "history": CHAT["history"], "supp_log": CHAT["supp_log"]},
                ensure_ascii=False) + "\n\n")
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            if subscriber in CHAT["subscribers"]:
                CHAT["subscribers"].remove(subscriber)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main() -> None:
    ap = argparse.ArgumentParser(description="interject demo server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default=None, help="path to a HuggingFace causal LM")
    args = ap.parse_args()
    if args.model:
        os.environ["INTERJECT_MODEL"] = args.model
    print(f"WebUI: http://{args.host}:{args.port}/chat", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
