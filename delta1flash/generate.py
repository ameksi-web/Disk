"""Generate text from a saved Delta 1 Flash model.

Works both as a module and as a direct script:

    python -m delta1flash.generate
    python delta1flash/generate.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

UNK_CONTEXT = "<UNK>"
FALLBACK_CHAR = "□"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    return project_root() / "runs" / "delta1flash_demo_model.json"


def load_model(model_path: Path) -> dict[str, Any]:
    """Load a model once and reuse it for generate_from_payload/chat.

    Loading the demo JSON can take noticeable time on Windows.  The chat command
    uses this helper so the model is read from disk only once per chat session.
    """

    return json.loads(model_path.read_text(encoding="utf-8"))


def softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def context_key(context: Sequence[int]) -> str:
    return ",".join(str(x) for x in context)


def suffix_contexts(context: Sequence[int]) -> list[list[int]]:
    return [list(context[-n:]) for n in range(len(context), 0, -1)]


def lookup_row(rows: dict[str, list[float]], context: Sequence[int], vocab_size: int) -> list[float]:
    for suffix in suffix_contexts(context):
        row = rows.get(context_key(suffix))
        if row is not None:
            return row
    return rows.get(UNK_CONTEXT) or [0.0 for _ in range(vocab_size)]


def encode_start(start: str, vocab: Sequence[str], token_mode: str) -> list[int]:
    if token_mode == "byte":
        return list(start.encode("utf-8", errors="replace")) or [32]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    fallback_id = stoi.get(FALLBACK_CHAR, stoi.get(" ", 0))
    return [stoi.get(ch, fallback_id) for ch in start] or [fallback_id]


def decode_ids(ids: Sequence[int], vocab: Sequence[str], token_mode: str) -> str:
    if token_mode == "byte":
        return bytes(max(0, min(255, i)) for i in ids).decode("utf-8", errors="replace")
    return "".join(vocab[i] if 0 <= i < len(vocab) else FALLBACK_CHAR for i in ids)


def generate_from_payload(
    payload: dict[str, Any],
    *,
    start: str,
    length: int,
    temperature: float,
    seed: int,
) -> str:
    """Generate text from an already loaded model payload."""

    vocab = payload["vocabulary"]
    token_mode = payload.get("config", {}).get("token_mode", "char")
    rng = random.Random(seed)

    # Current sparse-context format.
    if "rows" in payload:
        rows = payload["rows"]
        context_length = int(payload.get("config", {}).get("context_length", 3))
        out = encode_start(start, vocab, token_mode)
        vocab_size = len(vocab)
        while len(out) < length:
            context = out[-context_length:]
            if len(context) < context_length:
                context = [32 if vocab_size > 32 else 0] * (context_length - len(context)) + context
            row = lookup_row(rows, context, vocab_size)
            logits = [v / max(temperature, 1e-6) for v in row]
            probs = softmax(logits)
            r = rng.random()
            acc = 0.0
            next_id = vocab_size - 1
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    next_id = i
                    break
            out.append(next_id)
        return decode_ids(out[:length], vocab, token_mode)

    # Backward compatibility for early bigram JSON models.
    weights = payload["weights"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    x = stoi.get(start[:1], 0)
    out_ids = [x]
    for _ in range(max(0, length - 1)):
        logits = [v / max(temperature, 1e-6) for v in weights[x]]
        probs = softmax(logits)
        r = rng.random()
        acc = 0.0
        x = len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                x = i
                break
        out_ids.append(x)
    return decode_ids(out_ids, vocab, "char")


def generate(model_path: Path, *, start: str, length: int, temperature: float, seed: int) -> str:
    payload = load_model(model_path)
    return generate_from_payload(payload, start=start, length=length, temperature=temperature, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from Delta 1 Flash")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--start", default="Delta 1 Flash: ")
    parser.add_argument("--length", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(generate(args.model, start=args.start, length=args.length, temperature=args.temperature, seed=args.seed))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
