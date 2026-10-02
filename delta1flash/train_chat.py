"""Train the Delta 1 Flash chat intent model.

This is the default model used by `python -m delta1flash.chat --mode hybrid`.
It is a real trainable softmax classifier over hashed text features.  It is small
and fast, but it learns from examples and handles short/gibberish messages much
better than the raw byte language model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from delta1flash.chat_intent import (
        RESPONSES,
        default_chat_model_path,
        dot,
        extract_features,
        load_examples,
        predict,
        save_model,
        softmax,
    )
else:  # pragma: no cover
    from .chat_intent import (
        RESPONSES,
        default_chat_model_path,
        dot,
        extract_features,
        load_examples,
        predict,
        save_model,
        softmax,
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return project_root() / "data" / "chat_intents.jsonl"


def init_model(labels: Sequence[str], hash_size: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    weights = [[rng.uniform(-0.001, 0.001) for _ in range(hash_size)] for _ in labels]
    bias = [0.0 for _ in labels]
    return {
        "project": "delta1flash",
        "architecture": "hashed_char_word_softmax_intent_classifier",
        "labels": list(labels),
        "hash_size": hash_size,
        "weights": weights,
        "bias": bias,
        "responses": RESPONSES,
        "created_at_unix": time.time(),
        "training_log": [],
    }


def train_epoch(model: dict[str, Any], examples: list[dict[str, str]], *, lr: float, seed: int) -> float:
    rng = random.Random(seed)
    rng.shuffle(examples)
    labels = model["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    hash_size = int(model["hash_size"])
    weights = model["weights"]
    bias = model["bias"]
    total_loss = 0.0

    for item in examples:
        y = label_to_id[item["label"]]
        feats = extract_features(item["text"], hash_size)
        logits = [dot(weights[i], feats) + bias[i] for i in range(len(labels))]
        probs = softmax(logits)
        total_loss += -__import__("math").log(max(probs[y], 1e-12))
        probs[y] -= 1.0
        for i, grad_logit in enumerate(probs):
            bias[i] -= lr * grad_logit
            row = weights[i]
            for idx, value in feats.items():
                row[idx] -= lr * grad_logit * value
    return total_loss / max(1, len(examples))


def accuracy(model: dict[str, Any], examples: Sequence[dict[str, str]]) -> float:
    if not examples:
        return 0.0
    ok = 0
    for item in examples:
        label, _, _ = predict(model, item["text"])
        ok += int(label == item["label"])
    return ok / len(examples)


def split_examples(examples: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    # The demo dataset is intentionally tiny.  Train on all examples so short
    # messages like "ау" and profanity examples are actually learned.  Eval is
    # still printed, but it is a memorization/sanity check, not a benchmark.
    data = list(examples)
    random.Random(seed).shuffle(data)
    return data, data


def run_training(args: argparse.Namespace, *, model: dict[str, Any] | None = None) -> dict[str, Any]:
    examples = load_examples(args.data)
    if not examples:
        raise SystemExit("No chat intent examples found")
    labels = sorted({item["label"] for item in examples})
    train, eval_set = split_examples(examples, args.seed)
    model = model or init_model(labels, args.hash_size, args.seed)

    print("=" * 70)
    print("Delta 1 Flash chat intent training")
    print("Что обучается: softmax intent classifier на hashed char/word features")
    print(f"Данные: {sum(1 for _ in examples)} examples | labels: {', '.join(labels)}")
    print(f"Train/eval: {len(train)} / {len(eval_set)} | hash_size={args.hash_size}")
    print(f"Epochs: {args.epochs} | lr={args.learning_rate} | out={args.out}")
    print("=" * 70)

    log: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train, lr=args.learning_rate, seed=args.seed + epoch)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            train_acc = accuracy(model, train)
            eval_acc = accuracy(model, eval_set)
            event = {"epoch": float(epoch), "loss": loss, "train_acc": train_acc, "eval_acc": eval_acc}
            log.append(event)
            print(f"epoch {epoch}/{args.epochs} | loss {loss:.4f} | train_acc {train_acc:.3f} | eval_acc {eval_acc:.3f}")
    model["training_log"] = (model.get("training_log") or []) + log
    model["updated_at_unix"] = time.time()
    model["num_examples"] = len(examples)
    model["labels"] = labels
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Delta 1 Flash chat intent model")
    parser.add_argument("--data", nargs="+", type=Path, default=[default_data_path()])
    parser.add_argument("--out", type=Path, default=default_chat_model_path())
    parser.add_argument("--hash-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--forever", action="store_true", help="repeat training cycles until Ctrl+C")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited")
    args = parser.parse_args(argv)

    model: dict[str, Any] | None = None
    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"[chat-train] cycle {cycle}")
            model = run_training(args, model=model)
            model["cycles_completed"] = cycle
            save_model(args.out, model)
            print(f"saved: {args.out}")
            if not args.forever or (args.max_cycles and cycle >= args.max_cycles):
                break
    except KeyboardInterrupt:
        if model is not None:
            save_model(args.out, model)
            print(f"\nCtrl+C: saved latest model to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
