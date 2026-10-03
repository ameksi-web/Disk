"""Interactive terminal chat for Delta 1 Flash.

This chat uses ONLY the trained generative model.  It does not use intent rules,
prepared answers, or hard-coded response templates.  The reply is sampled from
`runs/delta1flash_demo_model.json`, so quality depends entirely on training data
and training time.

Run:

    python -m delta1flash.chat
    python delta1flash/chat.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from delta1flash.generate import default_model_path, generate_from_payload, load_model
else:  # pragma: no cover
    from .generate import default_model_path, generate_from_payload, load_model

STOP_MARKERS = ["\nВы:", "\nПользователь:", "\nUser:", "\nDelta:", "\n###", "§END§"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_history_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return project_root() / "chats" / f"chat_{stamp}.jsonl"


def build_prompt(message: str, history: Sequence[dict[str, str]], max_history: int) -> str:
    lines: list[str] = []
    for item in history[-max_history:]:
        user = item.get("user", "").strip()
        assistant = item.get("assistant", "").strip()
        if user and assistant:
            lines.append(f"Вы: {user}")
            lines.append(f"Delta: {assistant}")
    lines.append(f"Вы: {message.strip()}")
    lines.append("Delta:")
    return "\n".join(lines)


def trim_reply(full_text: str, prompt: str) -> str:
    reply = full_text[len(prompt) :].strip()
    stop = len(reply)
    for marker in STOP_MARKERS:
        idx = reply.find(marker)
        if idx != -1:
            stop = min(stop, idx)
    return reply[:stop].strip()


def repetition_score(text: str) -> float:
    words = text.lower().split()
    if len(words) < 8:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    return 1.0 - unique_ratio


def generate_reply(
    payload: dict,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    attempts: int,
) -> str:
    best = ""
    best_score = 999.0
    total_length = len(prompt.encode("utf-8")) + max_new_tokens
    for i in range(max(1, attempts)):
        full = generate_from_payload(
            payload,
            start=prompt,
            length=total_length,
            temperature=temperature,
            seed=seed + i * 1009,
        )
        reply = trim_reply(full, prompt)
        if not reply:
            continue
        score = repetition_score(reply) + (0.4 if "�" in reply else 0.0)
        if score < best_score:
            best = reply
            best_score = score
    return best


def chat_once(
    payload: dict,
    message: str,
    *,
    history: Sequence[dict[str, str]] = (),
    max_history: int = 3,
    max_new_tokens: int = 500,
    temperature: float = 0.75,
    seed: int | None = None,
    attempts: int = 4,
) -> str:
    prompt = build_prompt(message, history, max_history=max_history)
    if seed is None or seed == 0:
        seed = time.time_ns() % 2_147_483_647
    reply = generate_reply(
        payload,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        attempts=attempts,
    )
    return reply or "..."


def save_turn(history_path: Path, user: str, assistant: str) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.time(), "user": user, "assistant": assistant}
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def print_help() -> None:
    print("Команды чата:")
    print("  /help        показать помощь")
    print("  /reset       очистить историю текущего диалога")
    print("  /temp 0.8    изменить temperature")
    print("  /exit        выйти")
    print("Важно: чат генерирует только обученная модель, без готовых ответов.")


def interactive_chat(args: argparse.Namespace) -> int:
    print(f"Загружаю модель: {args.model}")
    payload = load_model(args.model)
    history: list[dict[str, str]] = []
    history_path = args.history or default_history_path()
    temperature = args.temperature
    print("Delta 1 Flash generative chat готов. Для выхода: /exit, помощь: /help")
    print(f"История: {history_path}")
    print("Режим: только обученная модель, без готовых фраз/intent-ответов.")
    while True:
        try:
            message = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return 0
        if not message:
            continue
        if message in {"/exit", "/quit", "выход"}:
            print("Выход.")
            return 0
        if message == "/help":
            print_help()
            continue
        if message == "/reset":
            history.clear()
            print("История текущего диалога очищена.")
            continue
        if message.startswith("/temp"):
            parts = message.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"temperature = {temperature}")
                except ValueError:
                    print("Нужно число, пример: /temp 0.8")
            else:
                print(f"temperature = {temperature}")
            continue

        reply = chat_once(
            payload,
            message,
            history=history,
            max_history=args.max_history,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            seed=0,
            attempts=args.attempts,
        )
        print(f"Delta: {reply}")
        history.append({"user": message, "assistant": reply})
        save_turn(history_path, message, reply)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generative terminal chat with Delta 1 Flash")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--message", help="single message mode; if omitted, starts interactive chat")
    parser.add_argument("--history", type=Path, help="JSONL history path")
    parser.add_argument("--max-history", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0, help="0 means random seed")
    parser.add_argument("--attempts", type=int, default=4, help="sample several replies and keep the least repetitive")
    args = parser.parse_args(argv)

    if args.message:
        payload = load_model(args.model)
        reply = chat_once(
            payload,
            args.message,
            max_history=args.max_history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            attempts=args.attempts,
        )
        print(reply)
        if args.history:
            save_turn(args.history, args.message, reply)
        return 0
    return interactive_chat(args)


if __name__ == "__main__":
    raise SystemExit(main())
