"""One-question wrapper for Delta 1 Flash chat.

For a real dialogue use:

    python -m delta1flash.chat

This command is for quick single prompts and uses the same hybrid answer engine
as chat.py: trained intent model + memory + filtering of broken raw model text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from delta1flash.chat import chat_once, default_memory_path, load_memory
    from delta1flash.chat_intent import default_chat_model_path, load_chat_model
    from delta1flash.generate import default_model_path, load_model
else:  # pragma: no cover
    from .chat import chat_once, default_memory_path, load_memory
    from .chat_intent import default_chat_model_path, load_chat_model
    from .generate import default_model_path, load_model


def adapt_question(question: str, language: str) -> str:
    if language == "code" and not any(word in question.lower() for word in ["код", "code", "python"]):
        return f"напиши код: {question}"
    return question


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask Delta 1 Flash one question")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--chat-model", type=Path, default=default_chat_model_path())
    parser.add_argument("--question", default="как ты будешь отвечать?")
    parser.add_argument("--language", choices=["ru", "en", "code"], default="ru")
    parser.add_argument("--memory", type=Path, default=default_memory_path())
    parser.add_argument("--mode", choices=["hybrid", "model", "rules"], default="hybrid")
    parser.add_argument("--intent-threshold", type=float, default=0.42)
    parser.add_argument("--length", type=int, default=900, help="kept for compatibility; maps to max-new-tokens")
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args(argv)

    payload = load_model(args.model)
    memory = load_memory(args.memory)
    intent_model = load_chat_model(args.chat_model)
    question = adapt_question(args.question, args.language)
    reply = chat_once(
        payload,
        question,
        memory=memory,
        intent_model=intent_model,
        intent_threshold=args.intent_threshold,
        max_new_tokens=args.length,
        temperature=args.temperature,
        seed=args.seed,
        mode=args.mode,
    )
    print(reply)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
