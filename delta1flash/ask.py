"""Ask Delta 1 Flash one question using only the trained generative model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from delta1flash.chat import chat_once
    from delta1flash.generate import default_model_path, load_model
else:
    from .chat import chat_once
    from .generate import default_model_path, load_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask Delta 1 Flash one question")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--question", default="привет")
    parser.add_argument("--length", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload = load_model(args.model)
    print(chat_once(payload, args.question, max_new_tokens=args.length, temperature=args.temperature, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
