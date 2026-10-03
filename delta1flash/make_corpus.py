"""Merge JSONL documents and plain text files into one training corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


def iter_jsonl_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = (obj.get("text") or "").strip()
            title = (obj.get("title") or obj.get("path") or "document").strip()
            if text:
                yield f"\n\n# {title}\n{text}\n"


def iter_plain_text(path: Path) -> Iterable[str]:
    yield path.read_text(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Delta 1 Flash training corpus")
    parser.add_argument("--out", type=Path, default=Path("data/corpus_5000.txt"))
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help=".jsonl or .txt files")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-chars", type=int, default=0, help="0 means unlimited")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    chars = 0
    with args.out.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            iterator = iter_jsonl_text(path) if path.suffix.lower() == ".jsonl" else iter_plain_text(path)
            for text in iterator:
                if args.max_docs and docs >= args.max_docs:
                    break
                if args.max_chars:
                    remaining = args.max_chars - chars
                    if remaining <= 0:
                        break
                    text = text[:remaining]
                out.write(text)
                out.write("\n")
                chars += len(text)
                docs += 1
            if (args.max_docs and docs >= args.max_docs) or (args.max_chars and chars >= args.max_chars):
                break
    print(f"wrote {chars} characters from {docs} documents to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
