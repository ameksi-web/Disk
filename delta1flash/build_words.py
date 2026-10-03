"""Build a vocabulary file from training corpora.

This tool does not pretend to know every word in the world.  It extracts every
word it sees in provided UTF-8 text/JSONL files and writes a sorted vocabulary.
Give it Wikipedia/books/code corpora to make the vocabulary large.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

WORD_RE = re.compile(r"[\w\-']+", re.UNICODE)


def iter_text(path: Path) -> Iterable[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield str(obj.get("text") or obj.get("title") or obj)
    else:
        yield path.read_text(encoding="utf-8", errors="replace")


def words_from_text(text: str) -> Iterable[str]:
    for match in WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("_-'")
        if token:
            yield token


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract vocabulary from corpora")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/world_words.txt"))
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--target", type=int, default=50000, help="desired number; warning if corpus has fewer")
    args = parser.parse_args(argv)

    counter: Counter[str] = Counter()
    for path in args.inputs:
        for text in iter_text(path):
            counter.update(words_from_text(text))

    words = [word for word, count in counter.items() if count >= args.min_count]
    words.sort(key=lambda w: (-counter[w], w))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"saved {len(words)} words to {args.out}")
    if len(words) < args.target:
        print(f"warning: target {args.target} not reached; add bigger corpora, e.g. Wikipedia/books")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
