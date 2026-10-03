"""Collect public-domain book text for Delta 1 Flash.

Use only texts you have the right to use.  The default helper downloads a small
set of public-domain Project Gutenberg books; you can also convert local UTF-8
text files into the same JSONL format.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

USER_AGENT = "Delta1FlashTrainingBot/0.1 (educational local script)"
DEFAULT_GUTENBERG_IDS = [11, 84, 1342, 1661, 2701, 345]


def clean_gutenberg(text: str) -> str:
    # Strip common Project Gutenberg header/footer markers when present.
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    if start and end and start.end() < end.start():
        text = text[start.end() : end.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def download_gutenberg(book_id: int, max_chars: int) -> dict[str, str]:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310 - public URL
        raw = response.read().decode("utf-8", errors="replace")
    text = clean_gutenberg(raw)
    return {
        "source": "project_gutenberg",
        "book_id": str(book_id),
        "url": url,
        "license": "public domain in many jurisdictions; verify for your country",
        "title": f"Project Gutenberg #{book_id}",
        "text": text[:max_chars],
    }


def local_doc(path: Path, max_chars: int) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "source": "local_text_file",
        "path": str(path),
        "license": "user-provided; verify rights before training",
        "title": path.name,
        "text": text[:max_chars],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect public-domain books/local text for Delta 1 Flash")
    parser.add_argument("--out", type=Path, default=Path("data/books_public_domain.jsonl"))
    parser.add_argument("--gutenberg-ids", default=",".join(map(str, DEFAULT_GUTENBERG_IDS)))
    parser.add_argument("--local", nargs="*", type=Path, default=[], help="local UTF-8 .txt files")
    parser.add_argument("--max-chars", type=int, default=250000)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    count = 0
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for item in [x.strip() for x in args.gutenberg_ids.split(",") if x.strip()]:
                doc = download_gutenberg(int(item), args.max_chars)
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
            for path in args.local:
                doc = local_doc(path, args.max_chars)
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"Cannot collect books right now: {exc}", file=sys.stderr)
        return 2

    tmp.replace(args.out)
    print(f"saved {count} book/local documents to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
