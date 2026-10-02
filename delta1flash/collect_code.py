"""Collect local source code files into JSONL for Delta 1 Flash training.

This does not scrape GitHub.  It scans directories that you own or explicitly
provide, preserving path/extension metadata.  Use only code you have the right
to train on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".c", ".h", ".cpp", ".hpp",
    ".java", ".kt", ".swift", ".cs", ".php", ".rb", ".lua", ".r", ".sql", ".html",
    ".css", ".scss", ".json", ".yaml", ".yml", ".toml", ".md", ".sh", ".ps1",
}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target", ".next"}


def iter_code_files(root: Path, max_bytes: int) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect local source code into JSONL")
    parser.add_argument("--root", type=Path, default=Path("."), help="directory to scan")
    parser.add_argument("--out", type=Path, default=Path("data/code_local.jsonl"))
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=200000, help="skip larger files")
    parser.add_argument("--max-chars", type=int, default=12000, help="truncate each file text")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    chars = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for path in iter_code_files(args.root, args.max_bytes):
            try:
                text = read_text(path)
            except OSError:
                continue
            rel = path.relative_to(args.root).as_posix() if path.is_relative_to(args.root) else path.as_posix()
            doc = {
                "source": "local_code",
                "path": rel,
                "extension": path.suffix.lower(),
                "license": "user-provided; verify rights before training",
                "title": rel,
                "text": text[: args.max_chars],
            }
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            chars += len(doc["text"])
            if count >= args.max_files:
                break
    print(f"saved {count} code files / {chars} characters to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
