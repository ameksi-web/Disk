"""Collect public Wikipedia text for Delta 1 Flash.

Examples:

    # 5000 articles about human/programming languages from many Wikipedias
    python -m delta1flash.collect_wikipedia --target 5000 --languages all --mode languages

    # 5000 articles about neural networks and machine learning
    python -m delta1flash.collect_wikipedia --target 5000 --languages en,ru --mode neural

The script writes JSONL documents with text plus attribution metadata.  It uses
Wikimedia's public API.  If the network is unavailable, the program exits with a
clear error and leaves existing output files untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
SITEMATRIX_API = "https://meta.wikimedia.org/w/api.php"
USER_AGENT = "Delta1FlashTrainingBot/0.2 (educational local script)"

# A broad fallback when the sitematrix endpoint is unavailable.  It is not every
# language in the world, but it covers many scripts and high-resource Wikipedias.
FALLBACK_ALL_WIKI_LANGS = [
    "en", "ru", "es", "fr", "de", "it", "pt", "pl", "uk", "nl", "sv", "ceb",
    "vi", "war", "ar", "zh", "ja", "ko", "hi", "bn", "fa", "tr", "id", "ms",
    "th", "ta", "te", "ur", "he", "el", "cs", "sk", "sl", "hr", "sr", "bg",
    "ro", "hu", "fi", "da", "no", "nn", "et", "lv", "lt", "ka", "hy", "az",
    "kk", "uz", "be", "mk", "sq", "ca", "eu", "gl", "la", "simple",
]
CORE_WIKI_LANGS = ["en", "ru", "es", "fr", "de", "zh", "ar", "hi", "ja", "pt"]

NEURAL_TOPICS = [
    "artificial neural network",
    "neural network software",
    "machine learning",
    "deep learning",
    "backpropagation",
    "gradient descent",
    "language model",
    "transformer machine learning model",
    "recurrent neural network",
    "convolutional neural network",
    "natural language processing",
    "tokenization",
]

LANGUAGE_TOPICS = [
    "language",
    "linguistics",
    "programming language",
    "natural language",
    "formal language",
    "language family",
    "syntax linguistics",
    "semantics",
    "compiler",
    "interpreter computing",
    "Python programming language",
    "JavaScript",
    "Rust programming language",
    "C programming language",
    "Java programming language",
    "Go programming language",
    "SQL",
    "Arabic language",
    "Chinese language",
    "English language",
    "Hindi",
    "Russian language",
    "Spanish language",
]

CODE_TOPICS = [
    "computer programming",
    "source code",
    "algorithm",
    "data structure",
    "software engineering",
    "compiler",
    "interpreter computing",
    "version control",
    "debugging",
    "unit testing",
    "Python programming language",
    "JavaScript",
    "TypeScript",
    "Rust programming language",
    "C programming language",
    "C++",
    "Java programming language",
    "Go programming language",
    "SQL",
    "HTML",
    "CSS",
]


def api_get_url(url: str, params: Dict[str, object], timeout: int = 30) -> Dict[str, object]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - public API URL
        return json.loads(response.read().decode("utf-8"))


def api_get(lang: str, params: Dict[str, object], timeout: int = 30) -> Dict[str, object]:
    return api_get_url(WIKI_API.format(lang=lang), params, timeout=timeout)


def fetch_all_wikipedia_languages(timeout: int = 30) -> List[str]:
    """Return active Wikipedia language codes from Wikimedia sitematrix."""

    data = api_get_url(SITEMATRIX_API, {"action": "sitematrix", "smlangprop": "code|site"}, timeout=timeout)
    codes: List[str] = []
    for key, group in data.get("sitematrix", {}).items():
        if not key.isdigit() or not isinstance(group, dict):
            continue
        code = group.get("code")
        sites = group.get("site") or []
        if not code or code in codes:
            continue
        if any(site.get("code") == "wiki" and not site.get("closed") for site in sites):
            codes.append(code)
    return sorted(codes)


def resolve_languages(value: str) -> List[str]:
    value = value.strip()
    if value == "core":
        return CORE_WIKI_LANGS
    if value == "all":
        try:
            langs = fetch_all_wikipedia_languages()
            if langs:
                return langs
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            # Let collection continue with a broad offline fallback.  A later API
            # call will still report an explicit network error if the network is
            # unavailable for article downloads too.
            return FALLBACK_ALL_WIKI_LANGS
        return FALLBACK_ALL_WIKI_LANGS
    return [x.strip() for x in value.split(",") if x.strip()]


def topics_for_mode(mode: str) -> List[str]:
    if mode == "neural":
        return NEURAL_TOPICS
    if mode == "languages":
        return LANGUAGE_TOPICS
    if mode == "code":
        return CODE_TOPICS
    if mode == "all":
        return NEURAL_TOPICS + LANGUAGE_TOPICS + CODE_TOPICS
    return NEURAL_TOPICS + LANGUAGE_TOPICS


def search_titles(lang: str, query: str, limit: int) -> List[str]:
    data = api_get(
        lang,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": min(50, limit),
            "srnamespace": 0,
        },
    )
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def random_titles(lang: str, limit: int) -> List[str]:
    data = api_get(
        lang,
        {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(500, limit),
        },
    )
    return [item["title"] for item in data.get("query", {}).get("random", [])]


def fetch_extracts(lang: str, titles: Sequence[str], max_chars: int) -> Iterator[Dict[str, str]]:
    if not titles:
        return
    for i in range(0, len(titles), 20):
        batch = titles[i : i + 20]
        data = api_get(
            lang,
            {
                "action": "query",
                "prop": "extracts|info",
                "inprop": "url",
                "explaintext": 1,
                "exsectionformat": "plain",
                "redirects": 1,
                "titles": "|".join(batch),
            },
        )
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            extract = clean_text(page.get("extract") or "")
            if len(extract) < 400:
                continue
            yield {
                "source": "wikipedia",
                "language": lang,
                "title": page.get("title", ""),
                "url": page.get("fullurl", ""),
                "license": "CC BY-SA",
                "text": extract[:max_chars],
            }


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def collect(
    *,
    out: Path,
    target: int,
    languages: Sequence[str],
    mode: str,
    max_chars: int,
    sleep: float,
    seed: int,
) -> int:
    rng = random.Random(seed)
    topics = topics_for_mode(mode)
    seen_urls: set[str] = set()
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8") as fh:
            # Topical search first.
            topic_queue = [(lang, topic) for lang in languages for topic in topics]
            rng.shuffle(topic_queue)
            for lang, topic in topic_queue:
                if written >= target:
                    break
                titles = search_titles(lang, topic, limit=50)
                for doc in fetch_extracts(lang, titles, max_chars=max_chars):
                    key = doc.get("url") or f"{doc['language']}:{doc['title']}"
                    if key in seen_urls:
                        continue
                    seen_urls.add(key)
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= target:
                        break
                time.sleep(sleep)

            # Fill the rest with random encyclopedia pages.
            while written < target:
                lang = rng.choice(list(languages))
                titles = random_titles(lang, min(500, target - written))
                for doc in fetch_extracts(lang, titles, max_chars=max_chars):
                    key = doc.get("url") or f"{doc['language']}:{doc['title']}"
                    if key in seen_urls:
                        continue
                    seen_urls.add(key)
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= target:
                        break
                print(f"collected {written}/{target}", file=sys.stderr)
                time.sleep(sleep)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Cannot reach Wikipedia API right now: {exc}") from exc

    tmp.replace(out)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Wikipedia articles for Delta 1 Flash")
    parser.add_argument("--out", type=Path, default=Path("data/wiki_5000.jsonl"))
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument(
        "--languages",
        default="core",
        help="comma-separated Wikipedia codes, or preset: core/all. 'all' queries Wikimedia sitematrix.",
    )
    parser.add_argument("--mode", choices=["neural", "languages", "code", "mixed", "all"], default="all")
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    languages = resolve_languages(args.languages)
    if not languages:
        print("No Wikipedia languages selected", file=sys.stderr)
        return 2

    try:
        count = collect(
            out=args.out,
            target=args.target,
            languages=languages,
            mode=args.mode,
            max_chars=args.max_chars,
            sleep=args.sleep,
            seed=args.seed,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"saved {count} articles to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
