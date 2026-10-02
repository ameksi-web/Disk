"""Trainable intent model for Delta 1 Flash chat.

This is a small real model: hashed character/word features go into a softmax
classifier trained with SGD.  It is not a large LLM, but it learns from examples
and prevents the default chat from repeating broken language-model fragments.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


RUSSIAN_LETTERS = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
RUSSIAN_ALPHABET_TEXT = " ".join(RUSSIAN_LETTERS)
LETTER_NAMES = {
    "а": "а", "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "е": "е", "ё": "ё",
    "ж": "жэ", "з": "зэ", "и": "и", "й": "и краткое", "к": "ка", "л": "эль",
    "м": "эм", "н": "эн", "о": "о", "п": "пэ", "р": "эр", "с": "эс", "т": "тэ",
    "у": "у", "ф": "эф", "х": "ха", "ц": "цэ", "ч": "че", "ш": "ша", "щ": "ща",
    "ъ": "твёрдый знак", "ы": "ы", "ь": "мягкий знак", "э": "э", "ю": "ю", "я": "я",
}


def squash_repeats(text: str, max_repeat: int = 1) -> str:
    """Collapse repeated characters: пппррривет -> привет.

    This helps the intent model understand stretched or mistyped Russian words.
    """

    out: list[str] = []
    prev = ""
    count = 0
    for ch in text:
        if ch == prev:
            count += 1
        else:
            prev = ch
            count = 1
        if count <= max_repeat:
            out.append(ch)
    return "".join(out)

RESPONSES: dict[str, list[str]] = {
    "greeting": [
        "Привет! Я Delta 1 Flash. Могу вести чат, объяснять обучение и показывать простые примеры кода.",
        "Я на месте. Напиши задачу: обучение, код, команды запуска или вопрос по модели.",
    ],
    "answer_style": [
        "Буду отвечать кратко, честно и по делу. Если не уверен или данных мало — скажу прямо. Если нужен код, покажу пример.",
    ],
    "answer_style_en": [
        "I will answer briefly and honestly, in your language, with code examples when useful. If I am unsure, I will say so.",
    ],
    "training": [
        "Обучение реальное: текст превращается в UTF-8 byte токены, из них делаются пары контекст -> следующий токен, softmax считает вероятности, cross entropy даёт loss, а SGD обновляет веса. В консоли видны batch loss, eval loss и ETA.",
    ],
    "forever": [
        "Бесконечное обучение: python -m delta1flash.train --forever --steps 500. Остановить можно Ctrl+C; модель сохраняется после каждого цикла.",
    ],
    "capabilities": [
        "Я умею вести терминальный чат, сохранять историю JSONL, учиться на /learn примерах, объяснять обучение, показывать код и запускать бесконечное обучение через train.py --forever.",
    ],
    "code_python": [
        "```python\ndef hello(name):\n    return f\"Привет, {name}!\"\n\nprint(hello(\"мир\"))\n```",
        "```python\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n```",
    ],
    "code_js": [
        "```javascript\nfunction add(a, b) {\n  return a + b;\n}\n\nconsole.log(add(2, 3));\n```",
    ],
    "alphabet": [
        "Русский алфавит: а б в г д е ё ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я. Если пишешь одну букву, я понимаю её как букву русского алфавита.",
    ],
    "dictionary": [
        "Словарь понимания лежит в data/chat_intents.jsonl и WORDS_RU.md. Там написано: фраза -> intent -> значение -> как говорить. Можно добавить свои фразы и переобучить: python -m delta1flash.train_chat.",
    ],
    "thanks": [
        "Пожалуйста. Если хочешь, могу дальше улучшить обучение, чат или примеры кода.",
    ],
    "frustration": [
        "Понял, прошлый ответ был плохой. Я убрал повторяющиеся фразы из обычного режима и добавил обучаемую intent-модель. Напиши конкретный вопрос — отвечу нормально.",
        "Да, повторения раздражают. В hybrid-режиме я теперь фильтрую сырой вывод модели и использую обученную chat-модель для стабильного ответа.",
    ],
    "unknown": [
        "Не понял сообщение. Напиши чуть подробнее: вопрос про обучение, код, чат или команду запуска.",
        "Слишком мало смысла в сообщении. Можешь переформулировать полным вопросом?",
    ],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_chat_model_path() -> Path:
    return project_root() / "runs" / "delta1flash_chat_model.json"


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def words(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            current.append(ch)
        elif current:
            out.append("".join(current))
            current.clear()
    if current:
        out.append("".join(current))
    return out


def extract_features(text: str, hash_size: int) -> dict[int, float]:
    text = text.lower().strip()
    squashed = squash_repeats(text)
    variants = [text] if squashed == text else [text, squashed]
    feats: dict[int, float] = {}

    def add(name: str, value: float = 1.0) -> None:
        idx = stable_hash(name, hash_size)
        feats[idx] = feats.get(idx, 0.0) + value

    add(f"len:{min(len(text) // 4, 20)}")
    for variant_i, variant in enumerate(variants):
        weight = 1.0 if variant_i == 0 else 0.8
        for w in words(variant):
            add(f"w:{w}", weight)
            if len(w) >= 4:
                add(f"wp:{w[:3]}", 0.5 * weight)
                add(f"ws:{w[-3:]}", 0.5 * weight)
        padded = f"^{variant}$"
        for n in (1, 2, 3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                add(f"c{n}:{padded[i:i+n]}", 0.35 * weight)
    if text and all(ch in RUSSIAN_LETTERS for ch in text):
        add(f"ru_letters_len:{min(len(text), 12)}", 0.8)
        if len(text) == 1:
            add("single_ru_letter", 2.0)
    # L2-ish normalization keeps long gibberish from dominating.
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {idx: value / norm for idx, value in feats.items()}


def softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def dot(weights: Sequence[float], features: dict[int, float]) -> float:
    return sum(weights[i] * v for i, v in features.items())


def predict(model: dict[str, Any], text: str) -> tuple[str, float, dict[str, float]]:
    labels = model["labels"]
    hash_size = int(model["hash_size"])
    weights = model["weights"]
    bias = model["bias"]
    feats = extract_features(text, hash_size)
    logits = [dot(weights[i], feats) + bias[i] for i in range(len(labels))]
    probs = softmax(logits)
    best_i = max(range(len(labels)), key=lambda i: probs[i])
    scores = {labels[i]: probs[i] for i in range(len(labels))}
    return labels[best_i], probs[best_i], scores


def choose_response(model: dict[str, Any], label: str, message: str) -> str:
    q = message.lower().strip()
    if label == "alphabet":
        if "алфав" in q or "азбук" in q or "все бук" in q or "список" in q:
            return f"Русский алфавит: {RUSSIAN_ALPHABET_TEXT}. Всего 33 буквы: 10 гласных, 21 согласная и знаки ъ/ь."
        q_words = words(q)
        for token in reversed(q_words):
            if len(token) == 1 and token in RUSSIAN_LETTERS:
                name = LETTER_NAMES.get(token, token)
                pos = RUSSIAN_LETTERS.index(token) + 1
                return f"Это буква «{token}» ({name}), номер {pos} в русском алфавите. Алфавит: {RUSSIAN_ALPHABET_TEXT}."
        cleaned = "".join(ch for ch in q if ch in RUSSIAN_LETTERS)
        if len(cleaned) == 1:
            name = LETTER_NAMES.get(cleaned, cleaned)
            pos = RUSSIAN_LETTERS.index(cleaned) + 1
            return f"Это буква «{cleaned}» ({name}), номер {pos} в русском алфавите. Алфавит: {RUSSIAN_ALPHABET_TEXT}."
        if cleaned and len(cleaned) <= 8 and all(ch in RUSSIAN_LETTERS for ch in cleaned):
            return f"Это набор русских букв: {' '.join(cleaned)}. Если хотел вопрос, напиши словами. Алфавит: {RUSSIAN_ALPHABET_TEXT}."
        return f"Русский алфавит: {RUSSIAN_ALPHABET_TEXT}. Всего 33 буквы: 10 гласных, 21 согласная и знаки ъ/ь."
    if label == "code_python":
        if any(word in q for word in ["add", "слож", "сум", "plus"]):
            return "```python\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n```"
        return "```python\ndef hello(name):\n    return f\"Привет, {name}!\"\n\nprint(hello(\"мир\"))\n```"
    responses = model.get("responses", RESPONSES).get(label) or RESPONSES.get(label) or RESPONSES["unknown"]
    idx = stable_hash(message + label, len(responses))
    return responses[idx]


def load_chat_model(path: Path | None = None) -> dict[str, Any] | None:
    path = path or default_chat_model_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_examples(paths: Sequence[Path]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = str(obj.get("text") or obj.get("question") or "").strip()
                label = str(obj.get("label") or "").strip()
                if text and label:
                    examples.append({"text": text, "label": label})
    return examples


def save_model(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
