"""Print the chat words/phrases Delta 1 Flash is trained to understand."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

MEANINGS = {
    "greeting": "приветствие / проверка связи",
    "answer_style": "как модель должна отвечать",
    "answer_style_en": "answer style in English",
    "alphabet": "буквы русского алфавита, азбука, одиночные буквы",
    "dictionary": "словарь слов, значений и команд",
    "training": "обучение, loss, SGD, улучшение модели",
    "forever": "бесконечное/постоянное обучение",
    "capabilities": "возможности и команды модели",
    "code_python": "запрос Python-кода",
    "code_js": "запрос JavaScript-кода",
    "thanks": "благодарность / подтверждение",
    "frustration": "недовольство, ругань, жалоба на ответ",
    "unknown": "мусорный или слишком короткий ввод",
}

HOW_TO_SAY = {
    "greeting": "Пиши: привет, ау, алло, ты тут, ping.",
    "answer_style": "Пиши: как ты будешь отвечать, как с тобой говорить.",
    "alphabet": "Пиши: алфавит, буква а, буква я, все буквы русского алфавита. Одиночная буква тоже понимается.",
    "dictionary": "Пиши: какие слова ты понимаешь, словарь команд, слова и значения.",
    "training": "Пиши: как обучается модель, что такое loss, как улучшить модель.",
    "forever": "Пиши: бесконечное обучение, train.py бесконечно, continuous training.",
    "capabilities": "Пиши: что ты умеешь, команды, помощь.",
    "code_python": "Пиши: напиши код hello на python, пример python, функция сложения.",
    "code_js": "Пиши: javascript код, js function add.",
    "thanks": "Пиши: спасибо, ок, понял, круто.",
    "frustration": "Пиши: плохо отвечаешь, хватит повторять, заебал — чат поймёт жалобу и не будет повторять мусор.",
    "unknown": "Если ввод бессмысленный, модель попросит переформулировать.",
}


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "chat_intents.jsonl"


def load_items(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            grouped[obj["label"]].append(obj["text"])
    return dict(grouped)


def render_markdown(grouped: dict[str, list[str]]) -> str:
    lines = ["# Слова, значения и как говорить с Delta 1 Flash", ""]
    lines.append("Это не список всех слов русского языка. Это обучающие фразы intent-модели чата: по ним чат понимает смысл короткого сообщения и выбирает нормальный ответ.")
    lines.append("")
    for label in sorted(grouped):
        lines.append(f"## `{label}` — {MEANINGS.get(label, 'значение не описано')}")
        if label in HOW_TO_SAY:
            lines.append(f"Как говорить: {HOW_TO_SAY[label]}")
        lines.append("")
        for phrase in sorted(set(grouped[label])):
            lines.append(f"- `{phrase}`")
        lines.append("")
    lines.append("## Как добавить свои слова")
    lines.append("")
    lines.append("1. Добавь строки в `data/chat_intents.jsonl` в формате JSONL:")
    lines.append("")
    lines.append('```json')
    lines.append('{"text":"твоя фраза","label":"training"}')
    lines.append('```')
    lines.append("")
    lines.append("2. Переобучи chat-модель:")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m delta1flash.train_chat --data data/chat_intents.jsonl --out runs/delta1flash_chat_model.json --epochs 100")
    lines.append("```")
    lines.append("")
    lines.append("3. Запусти чат:")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m delta1flash.chat")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show Delta 1 Flash chat words and meanings")
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--out", type=Path, help="write markdown to file")
    args = parser.parse_args(argv)
    text = render_markdown(load_items(args.data))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"saved: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
