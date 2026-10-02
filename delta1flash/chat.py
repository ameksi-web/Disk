"""Interactive terminal chat for Delta 1 Flash.

Run from the repository root:

    python -m delta1flash.chat

Or directly from the delta1flash folder:

    python chat.py

This is a real local chat loop: the model is loaded once, each user message is
processed, a reply is generated, and the conversation is saved to JSONL history.
Because Delta 1 Flash is a tiny educational model, the default mode is hybrid:
it tries the model first, filters broken/repetitive text, and falls back to a
small local answer engine plus user-taught memory.  Use --mode model to see raw
model behaviour only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from delta1flash.chat_intent import LETTER_NAMES, RUSSIAN_ALPHABET_TEXT, RUSSIAN_LETTERS, choose_response, default_chat_model_path, load_chat_model, predict
    from delta1flash.generate import default_model_path, generate_from_payload, load_model
else:  # pragma: no cover
    from .chat_intent import LETTER_NAMES, RUSSIAN_ALPHABET_TEXT, RUSSIAN_LETTERS, choose_response, default_chat_model_path, load_chat_model, predict
    from .generate import default_model_path, generate_from_payload, load_model

END_MARKER = "§END§"
STOP_MARKERS = [
    END_MARKER,
    "\nПользователь:",
    "\nUser:",
    "\nDeltaChat:",
    "\n@@C1@@:",
    "\n@@C2@@:",
    "\n@@C2EN@@:",
    "\n@@C3@@:",
    "\n@@C4@@:",
    "\n@@C5@@:",
    "\n@@C6@@:",
    "\nВопрос:",
    "\nQuestion:",
    "\nTask:",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_history_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return project_root() / "chats" / f"chat_{stamp}.jsonl"


def default_memory_path() -> Path:
    return project_root() / "data" / "chat_memory.jsonl"


def normalize(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wа-яА-ЯёЁ]+", text.lower()) if len(w) > 1}


def similarity(a: str, b: str) -> float:
    aa = normalize(a)
    bb = normalize(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def load_memory(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    items: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip()
            if q and a:
                items.append({"question": q, "answer": a})
    return items


def append_memory(path: Path, question: str, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.time(), "question": question.strip(), "answer": answer.strip()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def memory_lookup(message: str, memory: Sequence[dict[str, str]], threshold: float = 0.45) -> str | None:
    best_score = 0.0
    best_answer: str | None = None
    for item in memory:
        score = similarity(message, item.get("question", ""))
        if score > best_score:
            best_score = score
            best_answer = item.get("answer", "")
    return best_answer if best_answer and best_score >= threshold else None


def choose_reply_marker(message: str) -> str:
    q = message.lower().strip()
    if "python" in q or "код" in q or "code" in q:
        return "@@C6@@:"
    if q in {"привет", "hello", "hi", "здравствуй"} or q.startswith("привет "):
        return "@@C1@@:"
    if "как ты будешь отвечать" in q or "how will you answer" in q:
        return "@@C2@@:" if "how will" not in q else "@@C2EN@@:"
    if "алфав" in q or "букв" in q or (len(q) == 1 and q in RUSSIAN_LETTERS):
        return "@@C4@@:"
    if "слова" in q or "значен" in q or "словар" in q or "интент" in q or "понимаешь" in q:
        return "@@C4@@:"
    if "обуч" in q or "train" in q or "loss" in q:
        return "@@C3@@:"
    if "умеешь" in q or "что ты" in q or "help" in q:
        return "@@C4@@:"
    if "hello" in q:
        return "@@C6@@:"
    return "@@C5@@:"


def build_prompt(message: str, history: Sequence[dict[str, str]], max_history: int) -> str:
    marker = choose_reply_marker(message)
    lines = [
        "Диалог Delta 1 Flash.",
        "Правило: отвечай кратко, честно, на языке пользователя, с кодом если просят.",
    ]
    for item in history[-max_history:]:
        user = item.get("user", "").strip()
        assistant = item.get("assistant", "").strip()
        if user and assistant:
            lines.append(f"Пользователь: {user}")
            lines.append(f"DeltaChat: {assistant} {END_MARKER}")
    lines.append(f"Пользователь: {message.strip()}")
    lines.append(marker)
    return "\n".join(lines)


def trim_reply(full_text: str, prompt: str) -> str:
    reply = full_text[len(prompt) :].strip()
    stop = len(reply)
    for marker in STOP_MARKERS:
        idx = reply.find(marker)
        if idx != -1:
            stop = min(stop, idx)
    reply = reply[:stop].strip()
    reply = reply.replace(END_MARKER, "").strip()
    return reply


def looks_repetitive(text: str) -> bool:
    low = text.lower()
    bad_fragments = [
        "данных маленькие примеры",
        "маленькие примеры кода" * 2,
        "привет, {name}",
        "print(hello" if "```" not in text else "\0",
    ]
    if any(fragment in low for fragment in bad_fragments):
        return True
    words = re.findall(r"[\wа-яА-ЯёЁ{}()]+", low)
    if len(words) >= 18 and len(set(words)) / len(words) < 0.35:
        return True
    chunks = [low[i : i + 24] for i in range(0, max(0, len(low) - 24), 12)]
    return len(chunks) > 8 and len(set(chunks)) < len(chunks) * 0.55


def rule_reply(message: str, history: Sequence[dict[str, str]], memory: Sequence[dict[str, str]]) -> str:
    remembered = memory_lookup(message, memory)
    if remembered:
        return remembered

    q = message.lower().strip()
    if "python" in q or "код" in q or "code" in q:
        if "add" in q or "слож" in q or "сум" in q:
            return "```python\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n```"
        return "```python\ndef hello(name):\n    return f\"Привет, {name}!\"\n\nprint(hello(\"мир\"))\n```"
    if q in {"привет", "hello", "hi", "здравствуй"} or q.startswith("привет "):
        return "Привет! Я Delta 1 Flash. Могу вести локальный чат, объяснять обучение и показывать простые примеры кода."
    if "как ты будешь отвечать" in q:
        return "Ок. Буду отвечать кратко, честно, на языке пользователя, с примерами кода. Если не уверен — прямо скажу."
    if "how will you answer" in q:
        return "OK. I will answer briefly and honestly, in your language, with code examples when useful."
    if "бескон" in q or "forever" in q or "постоян" in q:
        return (
            "Бесконечное обучение запускается так: python -m delta1flash.train --forever --steps 500. "
            "Остановить можно Ctrl+C; модель сохраняется после каждого цикла."
        )
    if "алфав" in q or "букв" in q or (len(q) == 1 and q in RUSSIAN_LETTERS):
        if len(q) == 1 and q in RUSSIAN_LETTERS:
            return f"Это буква «{q}» ({LETTER_NAMES.get(q, q)}). Русский алфавит: {RUSSIAN_ALPHABET_TEXT}."
        return f"Русский алфавит: {RUSSIAN_ALPHABET_TEXT}. Все 33 буквы добавлены в data/chat_intents.jsonl и WORDS_RU.md."
    if "слова" in q or "значен" in q or "словар" in q or "интент" in q or "понимаешь" in q:
        return (
            "Список слов и значений лежит в WORDS_RU.md. Коротко: привет/ау -> greeting, "
            "алфавит/буква а -> alphabet, обучение/loss -> training, бесконечное обучение -> forever, "
            "код/python -> code_python, ругательство/жалоба -> frustration, мусорный ввод -> unknown."
        )
    if "обуч" in q or "train" in q or "loss" in q:
        return (
            "Обучение реальное: текст превращается в UTF-8 byte токены, из них делаются пары "
            "контекст -> следующий токен, softmax считает вероятности, cross entropy даёт loss, "
            "а SGD обновляет веса. В консоли видны batch loss, eval loss, ETA и путь сохранения модели."
        )
    if "умеешь" in q or "что ты" in q or "help" in q:
        return (
            "Я умею: 1) вести терминальный чат; 2) помнить последние реплики сессии; "
            "3) сохранять историю JSONL; 4) учиться на добавленных /learn примерах; "
            "5) показывать простые примеры кода; 6) запускать бесконечное обучение через train.py --forever."
        )
    tail = ""
    if history:
        tail = f" Я помню последнюю тему: {history[-1].get('user', '')[:60]}"
    return (
        f"Я понял запрос: «{message[:120]}».{tail} "
        "Я маленькая локальная модель, поэтому лучше отвечаю, когда вопрос конкретный: про обучение, код или команды запуска."
    )


def intent_reply(
    intent_model: dict[str, Any] | None,
    message: str,
    *,
    threshold: float,
) -> tuple[str | None, str, float]:
    if intent_model is None:
        return None, "", 0.0
    label, confidence, _ = predict(intent_model, message)
    # unknown and frustration are valid learned states even with lower confidence.
    if confidence >= threshold or label in {"unknown", "frustration"}:
        return choose_response(intent_model, label, message), label, confidence
    return None, label, confidence


def model_reply(
    payload: dict[str, Any],
    message: str,
    *,
    history: Sequence[dict[str, str]],
    max_history: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    prompt = build_prompt(message, history, max_history=max_history)
    length = len(prompt.encode("utf-8")) + max_new_tokens
    full_text = generate_from_payload(payload, start=prompt, length=length, temperature=temperature, seed=seed)
    return trim_reply(full_text, prompt)


def chat_once(
    payload: dict[str, Any],
    message: str,
    *,
    history: Sequence[dict[str, str]] = (),
    memory: Sequence[dict[str, str]] = (),
    intent_model: dict[str, Any] | None = None,
    intent_threshold: float = 0.42,
    max_history: int = 3,
    max_new_tokens: int = 500,
    temperature: float = 0.03,
    seed: int = 77,
    mode: str = "hybrid",
) -> str:
    remembered = memory_lookup(message, memory)
    if remembered:
        return remembered

    if mode == "rules":
        return rule_reply(message, history, memory)

    if mode == "hybrid":
        learned_reply, label, confidence = intent_reply(intent_model, message, threshold=intent_threshold)
        if learned_reply is not None:
            return learned_reply
        # If the intent model exists but is unsure, do not let the raw byte-LM
        # hallucinate repeated phrases.  Ask for a clearer message instead.
        if intent_model is not None:
            return (
                "Не понял сообщение достаточно уверенно "
                f"(intent={label or 'none'}, confidence={confidence:.2f}). "
                "Напиши подробнее: вопрос про обучение, код, чат или команду запуска."
            )

    reply = model_reply(
        payload,
        message,
        history=history,
        max_history=max_history,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    if mode == "model":
        return reply or "[model produced empty reply]"

    if not reply or len(reply) < 8 or looks_repetitive(reply):
        return rule_reply(message, history, memory)
    return reply


def save_turn(history_path: Path, user: str, assistant: str) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.time(), "user": user, "assistant": assistant}
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def print_help() -> None:
    print("Команды чата:")
    print("  /help                  показать помощь")
    print("  /reset                 очистить память текущего диалога")
    print("  /temp 0.05             изменить temperature")
    print("  /mode hybrid|model|rules переключить режим ответа")
    print("  /learn вопрос => ответ  добавить пример в память чата")
    print("  /exit                  выйти")


def interactive_chat(args: argparse.Namespace) -> int:
    print(f"Загружаю модель: {args.model}")
    payload = load_model(args.model)
    intent_model = load_chat_model(args.chat_model)
    history: list[dict[str, str]] = []
    history_path = args.history or default_history_path()
    memory_path = args.memory
    memory = load_memory(memory_path)
    temperature = args.temperature
    mode = args.mode

    print("Delta 1 Flash chat готов. Напиши сообщение или /help. Для выхода: /exit")
    print(f"Режим: {mode}. История: {history_path}. Память: {memory_path}")
    print(f"Intent-модель: {args.chat_model if intent_model else 'не найдена, fallback rules'}")
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
            print("Память текущего диалога очищена.")
            continue
        if message.startswith("/mode"):
            parts = message.split(maxsplit=1)
            if len(parts) == 2 and parts[1] in {"hybrid", "model", "rules"}:
                mode = parts[1]
                print(f"mode = {mode}")
            else:
                print("Пример: /mode hybrid")
            continue
        if message.startswith("/temp"):
            parts = message.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"temperature = {temperature}")
                except ValueError:
                    print("Нужно число, пример: /temp 0.05")
            else:
                print(f"temperature = {temperature}")
            continue
        if message.startswith("/learn"):
            payload_text = message[len("/learn") :].strip()
            if "=>" not in payload_text:
                print("Формат: /learn вопрос => ответ")
                continue
            question, answer = [part.strip() for part in payload_text.split("=>", 1)]
            if not question or not answer:
                print("Нужны и вопрос, и ответ.")
                continue
            append_memory(memory_path, question, answer)
            memory = load_memory(memory_path)
            print("Запомнил пример. Он будет использоваться в чате сразу; для весов модели запусти train.py --forever или обычное обучение на корпусе с памятью.")
            continue

        reply = chat_once(
            payload,
            message,
            history=history,
            memory=memory,
            intent_model=intent_model,
            intent_threshold=args.intent_threshold,
            max_history=args.max_history,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            seed=args.seed + len(history),
            mode=mode,
        )
        print(f"Delta: {reply}")
        history.append({"user": message, "assistant": reply})
        save_turn(history_path, message, reply)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive terminal chat with Delta 1 Flash")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--message", help="single message mode; if omitted, starts interactive chat")
    parser.add_argument("--history", type=Path, help="JSONL history path; default is chats/chat_TIMESTAMP.jsonl")
    parser.add_argument("--memory", type=Path, default=default_memory_path(), help="JSONL learned memory path")
    parser.add_argument("--chat-model", type=Path, default=default_chat_model_path(), help="trained intent model for stable chat")
    parser.add_argument("--intent-threshold", type=float, default=0.42, help="confidence threshold for intent model")
    parser.add_argument("--mode", choices=["hybrid", "model", "rules"], default="hybrid", help="hybrid uses trained intent model; model is raw byte-LM; rules is deterministic")
    parser.add_argument("--max-history", type=int, default=3, help="how many previous turns to include in the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="generation budget after the prompt")
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args(argv)

    if args.message:
        payload = load_model(args.model)
        memory = load_memory(args.memory)
        intent_model = load_chat_model(args.chat_model)
        reply = chat_once(
            payload,
            args.message,
            memory=memory,
            intent_model=intent_model,
            intent_threshold=args.intent_threshold,
            max_history=args.max_history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            mode=args.mode,
        )
        print(reply)
        if args.history:
            save_turn(args.history, args.message, reply)
        return 0

    return interactive_chat(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
