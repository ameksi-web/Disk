"""Train Delta 1 Flash, a tiny character/byte-level language model.

Delta 1 Flash intentionally uses only the Python standard library.  It is a real
trainable neural model: a context selects a row of trainable logits, softmax
turns the logits into next-token probabilities, and stochastic gradient descent
updates the row from data.

Recommended launch from the repository root:

    python -m delta1flash.train

Direct launch also works, including on Windows from the delta1flash folder:

    python train.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

if __package__ in (None, ""):
    # Support `python train.py` from inside the package directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from delta1flash import DEFAULT_MODEL_NAME, __version__
else:  # pragma: no cover - exercised when launched with -m
    from . import DEFAULT_MODEL_NAME, __version__

Context = Tuple[int, ...]
Example = Tuple[Context, int]
Rows = Dict[str, List[float]]

UNK_CONTEXT = "<UNK>"
FALLBACK_CHAR = "□"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_corpus_path() -> Path:
    root = project_root()
    for candidate in [
        root / "data" / "demo_focused_corpus.txt",
        root / "data" / "demo_universal_corpus.txt",
        root / "data" / "universal_code_seed.txt",
        root / "data" / "seed_corpus.txt",
    ]:
        if candidate.exists():
            return candidate
    return root / "data" / "seed_corpus.txt"


def default_model_path() -> Path:
    return project_root() / "runs" / "delta1flash_demo_model.json"


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def explain_token_mode(token_mode: str, max_vocab: int) -> str:
    if token_mode == "byte":
        return (
            "UTF-8 byte: любой текст и код переводятся в байты 0..255. "
            "Так модель видит русский, английский, китайский, арабский, эмодзи, "
            "пробелы, переносы строк, скобки и отступы кода без отдельного токенизатора."
        )
    return (
        f"char: модель строит словарь символов из корпуса. max_vocab={max_vocab}; "
        f"редкие символы заменяются на {FALLBACK_CHAR!r}."
    )


def print_training_plan(
    args: argparse.Namespace,
    *,
    text: str,
    vocab: Sequence[str],
    token_ids: Sequence[int],
    examples: Sequence[Example],
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
) -> None:
    print("=" * 72, flush=True)
    print(f"Delta 1 Flash - обучение модели: {args.model_name}", flush=True)
    print("=" * 72, flush=True)
    print("Что обучается:", flush=True)
    print("  - маленькая языковая нейросеть token_context_softmax_neural_net", flush=True)
    print("  - обучаемые параметры: rows[контекст] -> логиты следующего токена", flush=True)
    print("  - контекст: последние токены перед предсказываемым токеном", flush=True)
    print("Как обучается:", flush=True)
    print("  - задача: предсказать следующий токен по предыдущим", flush=True)
    print("  - вероятность считается через softmax", flush=True)
    print("  - ошибка: cross entropy / negative log likelihood", flush=True)
    print("  - оптимизация: SGD, обновление весов по градиенту", flush=True)
    print("  - backoff: кроме полного контекста учатся короткие суффиксы", flush=True)
    print(f"  - warm-start counts: {'включён' if args.warm_start_counts else 'выключен'}", flush=True)
    print("Данные:", flush=True)
    print(f"  - corpus: {args.corpus}", flush=True)
    print(f"  - output model: {args.out}", flush=True)
    print(f"  - символов текста: {fmt_int(len(text))}", flush=True)
    print(f"  - токенов: {fmt_int(len(token_ids))}", flush=True)
    print(f"  - обучающих примеров всего: {fmt_int(len(examples))}", flush=True)
    print(f"  - train/eval split: {fmt_int(len(train_examples))} / {fmt_int(len(eval_examples))}", flush=True)
    print(f"  - token mode: {args.token_mode}", flush=True)
    print(f"  - token mode detail: {explain_token_mode(args.token_mode, args.max_vocab)}", flush=True)
    print("Параметры запуска:", flush=True)
    print(f"  - steps: {fmt_int(args.steps)}", flush=True)
    print(f"  - batch size: {fmt_int(args.batch_size)}", flush=True)
    print(f"  - context length: {args.context_length}", flush=True)
    print(f"  - learning rate: {args.learning_rate}", flush=True)
    print(f"  - vocab size: {fmt_int(len(vocab))}", flush=True)
    print(f"  - seed: {args.seed}", flush=True)
    print(f"  - progress/eval every: {args.sample_every} steps", flush=True)
    print(f"  - eval limit: {args.eval_limit} examples", flush=True)
    if args.metrics_out:
        print(f"  - metrics JSONL: {args.metrics_out}", flush=True)
    if args.resume_model:
        print(f"  - resume model: {args.resume_model}", flush=True)
    print(f"  - auto resume: {'on' if args.auto_resume else 'off'}", flush=True)
    if args.forever:
        limit = "без ограничения" if args.max_cycles == 0 else str(args.max_cycles)
        print(f"  - FOREVER mode: включён, cycles={limit}, save every {args.save_every_cycles}", flush=True)
    print("=" * 72, flush=True)


def read_text(path: Path, limit_chars: int = 0) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text[:limit_chars] if limit_chars and limit_chars > 0 else text


def build_char_vocab(text: str, max_vocab: int = 512) -> Tuple[List[str], List[int], str]:
    """Build a compact character vocabulary and return token ids.

    If max_vocab <= 0, all characters are kept.  Otherwise rare characters are
    mapped to FALLBACK_CHAR.  For true multilingual training, use --token-mode
    byte, which represents every UTF-8 text with only 256 tokens.
    """

    if not text:
        raise ValueError("Training corpus is empty")

    counter = Counter(text)
    if max_vocab <= 0 or len(counter) + 1 <= max_vocab:
        chars = sorted(counter)
        if FALLBACK_CHAR not in chars:
            chars.append(FALLBACK_CHAR)
            chars = sorted(chars)
    else:
        protected = [ch for ch in ["\n", " "] if ch in counter]
        most_common = [ch for ch, _ in counter.most_common(max(1, max_vocab - 1))]
        chars: List[str] = []
        for ch in protected + most_common:
            if ch not in chars and ch != FALLBACK_CHAR:
                chars.append(ch)
            if len(chars) >= max_vocab - 1:
                break
        chars.append(FALLBACK_CHAR)
        chars = sorted(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    normalized = "".join(ch if ch in stoi else FALLBACK_CHAR for ch in text)
    ids = [stoi[ch] for ch in normalized]
    return chars, ids, normalized


def build_byte_vocab(text: str) -> Tuple[List[str], List[int], str]:
    """Represent text as UTF-8 bytes.

    Byte mode is the recommended mode for "all languages + code" because every
    Unicode character, emoji, bracket, newline, indentation and source-code sign
    can be represented without a gigantic character vocabulary.
    """

    if not text:
        raise ValueError("Training corpus is empty")
    ids = list(text.encode("utf-8", errors="replace"))
    vocab = [str(i) for i in range(256)]
    return vocab, ids, text


def prepare_tokens(text: str, token_mode: str, max_vocab: int) -> Tuple[List[str], List[int], str]:
    if token_mode == "byte":
        return build_byte_vocab(text)
    if token_mode == "char":
        return build_char_vocab(text, max_vocab=max_vocab)
    raise ValueError(f"Unknown token mode: {token_mode}")


def encode_start(start: str, vocab: Sequence[str], token_mode: str) -> List[int]:
    if token_mode == "byte":
        return list(start.encode("utf-8", errors="replace")) or [32]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    fallback_id = stoi.get(FALLBACK_CHAR, stoi.get(" ", 0))
    return [stoi.get(ch, fallback_id) for ch in start] or [fallback_id]


def decode_ids(ids: Sequence[int], vocab: Sequence[str], token_mode: str) -> str:
    if token_mode == "byte":
        return bytes(max(0, min(255, i)) for i in ids).decode("utf-8", errors="replace")
    return "".join(vocab[i] if 0 <= i < len(vocab) else FALLBACK_CHAR for i in ids)


def context_key(context: Sequence[int]) -> str:
    return ",".join(str(x) for x in context)


def make_examples_from_ids(ids: Sequence[int], context_length: int) -> List[Example]:
    if context_length < 1:
        raise ValueError("context_length must be at least 1")
    if len(ids) <= context_length:
        raise ValueError("Need more tokens than context_length to train")
    return [(tuple(ids[i - context_length : i]), ids[i]) for i in range(context_length, len(ids))]


def init_row(vocab_size: int, rng: random.Random) -> List[float]:
    scale = 0.01
    return [rng.uniform(-scale, scale) for _ in range(vocab_size)]


def get_row(rows: MutableMapping[str, List[float]], key: str, vocab_size: int, rng: random.Random) -> List[float]:
    row = rows.get(key)
    if row is None:
        row = init_row(vocab_size, rng)
        rows[key] = row
    return row


def suffix_contexts(context: Sequence[int]) -> List[Context]:
    # Longest context first for lookup; shorter suffixes are backoff rows.
    return [tuple(context[-n:]) for n in range(len(context), 0, -1)]


def lookup_row(rows: Rows, context: Sequence[int], vocab_size: int) -> List[float]:
    for suffix in suffix_contexts(context):
        row = rows.get(context_key(suffix))
        if row is not None:
            return row
    return rows.get(UNK_CONTEXT) or [0.0 for _ in range(vocab_size)]


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def loss_on_examples(rows: Rows, examples: Sequence[Example], vocab_size: int, limit: int = 4096) -> float:
    if not examples:
        return float("nan")
    total = 0.0
    n = min(len(examples), limit)
    for context, y in examples[:n]:
        row = lookup_row(rows, context, vocab_size)
        probs = softmax(row)
        total += -math.log(max(probs[y], 1e-12))
    return total / n


def rows_from_counts(examples: Sequence[Example], vocab_size: int, *, alpha: float = 0.05) -> Rows:
    """Warm-start logits from empirical next-token counts.

    The returned rows are still trainable parameters; SGD can continue updating
    them.  Count warm-start makes the tiny demo model much more coherent while
    preserving real gradient training.
    """

    count_rows: Dict[str, List[float]] = {UNK_CONTEXT: [alpha for _ in range(vocab_size)]}
    for context, y in examples:
        count_rows[UNK_CONTEXT][y] += 1.0
        for ctx in suffix_contexts(context):
            key = context_key(ctx)
            row = count_rows.get(key)
            if row is None:
                row = [alpha for _ in range(vocab_size)]
                count_rows[key] = row
            row[y] += 1.0

    rows: Rows = {}
    for key, counts in count_rows.items():
        total = sum(counts)
        # log-probabilities are logits: softmax(log p) == p.
        rows[key] = [math.log(c / total) for c in counts]
    return rows


def train_context_softmax(
    examples: Sequence[Example],
    vocab_size: int,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    sample_every: int,
    initial_rows: Rows | None = None,
    eval_examples: Sequence[Example] | None = None,
    eval_limit: int = 1024,
    verbose: bool = True,
) -> Tuple[Rows, List[Dict[str, float]]]:
    rng = random.Random(seed)
    rows: Rows = initial_rows if initial_rows is not None else {UNK_CONTEXT: init_row(vocab_size, rng)}
    log: List[Dict[str, float]] = []

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    t0 = time.time()
    for step in range(1, steps + 1):
        batch_loss = 0.0
        for _ in range(batch_size):
            context, y = examples[rng.randrange(len(examples))]
            contexts = suffix_contexts(context)
            step_lr = learning_rate / (batch_size * len(contexts))
            sample_loss = 0.0
            for ctx in contexts:
                row = get_row(rows, context_key(ctx), vocab_size, rng)
                probs = softmax(row)
                if ctx == context:
                    sample_loss = -math.log(max(probs[y], 1e-12))
                probs[y] -= 1.0  # d loss / d logits for softmax + cross entropy
                for j, grad in enumerate(probs):
                    row[j] -= step_lr * grad
            batch_loss += sample_loss

        if step == 1 or step == steps or (sample_every and step % sample_every == 0):
            elapsed = time.time() - t0
            current_loss = batch_loss / batch_size
            percent = 100.0 * step / max(1, steps)
            eta = (elapsed / step) * (steps - step) if step else 0.0
            eval_loss = (
                loss_on_examples(rows, eval_examples, vocab_size, limit=eval_limit)
                if eval_examples is not None
                else None
            )
            event = {
                "step": float(step),
                "batch_loss": current_loss,
                "eval_loss": eval_loss,
                "elapsed_sec": elapsed,
                "eta_sec": eta,
                "progress_percent": percent,
                "trained_context_rows": float(len(rows)),
            }
            log.append(event)
            if verbose:
                eval_part = f" | eval loss {eval_loss:.4f}" if eval_loss is not None else ""
                print(
                    f"[train] step {step}/{steps} ({percent:5.1f}%) | "
                    f"batch loss {current_loss:.4f}{eval_part} | rows {len(rows)} | "
                    f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)}",
                    flush=True,
                )

    return rows, log


def sample_ids(
    rows: Rows,
    vocab_size: int,
    *,
    start_ids: Sequence[int],
    length: int,
    temperature: float,
    seed: int,
    context_length: int,
) -> List[int]:
    rng = random.Random(seed)
    out = list(start_ids) or [32 if vocab_size > 32 else 0]
    while len(out) < length:
        context = out[-context_length:]
        if len(context) < context_length:
            context = [32 if vocab_size > 32 else 0] * (context_length - len(context)) + context
        row = lookup_row(rows, context, vocab_size)
        logits = [v / max(temperature, 1e-6) for v in row]
        probs = softmax(logits)
        r = rng.random()
        acc = 0.0
        next_id = vocab_size - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                next_id = i
                break
        out.append(next_id)
    return out[:length]


def sample_text(
    rows: Rows,
    vocab: Sequence[str],
    *,
    token_mode: str,
    start: str = "Delta 1 Flash: ",
    length: int = 500,
    temperature: float = 0.8,
    seed: int = 7,
    context_length: int = 3,
) -> str:
    ids = sample_ids(
        rows,
        len(vocab),
        start_ids=encode_start(start, vocab, token_mode),
        length=length,
        temperature=temperature,
        seed=seed,
        context_length=context_length,
    )
    return decode_ids(ids, vocab, token_mode)


def save_model(
    out_path: Path,
    *,
    model_name: str,
    vocab: Sequence[str],
    rows: Rows,
    config: Dict[str, object],
    training_log: Sequence[Dict[str, float]],
    corpus_path: Path,
    final_sample: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "project": "delta1flash",
        "version": __version__,
        "architecture": "token_context_softmax_neural_net",
        "created_at_unix": time.time(),
        "corpus": str(corpus_path),
        "vocabulary": list(vocab),
        "rows": {key: [round(v, 7) for v in row] for key, row in sorted(rows.items())},
        "config": config,
        "training_log": list(training_log),
        "final_sample": final_sample,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_rows_from_model(path: Path, *, expected_vocab_size: int, token_mode: str, context_length: int) -> Rows:
    """Load trainable rows from an existing Delta 1 Flash model for resume/forever training."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    vocab = payload.get("vocabulary", [])
    if len(vocab) != expected_vocab_size:
        raise ValueError(f"Cannot resume: vocab size mismatch {len(vocab)} != {expected_vocab_size}")
    if config.get("token_mode") != token_mode:
        raise ValueError(f"Cannot resume: token mode mismatch {config.get('token_mode')} != {token_mode}")
    if int(config.get("context_length", context_length)) != context_length:
        raise ValueError(
            f"Cannot resume: context length mismatch {config.get('context_length')} != {context_length}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("Cannot resume: model does not contain sparse rows")
    return {str(key): [float(x) for x in row] for key, row in rows.items()}


def write_metrics(path: Path, events: Sequence[Dict[str, float]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def make_config(
    args: argparse.Namespace,
    *,
    normalized: str,
    token_ids: Sequence[int],
    examples: Sequence[Example],
    vocab: Sequence[str],
    rows: Rows,
    warm_start_loss: float | None,
    initial_loss: float,
    final_loss: float,
    total_steps: int | None = None,
    cycles_completed: int | None = None,
) -> Dict[str, object]:
    return {
        "token_mode": args.token_mode,
        "steps": args.steps,
        "total_steps": total_steps if total_steps is not None else args.steps,
        "cycles_completed": cycles_completed,
        "forever": args.forever,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "context_length": args.context_length,
        "max_vocab": args.max_vocab,
        "limit_chars": args.limit_chars,
        "seed": args.seed,
        "eval_limit": args.eval_limit,
        "metrics_out": str(args.metrics_out) if args.metrics_out else None,
        "resume_model": str(args.resume_model) if args.resume_model else None,
        "auto_resume": args.auto_resume,
        "num_characters": len(normalized),
        "num_tokens": len(token_ids),
        "num_examples": len(examples),
        "vocab_size": len(vocab),
        "trained_context_rows": len(rows),
        "backoff_suffix_rows": True,
        "warm_start_counts": args.warm_start_counts,
        "warm_start_eval_loss": warm_start_loss,
        "initial_eval_loss": initial_loss,
        "final_eval_loss": final_loss,
        "loss_delta": initial_loss - final_loss,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Delta 1 Flash tiny language model")
    parser.add_argument("--corpus", type=Path, default=default_corpus_path(), help="UTF-8 text file used for training")
    parser.add_argument("--out", type=Path, default=default_model_path(), help="model JSON path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="saved model name")
    parser.add_argument("--token-mode", choices=["byte", "char"], default="byte", help="byte supports all Unicode languages and code")
    parser.add_argument("--steps", type=int, default=1500, help="SGD update steps")
    parser.add_argument("--batch-size", type=int, default=64, help="training examples per step")
    parser.add_argument("--learning-rate", type=float, default=0.6, help="SGD learning rate")
    parser.add_argument("--context-length", type=int, default=8, help="previous tokens used as context")
    parser.add_argument("--max-vocab", type=int, default=512, help="char mode only; <=0 keeps every character")
    parser.add_argument("--limit-chars", type=int, default=0, help="optional quick-run limit for corpus characters")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--sample-every", type=int, default=500, help="log/evaluate every N steps")
    parser.add_argument("--eval-limit", type=int, default=1024, help="max eval examples for progress loss")
    parser.add_argument("--metrics-out", type=Path, help="optional JSONL file for per-step training metrics")
    parser.add_argument("--sample-length", type=int, default=700, help="tokens to generate after training")
    parser.add_argument("--start", default="Delta 1 Flash: ", help="generation prefix after training")
    parser.add_argument("--temperature", type=float, default=0.12, help="generation temperature")
    parser.add_argument("--warm-start-counts", dest="warm_start_counts", action="store_true", default=True, help="initialize rows from n-gram counts before SGD")
    parser.add_argument("--no-warm-start-counts", dest="warm_start_counts", action="store_false", help="disable count warm-start")
    parser.add_argument("--count-alpha", type=float, default=0.05, help="smoothing for count warm-start")
    parser.add_argument("--forever", action="store_true", help="train forever in cycles until Ctrl+C; saves model after cycles")
    parser.add_argument("--once", dest="forever", action="store_false", help="force one training run; useful because no-arg train.py starts forever")
    parser.add_argument("--max-cycles", type=int, default=0, help="for --forever tests; 0 means no limit")
    parser.add_argument("--save-every-cycles", type=int, default=1, help="save every N cycles in --forever mode")
    parser.add_argument("--resume-model", type=Path, help="continue training from an existing model with matching token settings")
    parser.add_argument("--auto-resume", dest="auto_resume", action="store_true", default=True, help="if --out exists, continue from it automatically")
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false", help="do not continue from an existing --out model")
    parser.add_argument("--quiet", action="store_true", help="print only final metrics, without detailed training explanation/progress")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # User-friendly default: plain `python train.py` or
    # `python -m delta1flash.train` starts continuous training.
    # Passing any argument keeps normal explicit behavior unless --forever is set.
    invoked_without_args = argv is None and len(sys.argv) == 1
    if invoked_without_args:
        args.forever = True

    if args.auto_resume and args.resume_model is None and args.out.exists():
        args.resume_model = args.out

    text = read_text(args.corpus, limit_chars=args.limit_chars)
    vocab, token_ids, normalized = prepare_tokens(text, token_mode=args.token_mode, max_vocab=args.max_vocab)
    examples = make_examples_from_ids(token_ids, args.context_length)

    split = max(1, int(len(examples) * 0.9))
    train_examples = examples[:split]
    eval_examples = examples[split:] if split < len(examples) else examples[: min(1024, len(examples))]

    rng = random.Random(args.seed)
    rng.shuffle(train_examples)

    verbose = not args.quiet
    if verbose:
        print_training_plan(
            args,
            text=text,
            vocab=vocab,
            token_ids=token_ids,
            examples=examples,
            train_examples=train_examples,
            eval_examples=eval_examples,
        )
        print("[prepare] Считаю baseline loss для модели без обучения...", flush=True)

    baseline_rows: Rows = {UNK_CONTEXT: [0.0 for _ in vocab]}
    initial_loss = loss_on_examples(baseline_rows, eval_examples, len(vocab))

    if verbose:
        print(f"[prepare] Initial eval loss: {initial_loss:.4f}", flush=True)

    if args.warm_start_counts and verbose:
        print("[prepare] Строю warm-start по частотам n-грамм...", flush=True)
    initial_rows = rows_from_counts(train_examples, len(vocab), alpha=args.count_alpha) if args.warm_start_counts else None
    warm_start_loss = loss_on_examples(initial_rows, eval_examples, len(vocab)) if initial_rows is not None else None
    if warm_start_loss is not None and verbose:
        print(f"[prepare] Warm-start eval loss: {warm_start_loss:.4f}", flush=True)
        print(f"[prepare] Стартовых строк контекста: {len(initial_rows) if initial_rows is not None else 0}", flush=True)
    if args.resume_model:
        if verbose:
            print(f"[prepare] Загружаю веса для продолжения обучения: {args.resume_model}", flush=True)
        initial_rows = load_rows_from_model(
            args.resume_model,
            expected_vocab_size=len(vocab),
            token_mode=args.token_mode,
            context_length=args.context_length,
        )
        if verbose:
            resume_loss = loss_on_examples(initial_rows, eval_examples, len(vocab))
            print(f"[prepare] Загружено строк контекста: {len(initial_rows)}", flush=True)
            print(f"[prepare] Resume eval loss: {resume_loss:.4f}", flush=True)

    if args.forever:
        rows = initial_rows if initial_rows is not None else {UNK_CONTEXT: [0.0 for _ in vocab]}
        all_logs: List[Dict[str, float]] = []
        total_steps = 0
        cycle = 0
        interrupted = False
        try:
            while True:
                cycle += 1
                if verbose:
                    print(f"[forever] Цикл {cycle} начался. Остановить можно Ctrl+C.", flush=True)
                rows, chunk_log = train_context_softmax(
                    train_examples,
                    len(vocab),
                    steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed + cycle,
                    sample_every=args.sample_every,
                    initial_rows=rows,
                    eval_examples=eval_examples,
                    eval_limit=args.eval_limit,
                    verbose=verbose,
                )
                for event in chunk_log:
                    event["cycle"] = float(cycle)
                    event["total_step"] = float(total_steps + int(event["step"]))
                all_logs.extend(chunk_log)
                total_steps += args.steps
                final_loss = loss_on_examples(rows, eval_examples, len(vocab))
                final_sample = sample_text(
                    rows,
                    vocab,
                    token_mode=args.token_mode,
                    start=args.start,
                    length=args.sample_length,
                    temperature=args.temperature,
                    seed=args.seed + cycle + 1,
                    context_length=args.context_length,
                )
                config = make_config(
                    args,
                    normalized=normalized,
                    token_ids=token_ids,
                    examples=examples,
                    vocab=vocab,
                    rows=rows,
                    warm_start_loss=warm_start_loss,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    total_steps=total_steps,
                    cycles_completed=cycle,
                )
                if args.metrics_out:
                    write_metrics(args.metrics_out, chunk_log, append=cycle > 1)
                if cycle % max(1, args.save_every_cycles) == 0:
                    save_model(
                        args.out,
                        model_name=args.model_name,
                        vocab=vocab,
                        rows=rows,
                        config=config,
                        training_log=all_logs[-200:],
                        corpus_path=args.corpus,
                        final_sample=final_sample,
                    )
                    print(
                        f"[forever] saved cycle={cycle} total_steps={total_steps} "
                        f"eval_loss={final_loss:.4f} -> {args.out}",
                        flush=True,
                    )
                if args.max_cycles and cycle >= args.max_cycles:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("\n[forever] Ctrl+C: сохраняю последнюю модель...", flush=True)

        final_loss = loss_on_examples(rows, eval_examples, len(vocab))
        final_sample = sample_text(
            rows,
            vocab,
            token_mode=args.token_mode,
            start=args.start,
            length=args.sample_length,
            temperature=args.temperature,
            seed=args.seed + cycle + 10,
            context_length=args.context_length,
        )
        config = make_config(
            args,
            normalized=normalized,
            token_ids=token_ids,
            examples=examples,
            vocab=vocab,
            rows=rows,
            warm_start_loss=warm_start_loss,
            initial_loss=initial_loss,
            final_loss=final_loss,
            total_steps=total_steps,
            cycles_completed=cycle,
        )
        config["interrupted"] = interrupted
        save_model(
            args.out,
            model_name=args.model_name,
            vocab=vocab,
            rows=rows,
            config=config,
            training_log=all_logs[-200:],
            corpus_path=args.corpus,
            final_sample=final_sample,
        )
        print(f"Model: {args.model_name}")
        print(f"FOREVER cycles completed: {cycle} | total steps: {total_steps}")
        print(f"Final eval loss: {final_loss:.4f}")
        print(f"Saved: {args.out}")
        print("--- sample ---")
        print(final_sample)
        return 0

    if verbose:
        print("[train] Начинаю SGD обучение...", flush=True)

    rows, training_log = train_context_softmax(
        train_examples,
        len(vocab),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        sample_every=args.sample_every,
        initial_rows=initial_rows,
        eval_examples=eval_examples,
        eval_limit=args.eval_limit,
        verbose=verbose,
    )
    final_loss = loss_on_examples(rows, eval_examples, len(vocab))
    final_sample = sample_text(
        rows,
        vocab,
        token_mode=args.token_mode,
        start=args.start,
        length=args.sample_length,
        temperature=args.temperature,
        seed=args.seed + 1,
        context_length=args.context_length,
    )

    config = make_config(
        args,
        normalized=normalized,
        token_ids=token_ids,
        examples=examples,
        vocab=vocab,
        rows=rows,
        warm_start_loss=warm_start_loss,
        initial_loss=initial_loss,
        final_loss=final_loss,
    )

    if args.metrics_out:
        write_metrics(args.metrics_out, training_log)
        if verbose:
            print(f"[save] Метрики обучения записаны: {args.metrics_out}", flush=True)

    if verbose:
        print("[save] Сохраняю модель и пример генерации...", flush=True)

    save_model(
        args.out,
        model_name=args.model_name,
        vocab=vocab,
        rows=rows,
        config=config,
        training_log=training_log,
        corpus_path=args.corpus,
        final_sample=final_sample,
    )

    print(f"Model: {args.model_name}")
    print(f"Corpus characters: {len(normalized)} | tokens: {len(token_ids)} | examples: {len(examples)} | vocab: {len(vocab)}")
    print(f"Token mode: {args.token_mode} | context length: {args.context_length} | trained rows: {len(rows)}")
    print(f"Initial eval loss: {initial_loss:.4f}")
    if warm_start_loss is not None:
        print(f"Warm-start loss:   {warm_start_loss:.4f}")
    print(f"Final eval loss:   {final_loss:.4f}")
    print(f"Loss delta:        {initial_loss - final_loss:.4f}")
    print(f"Saved: {args.out}")
    print("--- sample ---")
    print(final_sample)
    print("--- try ask ---")
    print(f"python -m delta1flash.ask --model {args.out} --question \"как ты будешь отвечать?\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
