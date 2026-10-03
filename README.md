# Delta 1 Flash

`Delta 1 Flash` теперь работает как **только обучаемая generative-модель**: чат берёт ответ из обученных весов `runs/delta1flash_demo_model.json`. Готовые intent-ответы и заранее прописанные фразы для чата удалены.

## Что важно

- `train.py` без аргументов запускает бесконечное обучение до `Ctrl+C`.
- Если модель уже есть в `runs/delta1flash_demo_model.json`, обучение автоматически продолжает старые веса.
- Чат `delta1flash.chat` генерирует ответ только из модели, без правил и шаблонов.
- `data/world_words.txt` содержит 50 000 обучающих слов/буквосочетаний для покрытия русского алфавита и словаря.
- Полностью «все слова мира» невозможно хранить честно в маленьком репозитории; добавлен `delta1flash.build_words`, чтобы собирать все слова из больших корпусов: Wikipedia, книг, кода.

## Запуск чата

```bash
python3 -m delta1flash.chat
```

Один вопрос:

```bash
python3 -m delta1flash.chat --message "привет"
```

Ответы разные, потому что используется sampling:

```bash
python3 -m delta1flash.chat --message "привет" --temperature 0.5
python3 -m delta1flash.chat --message "привет" --temperature 0.9
```

## Бесконечное обучение

Простой запуск:

```bash
python3 -m delta1flash.train
```

Это включает `--forever` автоматически, если нет аргументов. Остановка: `Ctrl+C`.

Один обычный прогон:

```bash
python3 -m delta1flash.train --once
```

Один прогон с нуля:

```bash
python3 -m delta1flash.train --once --no-auto-resume
```

## Сбор 50 000 слов из корпуса

```bash
python3 -m delta1flash.build_words \
  --inputs data/demo_focused_corpus.txt data/seed_corpus.txt \
  --out data/world_words.txt \
  --target 50000
```

Для настоящего большого словаря сначала собери Wikipedia/книги:

```bash
python3 -m delta1flash.collect_wikipedia \
  --target 5000 \
  --mode all \
  --languages all \
  --out data/wiki_all_5000.jsonl

python3 -m delta1flash.build_words \
  --inputs data/wiki_all_5000.jsonl \
  --out data/world_words.txt \
  --target 50000
```

## Обучение на своём корпусе

```bash
python3 -m delta1flash.make_corpus \
  --inputs data/wiki_all_5000.jsonl data/books_public_domain.jsonl data/code_local.jsonl \
  --out data/corpus_5000.txt

python3 -m delta1flash.train \
  --corpus data/corpus_5000.txt \
  --out runs/delta1flash_demo_model.json \
  --token-mode char \
  --max-vocab 0 \
  --steps 5000 \
  --forever
```

## Файлы

- `delta1flash/train.py` — реальное обучение модели.
- `delta1flash/chat.py` — generative чат только на обученной модели.
- `delta1flash/generate.py` — генерация текста из модели.
- `delta1flash/build_words.py` — сбор слов из корпусов.
- `delta1flash/collect_wikipedia.py` — сбор Wikipedia.
- `delta1flash/collect_books.py` — сбор книг/локальных txt.
- `delta1flash/collect_code.py` — сбор локального кода.
- `data/world_words.txt` — 50 000 обучающих слов/буквосочетаний.
- `data/demo_focused_corpus.txt` — текущий обучающий корпус без готовых ответов.
- `runs/delta1flash_demo_model.json` — обученная generative-модель.
