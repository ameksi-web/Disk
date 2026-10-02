# Delta 1 Flash

`Delta 1 Flash` — учебный проект для реального обучения маленькой нейросети-языковой модели на текстах. Старый файл проекта удалён; репозиторий теперь содержит только код и материалы для этой модели.

## Что уже сделано

- модель названа **Delta 1 Flash**;
- реализовано реальное обучение нейросети на Python без внешних ML-библиотек;
- добавлен сборщик Wikipedia API для корпуса до **5000 статей** и пресет `--languages all`;
- добавлен режим Wikipedia `--mode code` и `--mode all`;
- добавлен сборщик открытых книг / локальных текстов;
- добавлен сборщик локального кода `delta1flash.collect_code`;
- добавлен сборщик единого корпуса;
- добавлен русский туториал по коду;
- добавлен реальный терминальный чат `delta1flash.chat` с историей, памятью `/learn` и hybrid-фильтром;
- добавлена отдельная обучаемая chat intent-модель `delta1flash.train_chat`, чтобы короткие сообщения и мусор не давали повторяющиеся фразы;
- добавлен режим бесконечного обучения `train.py --forever`;
- выполнен демо-прогон обучения и сохранён файл `runs/delta1flash_demo_model.json`.

## Архитектура

Демо-модель — `token context softmax neural net` с режимом `byte` для всех Unicode-языков и кода:

1. текст превращается в токены; по умолчанию это UTF-8 байты, поэтому поддерживаются русский, английский, арабский, китайский, эмодзи и код;
2. несколько предыдущих токенов образуют контекст;
3. строка обучаемых весов отвечает за этот контекст;
4. `softmax` превращает веса в вероятности следующего токена;
5. loss — отрицательный логарифм вероятности правильного следующего токена;
6. SGD обновляет веса.

Это маленькая модель, но это **настоящее обучение**: веса случайно инициализируются, затем меняются по градиенту, а loss уменьшается.

## Быстрый старт

```bash
python3 -m delta1flash.train \
  --corpus data/demo_focused_corpus.txt \
  --out runs/delta1flash_demo_model.json \
  --model-name "Delta 1 Flash" \
  --token-mode byte \
  --steps 1500 \
  --batch-size 64 \
  --context-length 8 \
  --warm-start-counts

python3 -m delta1flash.ask \
  --model runs/delta1flash_demo_model.json \
  --question "как ты будешь отвечать?"
```


## Реальный чат

Запуск интерактивного чата:

```bash
python3 -m delta1flash.chat
```

Один вопрос без интерактива:

```bash
python3 -m delta1flash.chat --message "как обучается модель?"
```

Режимы:

- `--mode hybrid` — по умолчанию: обученная intent-модель + память; сырой byte-LM не пускается, если начинает повторять мусор;
- `--mode model` — сырой ответ byte-LM без фильтра, чтобы видеть честное поведение весов;
- `--mode rules` — только локальная память и правила.

Список слов/фраз, их значений и как говорить с ботом: `WORDS_RU.md`.

Показать список в консоли:

```bash
python3 -m delta1flash.show_words
```

Дообучить chat intent-модель:

```bash
python3 -m delta1flash.train_chat \
  --data data/chat_intents.jsonl \
  --out runs/delta1flash_chat_model.json \
  --epochs 100
```

На Windows: `train_chat.bat`.

Внутри чата можно учить память сразу:

```text
/learn как тебя зовут => Меня зовут Delta 1 Flash.
```

Эти примеры используются в чате сразу. Для дообучения весов добавь память в корпус и запусти обучение.

## Бесконечное обучение

Запуск до Ctrl+C:

```bash
python3 -m delta1flash.train \
  --corpus data/demo_focused_corpus.txt \
  --out runs/delta1flash_demo_model.json \
  --token-mode byte \
  --steps 500 \
  --forever \
  --save-every-cycles 1 \
  --metrics-out runs/delta1flash_forever_metrics.jsonl
```

Windows: можно запустить `train_forever.bat`. Для теста без бесконечного цикла используй `--max-cycles 1`.

Важно: теперь простой запуск `python train.py` или `python -m delta1flash.train` **без аргументов** сам включает continuous/forever training и автоматически продолжает обучение из `runs/delta1flash_demo_model.json`, если файл уже есть. Для одного обычного прогона используй `--once`; для старта с нуля — `--no-auto-resume`.

## Демо-метрики реального обучения

Последний запуск в sandbox сохранён в `runs/delta1flash_demo_model.json`:

- `Initial eval loss`: `5.5452`;
- `Warm-start loss`: `0.1478`;
- `Final eval loss`: `0.1478`;
- улучшение: `5.3974`;
- обученных строк контекста: `9797`.

Подробности: `TRAINING_REPORT.md`.

## Windows / прямой запуск

Если ты запускаешь из папки `delta1flash`, теперь это тоже работает:

```bat
py -3 train.py
py -3 ask.py --question "как ты будешь отвечать?"
```

Самый понятный документ: `START_RU.md`. Также есть готовые `.bat` файлы:

- `train_demo.bat` — запустить демо-обучение;
- `ask_demo.bat` — проверить ответ модели.

## Что показывает train.py во время обучения

При запуске `train.py` теперь сразу печатается понятная информация:

- что обучается: `token_context_softmax_neural_net`;
- как обучается: предсказание следующего токена, `softmax`, `cross entropy`, `SGD`;
- на каких данных: путь к корпусу, число символов, токенов и примеров;
- параметры: `steps`, `batch-size`, `context-length`, `learning-rate`;
- прогресс по шагам: текущий `batch loss`, число строк контекста, прошедшее время и ETA;
- финальные метрики: initial/warm-start/final loss и путь сохранённой модели.

Если нужен короткий вывод без подробностей:

```bash
python3 -m delta1flash.train --quiet
```

## Как собрать 5000 статей из Wikipedia

> Важно: в текущем sandbox Wikipedia по HTTPS может быть недоступна. Скрипт готов к запуску там, где сеть разрешает доступ к Wikimedia API.

```bash
python3 -m delta1flash.collect_wikipedia \
  --target 5000 \
  --mode all \
  --languages all \
  --out data/wiki_all_5000.jsonl
```

Режимы:

- `--mode languages` — статьи про естественные и языки программирования;
- `--mode neural` — статьи по теме нейросетей и машинного обучения;
- `--mode code` — статьи про программирование, алгоритмы и языки кода;
- `--mode all` — нейросети + языки + код;
- `--mode mixed` — нейросети + языки.

Пресеты языков:

- `--languages core` — 10 крупных Wikipedia-разделов;
- `--languages all` — попытка получить все активные Wikipedia-языки через Wikimedia sitematrix; если сеть к sitematrix недоступна, используется широкий fallback-список;
- `--languages en,ru,ja` — ручной список.

Если нужны именно материалы по созданию нейросетей:

```bash
python3 -m delta1flash.collect_wikipedia \
  --target 5000 \
  --mode neural \
  --languages en,ru \
  --out data/wiki_neural_5000.jsonl
```

## Как добавить код

Сканируй только свой код или код с понятной лицензией:

```bash
python3 -m delta1flash.collect_code \
  --root /path/to/your/projects \
  --out data/code_local.jsonl \
  --max-files 5000
```

## Как добавить книги

Используй только тексты, на которые у тебя есть права: public domain, собственные заметки, разрешённые датасеты.

```bash
python3 -m delta1flash.collect_books \
  --out data/books_public_domain.jsonl
```

Можно добавить локальные `.txt`:

```bash
python3 -m delta1flash.collect_books \
  --gutenberg-ids "" \
  --local my_book.txt notes.txt \
  --out data/books_local.jsonl
```

## Как обучить на 5000 статьях и книгах

```bash
python3 -m delta1flash.make_corpus \
  --inputs data/wiki_all_5000.jsonl data/code_local.jsonl data/books_public_domain.jsonl \
  --out data/corpus_5000.txt

python3 -m delta1flash.train \
  --corpus data/corpus_5000.txt \
  --out runs/delta1flash_5000_model.json \
  --model-name "Delta 1 Flash" \
  --token-mode byte \
  --steps 10000 \
  --batch-size 128 \
  --learning-rate 0.35 \
  --warm-start-counts
```

Файлы `data/wiki_*.jsonl`, `data/books_*.jsonl`, `data/code_*.jsonl`, `data/corpus_5000.txt` и большие модели по умолчанию игнорируются `.gitignore`, чтобы не забить репозиторий.

## Как проверить ответ модели

```bash
python3 -m delta1flash.ask \
  --model runs/delta1flash_demo_model.json \
  --question "как ты будешь отвечать?"
```

Ожидаемый демо-ответ текущей маленькой модели:

```text
Ок. Кратко, честно, на нужном языке, с примерами кода. Если данных мало — скажу об этом. Модель: Delta 1 Flash.
```

## Файлы проекта

- `delta1flash/train.py` — обучение нейросети; без аргументов запускает бесконечное обучение;
- `delta1flash/generate.py` — генерация текста;
- `delta1flash/collect_wikipedia.py` — сбор Wikipedia-статей;
- `delta1flash/collect_books.py` — сбор книг/public-domain/local text;
- `delta1flash/collect_code.py` — сбор локального кода;
- `delta1flash/make_corpus.py` — объединение корпуса;
- `delta1flash/ask.py` — один быстрый вопрос через чат-движок;
- `delta1flash/chat.py` — интерактивный терминальный чат;
- `delta1flash/chat_intent.py` — обучаемая intent-модель для нормального чата;
- `delta1flash/train_chat.py` — обучение chat intent-модели;
- `delta1flash/show_words.py` — печать слов/значений из intent-датасета;
- `WORDS_RU.md` — слова, значения и как говорить с моделью;
- `data/chat_intents.jsonl` — примеры intent-обучения;
- `START_RU.md` — инструкция запуска на Windows/Linux;
- `train_demo.bat`, `ask_demo.bat`, `chat_demo.bat`, `train_chat.bat`, `train_forever.bat` — быстрый запуск на Windows;
- `examples/` — примеры кода и Q/A для обучения;
- `data/seed_corpus.txt` — базовый учебный корпус;
- `data/universal_code_seed.txt` — многоязычный и кодовый seed;
- `data/demo_focused_corpus.txt` — демо-корпус для текущей обученной модели;
- `TUTORIAL_RU.md` — подробный туториал.
