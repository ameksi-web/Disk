# Туториал: как сделать свою нейросеть Delta 1 Flash

Этот проект специально сделан без тяжёлых зависимостей. Он показывает основу языковой модели на понятном коде. Для режима «все языки + код» используется `--token-mode byte`: UTF-8 байты поддерживают любой Unicode-текст и символы программирования.

## 1. Данные

Для обучения нужны тексты. В проекте есть три пути:

1. `data/demo_focused_corpus.txt` — маленький встроенный корпус для проверки.
2. `delta1flash.collect_wikipedia` — сбор статей из Wikipedia API.
3. `delta1flash.collect_books` — сбор книг из открытых источников или твоих локальных `.txt`.

Нельзя просто брать любые современные книги из интернета: у них может быть авторское право. Лучше использовать Wikipedia с лицензией, Project Gutenberg/public domain или свои тексты.

## 2. Преобразование текста

В `train.py` текст читается как UTF-8. Для режима всех языков используется `--token-mode byte`: строка превращается в байты UTF-8. Поэтому словарь всегда размера `256` и не ломается на китайском, арабском, кириллице, эмодзи и символах кода.

```python
ids = list(text.encode("utf-8", errors="replace"))
vocab = [str(i) for i in range(256)]
```

Есть и режим `--token-mode char`, где строится словарь символов, но для многоязычного корпуса лучше byte.

## 3. Обучающие пары

Модель учит переходы из контекста в следующий токен:

```text
Del -> t
elt -> a
lta -> пробел
...
```

В коде это пары `(контекст, следующий_символ)`:

```python
examples = [(tuple(ids[i-k:i]), ids[i]) for i in range(k, len(ids))]
```

## 4. Веса модели

Главный параметр модели — таблица строк `rows`: каждому контексту соответствует вектор весов размера `vocab_size`.

- ключ строки — несколько предыдущих токенов;
- столбец — кандидат следующего токена;
- значение — логит, то есть число до softmax.

Инициализация случайная:

```python
row = [rng.uniform(-0.01, 0.01) for _ in range(vocab_size)]
```

## 5. Softmax

Softmax превращает числа в вероятности:

```python
exps = [math.exp(x - max_logit) for x in logits]
probs = [x / sum(exps) for x in exps]
```

Сумма `probs` равна `1.0`.

## 6. Loss

Если правильный следующий символ имеет вероятность `p`, loss равен:

```python
loss = -math.log(p)
```

Чем выше вероятность правильного токена, тем меньше loss.

## 7. Градиентный спуск

Для softmax + cross entropy градиент простой:

```python
probs[y] -= 1.0
row[j] -= learning_rate * probs[j]
```

`y` — правильный следующий токен. Эта операция увеличивает вес правильного символа и уменьшает лишние вероятности.

## 8. Запуск демо-обучения

```bash
python3 -m delta1flash.train \
  --corpus data/demo_focused_corpus.txt \
  --out runs/delta1flash_demo_model.json \
  --model-name "Delta 1 Flash" \
  --steps 1500 \
  --token-mode byte \
  --warm-start-counts
```

После запуска смотри строки:

```text
Initial eval loss: ...
Final eval loss: ...
Loss delta: ...
```

Если `Final eval loss` ниже начального — обучение реально произошло.

## 9. Генерация текста

```bash
python3 -m delta1flash.generate \
  --model runs/delta1flash_demo_model.json \
  --start "Язык " \
  --length 700 \
  --temperature 0.3
```

`temperature` управляет случайностью:

- ниже `0.7` — скучнее и повторяемее;
- около `0.8-1.0` — нормальный режим;
- выше `1.2` — больше хаоса.

## 10. Запуск на 5000 статьях

Когда есть сеть к Wikipedia:

```bash
python3 -m delta1flash.collect_wikipedia \
  --target 5000 \
  --mode all \
  --languages all \
  --out data/wiki_all_5000.jsonl
```

Потом можно добавить код и книги:

```bash
python3 -m delta1flash.collect_code \
  --root /path/to/your/projects \
  --out data/code_local.jsonl \
  --max-files 5000

python3 -m delta1flash.collect_books \
  --out data/books_public_domain.jsonl
```

Собрать общий корпус:

```bash
python3 -m delta1flash.make_corpus \
  --inputs data/wiki_all_5000.jsonl data/code_local.jsonl data/books_public_domain.jsonl \
  --out data/corpus_5000.txt
```

Обучить:

```bash
python3 -m delta1flash.train \
  --corpus data/corpus_5000.txt \
  --out runs/delta1flash_5000_model.json \
  --model-name "Delta 1 Flash" \
  --steps 10000 \
  --batch-size 128 \
  --learning-rate 0.35 \
  --token-mode byte \
  --warm-start-counts
```

## 11. Что улучшить дальше

Эта версия учебная. Чтобы сделать модель сильнее:

- усилить context-softmax до MLP или Transformer;
- добавить BPE-токенизатор;
- использовать PyTorch;
- сделать validation/test split;
- сохранять checkpoints;
- обучать дольше на GPU;
- фильтровать корпус по качеству и лицензии.


## 12. Все языки и код

Для режима «все языки» используй Wikipedia-пресет `--languages all`. Скрипт пытается получить активные языковые разделы через Wikimedia sitematrix. Если sitematrix недоступен, используется широкий fallback-список языков.

```bash
python3 -m delta1flash.collect_wikipedia \
  --target 5000 \
  --languages all \
  --mode all \
  --out data/wiki_all_5000.jsonl
```

Для кода добавлен отдельный сборщик локальных файлов:

```bash
python3 -m delta1flash.collect_code \
  --root /path/to/your/projects \
  --out data/code_local.jsonl \
  --max-files 5000
```

Проверка ответа модели:

```bash
python3 -m delta1flash.ask \
  --model runs/delta1flash_demo_model.json \
  --question "как ты будешь отвечать?"
```

Текущий демо-ответ после обучения:

```text
Ок. Кратко, честно, на нужном языке, с примерами кода. Если данных мало — скажу об этом. Модель: Delta 1 Flash.
```


## 13. Реальный чат

Запуск интерактивного чата:

```bash
python3 -m delta1flash.chat
```

Один вопрос:

```bash
python3 -m delta1flash.chat --message "как обучается модель?"
```

Режим `hybrid` включён по умолчанию. Он пытается использовать модель, но если маленькая модель начинает повторять мусорные фразы, ответ заменяется локальным фильтром/памятью. Сырой режим для проверки весов:

```bash
python3 -m delta1flash.chat --mode model --message "привет"
```

В чате можно добавить память:

```text
/learn вопрос => правильный ответ
```

## 14. Бесконечное обучение

Запуск до Ctrl+C:

```bash
python3 -m delta1flash.train \
  --corpus data/demo_focused_corpus.txt \
  --out runs/delta1flash_demo_model.json \
  --token-mode byte \
  --steps 500 \
  --forever \
  --save-every-cycles 1
```

Для теста одного цикла:

```bash
python3 -m delta1flash.train \
  --limit-chars 3000 \
  --steps 2 \
  --forever \
  --max-cycles 1
```


## 15. Слова, значения и понимание

За нормальное понимание коротких сообщений отвечает отдельная обучаемая intent-модель. Её датасет: `data/chat_intents.jsonl`. Документ для человека: `WORDS_RU.md`.

Показать слова:

```bash
python3 -m delta1flash.show_words
```

Добавить фразу:

```json
{"text":"твоя фраза","label":"training"}
```

Переобучить понимание:

```bash
python3 -m delta1flash.train_chat --data data/chat_intents.jsonl --out runs/delta1flash_chat_model.json --epochs 100
```

## 16. train.py теперь учит бесконечно без аргументов

Простой запуск:

```bash
python3 -m delta1flash.train
```

теперь включает `--forever`, сохраняет модель после циклов и автоматически продолжает `runs/delta1flash_demo_model.json`, если файл уже есть.

Один прогон:

```bash
python3 -m delta1flash.train --once
```

Один прогон с нуля:

```bash
python3 -m delta1flash.train --once --no-auto-resume
```
