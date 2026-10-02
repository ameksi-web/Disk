# Как запускать Delta 1 Flash на Windows и Linux

Ошибка вида:

```text
ImportError: attempted relative import with no known parent package
```

появлялась, если запускать `delta1flash/train.py` напрямую. Я это исправил: теперь работают оба способа запуска.

## 1. Самый правильный запуск из корня проекта

Открой CMD или PowerShell в папке проекта, например:

```bat
cd C:\Users\ЗНЕ\Downloads\Disk-arena-01a0fd32-disk\Disk-arena-01a0fd32-disk
```

Запусти обучение. Теперь эта команда **учит бесконечно до Ctrl+C** и автоматически продолжает существующую модель:

```bat
py -3 -m delta1flash.train
```

Если `py` не найден:

```bat
python -m delta1flash.train
```

Один обычный прогон без бесконечного цикла:

```bat
py -3 -m delta1flash.train --once
```

Старт с нуля без продолжения старой модели:

```bat
py -3 -m delta1flash.train --once --no-auto-resume
```

## 1.1. Что будет видно во время обучения

`train.py` теперь пишет в консоль:

- что именно обучается;
- как работает обучение: softmax, loss, SGD;
- какой корпус читается;
- сколько символов, токенов и примеров;
- какие параметры запуска выбраны;
- прогресс по шагам: `step`, `batch loss`, `rows`, `elapsed`, `eta`;
- куда сохранена модель.

Чтобы убрать подробное описание и оставить только финал:

```bat
py -3 -m delta1flash.train --quiet
```

## 2. Если ты уже внутри папки delta1flash

Например ты находишься тут:

```bat
C:\Users\ЗНЕ\Downloads\Disk-arena-01a0fd32-disk\Disk-arena-01a0fd32-disk\delta1flash>
```

Теперь можно так:

```bat
py -3 train.py
```

или:

```bat
python train.py
```

## 3. Быстрая проверка ответа

Из корня проекта:

```bat
py -3 -m delta1flash.ask --question "как ты будешь отвечать?"
```

Из папки `delta1flash`:

```bat
py -3 ask.py --question "как ты будешь отвечать?"
```

Ожидаемый демо-ответ:

```text
Ок. Кратко, честно, на нужном языке, с примерами кода. Если данных мало — скажу об этом. Модель: Delta 1 Flash.
```

## 4. Полная команда обучения

```bat
py -3 -m delta1flash.train ^
  --corpus data\demo_focused_corpus.txt ^
  --out runs\delta1flash_demo_model.json ^
  --model-name "Delta 1 Flash" ^
  --token-mode byte ^
  --steps 1500 ^
  --batch-size 64 ^
  --context-length 8 ^
  --learning-rate 0.6 ^
  --warm-start-counts
```

`--token-mode byte` важен: он позволяет учиться на любых языках Unicode и на коде без огромного словаря символов.

## 5. Очень быстрый тест, если компьютер слабый

```bat
py -3 -m delta1flash.train ^
  --limit-chars 50000 ^
  --steps 300 ^
  --out runs\delta1flash_quick_model.json
```

## 6. Сбор своего кода для обучения

```bat
py -3 -m delta1flash.collect_code ^
  --root C:\path\to\your\projects ^
  --out data\code_local.jsonl ^
  --max-files 5000
```

Потом собрать общий корпус:

```bat
py -3 -m delta1flash.make_corpus ^
  --inputs data\wiki_all_5000.jsonl data\code_local.jsonl data\books_public_domain.jsonl ^
  --out data\corpus_5000.txt
```

И обучить:

```bat
py -3 -m delta1flash.train ^
  --corpus data\corpus_5000.txt ^
  --out runs\delta1flash_5000_model.json ^
  --token-mode byte ^
  --steps 10000 ^
  --batch-size 128 ^
  --warm-start-counts
```


## 7. Реальный чат

Интерактивный чат:

```bat
py -3 -m delta1flash.chat
```

Или просто запусти:

```bat
chat_demo.bat
```

Один вопрос:

```bat
py -3 -m delta1flash.chat --message "как обучается модель?"
```

## 7.1. Обучить чат-модель

Чтобы чат перестал отвечать заученными обрывками, добавлена отдельная обучаемая intent-модель:

```bat
py -3 -m delta1flash.train_chat --data data\chat_intents.jsonl --out runs\delta1flash_chat_model.json --epochs 100
```

Или запусти:

```bat
train_chat.bat
```

## 7.2. Слова и значения

Список фраз, intent-значений и как говорить с моделью лежит в:

```text
WORDS_RU.md
```

Показать в консоли:

```bat
py -3 -m delta1flash.show_words
```

Если модель начинает повторять фразы, обычный режим `hybrid` фильтрует плохой сырой вывод. Для проверки чистых весов можно так:

```bat
py -3 -m delta1flash.chat --mode model --message "привет"
```

## 8. Бесконечное обучение

Запустить до Ctrl+C:

```bat
py -3 -m delta1flash.train ^
  --corpus data\demo_focused_corpus.txt ^
  --out runs\delta1flash_demo_model.json ^
  --token-mode byte ^
  --steps 500 ^
  --forever ^
  --save-every-cycles 1 ^
  --metrics-out runs\delta1flash_forever_metrics.jsonl
```

Или запусти:

```bat
train_forever.bat
```

Остановка: `Ctrl+C`. После каждого цикла модель сохраняется.
