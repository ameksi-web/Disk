# Запуск Delta 1 Flash

## Чат

```bat
py -3 -m delta1flash.chat
```

Или:

```bat
chat_demo.bat
```

Один вопрос:

```bat
py -3 -m delta1flash.chat --message "привет"
```

Чат теперь использует только обученную generative-модель. Готовые ответы удалены.

## Бесконечное обучение

```bat
py -3 -m delta1flash.train
```

Без аргументов `train.py` учится бесконечно до `Ctrl+C` и продолжает существующую модель.

Один прогон:

```bat
py -3 -m delta1flash.train --once
```

С нуля:

```bat
py -3 -m delta1flash.train --once --no-auto-resume
```

## 50 000 слов

Файл:

```text
data\world_words.txt
```

Собрать слова из корпуса:

```bat
py -3 -m delta1flash.build_words --inputs data\demo_focused_corpus.txt --out data\world_words.txt --target 50000
```

Настоящий большой словарь делается из больших данных: Wikipedia, книг, кода.
