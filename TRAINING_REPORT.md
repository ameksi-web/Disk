# Отчёт Delta 1 Flash

## Убрано

- Готовые intent-ответы.
- Правила, которые отвечали шаблонами.
- Chat intent model.

## Оставлено

- Только обученная generative-модель `runs/delta1flash_demo_model.json`.
- Реальное обучение через `train.py`: context rows, softmax, cross entropy, SGD.
- Бесконечное обучение без аргументов.
- 50 000 обучающих слов/буквосочетаний в `data/world_words.txt`.

## Последнее обучение

```text
Corpus characters: 403792
Token mode: char
Vocab size: 76
Context length: 6
Initial eval loss: 4.3307
Final eval loss: 0.1000
Loss delta: 4.2308
```

## Проверка

```bash
python3 -m delta1flash.chat --message "привет" --temperature 0.7
python3 -m delta1flash.chat --message "алфавит" --temperature 0.5
python3 -m delta1flash.train --forever --max-cycles 1 --steps 1 --limit-chars 2000
```
