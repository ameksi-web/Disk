# Примеры для Delta 1 Flash

Эти файлы можно использовать как маленький кодовый корпус.

Собрать их в JSONL:

```bash
python -m delta1flash.collect_code --root examples --out data/code_examples.jsonl
```

Собрать общий корпус с примерами:

```bash
python -m delta1flash.make_corpus \
  --inputs data/demo_universal_corpus.txt data/code_examples.jsonl \
  --out data/corpus_with_examples.txt
```

Обучить:

```bash
python -m delta1flash.train \
  --corpus data/corpus_with_examples.txt \
  --out runs/delta1flash_examples_model.json \
  --token-mode byte \
  --steps 1500 \
  --warm-start-counts
```
