# Statistical approach to data segmentation for LLMs

### Training pipeline

For model training use the following script:

```bash
python segmentation.py \
  --input words.csv \
  --output_model model.json \
  --output_test test_results.json \
  --test_words "ультравысокочастотными" "новоеслово" \
  --max_iterations 50 \
  --min_segment_count 1
```

Output format:

```json
{
  "segment_probs": {
    "уль": 0.15,
    "тра": 0.12,
    ...
  },
  "transition_probs": {
    "уль,тра": 0.95,
    "тра,вы": 0.92,
    ...
  },
  "parameters": {
    "max_iterations": 50,
    "convergence_threshold": 1e-05,
    ...
  }
}
```

Output test results:

```json
[
  {
    "word": "ультравысокочастотными",
    "segmentation": [
      "уль",
      "трав",
      "ы",
      "соко",
      "част",
      "о",
      "тным",
      "и"
    ],
    "hyphenated": "уль-трав-ы-соко-част-о-тным-и"
  },
  ...
]
```
