# Statistical approach to data segmentation for LLMs

This repository presents a novel tokenizer for training machine learning models for natural languages processing

Our tokenizer is an alternative approach to byte-pair-encoding aiming to mitigate hallucinations in large language models and enhance machine learning reasoning 

The tokenizer uses syllable segmentation to tokenize the data and convert the tokenized dataset to a tensor compatible with PyTorch

The tool is trained based on Bayesian approaches and uses Expectation-Maximization (EM) algorithm

**Our algorithm**

1. Collect the data from wiktionary
2. Apply rule-based syllable segmentation 
3. Train EM on the annotated data
4. Collect the dictionary (see `model.json`) for machine learning

**Model usage and compatibility**

Use `vectorize.sh` to vectorize your data with our algorithm to prevent any incompatibilities

The command line script returns JSON object with original text, segments, and vectors

Vector dimensionality equals vocabulary size (the number of unique segments from our EM-model). The vectors are built in one-hot-encoding manner and can be easily converted to any tensor format (PyTorch, TF, etc.), as well as converted from parse to dense type

*Example usage*

```bash
./vectorize.sh ~/stat-llm/train/model.json "ультравысокочастотными" sample_outputs/sample_output.json json
```

*Sample output*

```json
{
  "text": "ультравысокочастотными",
  "segments": [
    "уль",
    "трав",
    "ы",
    "соко",
    "част",
    "о",
    "тным",
    "и"
  ],
  "vector": [
    1,
    0,
    ...
  ],
  "vocab_size": [
    6413
  ]
}
```

In Python, use can use our tool as a module (see `vectorizer.py`). The module is fully compatible with PyTorch workflows, including HuggingFace integrations

### Training pipeline

For model training use the following script:

```bash
python3 train.py \
  --input ~/stat-llm/data/train_data.csv \
  --output_model model.json \
  --output_test test_results.json \
  --test_words "ультравысокочастотными" "новоеслово" \
  --max_iterations 10 \
  --min_segment_count 10
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

### Model testing

For model testing use the following script:

```bash
python3 test.py \
  --model ~/stat-llm/train/model.json \
  --test_data ~/stat-llm/data/test_data.csv \
  --output test_results.json
```

Example outputs:

```bash
Evaluating model...

Evaluation Report:
Metric         Score     
-------------------------
Precision      0.5740
Recall         0.5832
F1-Score       0.5786
Accuracy       0.1967

Confusion Matrix:
True Positives: 167320
False Positives: 124166
False Negatives: 119591

Correct Words: 15418/78388 (19.67%)
Saving detailed results to test_results.json...
Done!
```

-----

*TODO:*

- apply NKFD normalization in training data
- clean training data
