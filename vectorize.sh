#!/bin/bash

# Usage: ./vectorize.sh model.json "text to vectorize" output_file [output_format]
# output_format can be "list" (default) or "json"

if [ $# -lt 3 ]; then
    echo "Usage: $0 model.json \"text to vectorize\" output_file [output_format]"
    exit 1
fi

MODEL_FILE=$1
TEXT=$2
OUTPUT_FILE=$3
FORMAT=${4:-"list"}

# Python script that processes the text and saves to file
python3 - <<EOF > "$OUTPUT_FILE"
import json
import sys
from collections import defaultdict

def load_model(model_path):
    with open(model_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    return defaultdict(float, model_data['segment_probs'])

def segment_text(text, segment_probs):
    segments = []
    i = 0
    n = len(text)
    
    while i < n:
        found = False
        for j in range(min(i + 10, n), i, -1):
            segment = text[i:j]
            if segment in segment_probs:
                segments.append(segment)
                i = j
                found = True
                break
        
        if not found:
            segments.append(text[i])
            i += 1
    return segments

def text_to_vector(text, segment_probs, vocab):
    segments = segment_text(text, segment_probs)
    vector = [0] * len(vocab)
    
    for seg in segments:
        if seg in vocab:
            vector[vocab[seg]] = 1
    
    return vector

model = load_model('$MODEL_FILE')
vocab = {seg: idx for idx, seg in enumerate(model.keys())}
segments = segment_text('$TEXT', model)
vector = text_to_vector('$TEXT', model, vocab)

if '$FORMAT' == 'json':
    output = {
        "text": "$TEXT",
        "segments": segments,
        "vector": vector,
        "vocab_size": len(vocab)
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
else:
    # For list format, create a more compact output
    output = {
        "segments": " ".join(segments),
        "vector": vector
    }
    json.dump(output, sys.stdout, ensure_ascii=False)
EOF

echo "Results saved to $OUTPUT_FILE"
