import json
from collections import defaultdict
from typing import Dict, List, Union
import torch

class TextVectorizer:
    def __init__(self, model_path: str):
        """
        Initialize vectorizer with trained model
        Args:
            model_path: Path to model.json file
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.segment_probs = defaultdict(float, model_data['segment_probs'])
        self.vocab = {seg: idx for idx, seg in enumerate(self.segment_probs.keys())}
        self.vocab_size = len(self.vocab)
        
        # Build transition probabilities if available
        self.transition_probs = {}
        if 'transition_probs' in model_data:
            self.transition_probs = {
                tuple(key.split(',')): value 
                for key, value in model_data['transition_probs'].items()
            }

    def segment_text(self, text: str) -> List[str]:
        """
        Segment text using the loaded model
        Args:
            text: Input text to segment
        Returns:
            List of segments
        """
        segments = []
        i = 0
        n = len(text)
        
        while i < n:
            found = False
            # Try to find the longest possible segment starting at i
            for j in range(min(i + 10, n), i, -1):
                segment = text[i:j]
                if segment in self.segment_probs:
                    segments.append(segment)
                    i = j
                    found = True
                    break
            
            if not found:
                # No known segment - add single character
                segments.append(text[i])
                i += 1
        
        return segments

    def text_to_vector(self, text: str, output_format: str = 'tensor') -> Union[List[int], torch.Tensor]:
        """
        Convert text to vector representation
        Args:
            text: Input text to vectorize
            output_format: 'list' or 'tensor'
        Returns:
            Vector representation (either list or torch tensor)
        """
        segments = self.segment_text(text)
        vector = [0] * self.vocab_size
        
        for seg in segments:
            if seg in self.vocab:
                vector[self.vocab[seg]] = 1
        
        if output_format == 'tensor':
            return torch.tensor(vector, dtype=torch.float32)
        return vector

    def batch_vectorize(self, texts: List[str], output_format: str = 'tensor') -> Union[List[List[int]], torch.Tensor]:
        """
        Vectorize a batch of texts
        Args:
            texts: List of texts to vectorize
            output_format: 'list' or 'tensor'
        Returns:
            Batch of vector representations
        """
        vectors = [self.text_to_vector(text, 'list') for text in texts]
        
        if output_format == 'tensor':
            return torch.stack([torch.tensor(v, dtype=torch.float32) for v in vectors])
        return vectors

    def get_vocabulary(self) -> Dict[str, int]:
        """Return the vocabulary mapping"""
        return self.vocab


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Text Vectorization Tool")
    parser.add_argument('--model', type=str, required=True, help='Path to model.json file')
    parser.add_argument('--text', type=str, help='Text to vectorize')
    parser.add_argument('--file', type=str, help='File containing texts to vectorize (one per line)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'list'], help='Output format')
    
    args = parser.parse_args()
    
    vectorizer = TextVectorizer(args.model)
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        results = [{
            "text": text,
            "segments": vectorizer.segment_text(text),
            "vector": vectorizer.text_to_vector(text, 'list'),
            "vocab_size": vectorizer.vocab_size
        } for text in texts]
    else:
        results = {
            "text": args.text,
            "segments": vectorizer.segment_text(args.text),
            "vector": vectorizer.text_to_vector(args.text, 'list'),
            "vocab_size": vectorizer.vocab_size
        }
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))
