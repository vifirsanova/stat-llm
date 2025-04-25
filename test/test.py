import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import csv

class EMSegmentationTester:
    def __init__(self, model_path: str):
        """Load trained model from JSON file"""
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.segment_probs = defaultdict(float, model_data['segment_probs'])
        self.transition_probs = {
            tuple(key.split(',')): value 
            for key, value in model_data['transition_probs'].items()
        }

    def segment_word(self, word: str) -> List[str]:
        """Segment a word using the loaded model"""
        segments = []
        i = 0
        n = len(word)
        
        while i < n:
            found = False
            # Try to find the longest possible segment starting at i
            for j in range(min(i + 10, n), i, -1):
                segment = word[i:j]
                if segment in self.segment_probs:
                    segments.append(segment)
                    i = j
                    found = True
                    break
            
            if not found:
                # No known segment - add single character
                segments.append(word[i])
                i += 1
        
        return segments

    def evaluate(self, test_data: Dict[str, List[List[str]]]) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        Returns dictionary with precision, recall, f1, and accuracy
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        correct_words = 0
        total_words = len(test_data)

        for word, gold_segmentations in test_data.items():
            pred_segmentation = self.segment_word(word)
            
            # Convert segmentations to sets of split points for comparison
            gold_splits = set()
            for seg in gold_segmentations:
                split_points = []
                pos = 0
                for s in seg[:-1]:  # All but last segment
                    pos += len(s)
                    split_points.append(pos)
                gold_splits.add(tuple(split_points))
            
            pred_split_points = []
            pos = 0
            for s in pred_segmentation[:-1]:
                pos += len(s)
                pred_split_points.append(pos)
            pred_splits = set(pred_split_points)
            
            # Check if any gold segmentation matches exactly
            if tuple(pred_split_points) in gold_splits:
                correct_words += 1
            
            # Compare against all gold segmentations
            for gold_split in gold_splits:
                gold_split_set = set(gold_split)
                true_positives += len(pred_splits & gold_split_set)
                false_positives += len(pred_splits - gold_split_set)
                false_negatives += len(gold_split_set - pred_splits)

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_words / total_words if total_words > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'correct_words': correct_words,
            'total_words': total_words
        }

    def print_evaluation_report(self, metrics: Dict[str, float]):
        """Print formatted evaluation report similar to sklearn"""
        print("\nEvaluation Report:")
        print(f"{'Metric':<15}{'Score':<10}")
        print("-" * 25)
        print(f"{'Precision':<15}{metrics['precision']:.4f}")
        print(f"{'Recall':<15}{metrics['recall']:.4f}")
        print(f"{'F1-Score':<15}{metrics['f1']:.4f}")
        print(f"{'Accuracy':<15}{metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"\nCorrect Words: {metrics['correct_words']}/{metrics['total_words']} "
              f"({metrics['accuracy']:.2%})")


def load_test_data(csv_path: str) -> Dict[str, List[List[str]]]:
    """Load test data from CSV file with words,hyphenated_words columns"""
    test_data = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['words'].strip()
            hyphenated = row['hyphenated_words'].strip()
            segments = hyphenated.split('-')
            test_data[word].append(segments)
    
    return test_data


def main():
    parser = argparse.ArgumentParser(description="Test EM Segmentation Model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model JSON file')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test CSV file with words,hyphenated_words columns')
    parser.add_argument('--output', type=str,
                       help='Path to save detailed test results (JSON format)')
    
    args = parser.parse_args()

    # Load model and test data
    print(f"Loading model from {args.model}...")
    tester = EMSegmentationTester(args.model)
    
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)

    # Evaluate model
    print("Evaluating model...")
    metrics = tester.evaluate(test_data)
    tester.print_evaluation_report(metrics)

    # Save detailed results if requested
    if args.output:
        print(f"Saving detailed results to {args.output}...")
        detailed_results = []
        
        for word, gold_segmentations in test_data.items():
            pred_segmentation = tester.segment_word(word)
            detailed_results.append({
                'word': word,
                'predicted': pred_segmentation,
                'gold': gold_segmentations,
                'correct': any(
                    pred_segmentation == gold 
                    for gold in gold_segmentations
                )
            })
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'results': detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        print("Done!")


if __name__ == "__main__":
    main()
