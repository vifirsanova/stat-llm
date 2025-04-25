import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import random

# Types
Segment = str
Segmentation = List[Segment]
HyphenatedData = Dict[str, List[Segmentation]]  # word -> list of possible segmentations

class EMSegmentationTrainer:
    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        min_segment_count: int = 2,
        smoothing_alpha: float = 0.1,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_segment_count = min_segment_count
        self.smoothing_alpha = smoothing_alpha  # Additive smoothing parameter
        self.segment_probs: Dict[Segment, float] = defaultdict(float)
        self.transition_probs: Dict[Tuple[Segment, Segment], float] = defaultdict(float)
        self.segment_counts: Dict[Segment, int] = defaultdict(int)
        self.pair_counts: Dict[Tuple[Segment, Segment], int] = defaultdict(int)

    def initialize_probs(self, data: HyphenatedData):
        """Initialize probabilities based on observed segmentations"""
        # Count segments and segment pairs
        for segmentations in data.values():
            for seg in segmentations:
                for i in range(len(seg)):
                    self.segment_counts[seg[i]] += 1
                    if i < len(seg) - 1:
                        self.pair_counts[(seg[i], seg[i+1])] += 1
        
        # Initialize probabilities with add-α smoothing
        total_segments = sum(self.segment_counts.values()) + self.smoothing_alpha * len(self.segment_counts)
        for seg, count in self.segment_counts.items():
            self.segment_probs[seg] = (count + self.smoothing_alpha) / total_segments
        
        # Initialize transition probabilities
        for (s1, s2), count in self.pair_counts.items():
            total_transitions = self.segment_counts[s1] + self.smoothing_alpha * len(self.segment_counts)
            self.transition_probs[(s1, s2)] = (count + self.smoothing_alpha) / total_transitions

    def e_step(self, data: HyphenatedData) -> Dict[str, List[Tuple[Segmentation, float]]]:
        """Compute expected counts for each possible segmentation"""
        expected_counts = defaultdict(list)
        
        for word, segmentations in data.items():
            total_prob = 0.0
            seg_probs = []
            
            for seg in segmentations:
                # Calculate probability of this segmentation
                prob = 1.0
                # Start with initial segment probability
                if seg:
                    prob *= self.segment_probs[seg[0]]
                
                # Multiply by transition probabilities
                for i in range(len(seg) - 1):
                    prob *= self.transition_probs[(seg[i], seg[i+1])]
                
                seg_probs.append(prob)
                total_prob += prob
            
            # Normalize probabilities
            if total_prob > 0:
                normalized_probs = [p / total_prob for p in seg_probs]
            else:
                normalized_probs = [1.0 / len(segmentations)] * len(segmentations)
            
            expected_counts[word] = list(zip(segmentations, normalized_probs))
        
        return expected_counts

    def m_step(self, expected_counts: Dict[str, List[Tuple[Segmentation, float]]]):
        """Update probabilities based on expected counts"""
        # Reset counts
        new_segment_counts = defaultdict(float)
        new_pair_counts = defaultdict(float)
        
        # Accumulate expected counts
        for seg_probs in expected_counts.values():
            for seg, prob in seg_probs:
                for i in range(len(seg)):
                    new_segment_counts[seg[i]] += prob
                    if i < len(seg) - 1:
                        new_pair_counts[(seg[i], seg[i+1])] += prob
        
        # Update segment probabilities
        total_segments = sum(new_segment_counts.values()) + self.smoothing_alpha * len(new_segment_counts)
        for seg in new_segment_counts:
            self.segment_probs[seg] = (new_segment_counts[seg] + self.smoothing_alpha) / total_segments
        
        # Update transition probabilities
        for (s1, s2) in new_pair_counts:
            total_transitions = new_segment_counts[s1] + self.smoothing_alpha * len(new_segment_counts)
            self.transition_probs[(s1, s2)] = (new_pair_counts[(s1, s2)] + self.smoothing_alpha) / total_transitions

    def train(self, data: HyphenatedData):
        """Train the model using EM algorithm"""
        self.initialize_probs(data)
        
        prev_log_likelihood = -float('inf')
        
        for iteration in range(self.max_iterations):
            # E-step
            expected_counts = self.e_step(data)
            
            # M-step
            self.m_step(expected_counts)
            
            # Compute log-likelihood to check convergence
            log_likelihood = self.compute_log_likelihood(data)
            
            # Check for convergence
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < self.convergence_threshold:
                print(f"Converged after {iteration} iterations")
                break
                
            prev_log_likelihood = log_likelihood
            
            # Prune low-probability segments
            self.prune_segments()
            
            print(f"Iteration {iteration + 1}, Log-likelihood: {log_likelihood:.2f}")
    
    def prune_segments(self):
        """Remove segments with low counts"""
        segments_to_keep = [seg for seg, count in self.segment_counts.items() 
                           if count >= self.min_segment_count]
        
        # Update segment probabilities
        total = sum(self.segment_probs[seg] for seg in segments_to_keep)
        for seg in list(self.segment_probs.keys()):
            if seg in segments_to_keep:
                self.segment_probs[seg] /= total
            else:
                del self.segment_probs[seg]
        
        # Update transition probabilities
        for (s1, s2) in list(self.transition_probs.keys()):
            if s1 not in segments_to_keep or s2 not in segments_to_keep:
                del self.transition_probs[(s1, s2)]

    def compute_log_likelihood(self, data: HyphenatedData) -> float:
        """Compute the log-likelihood of the data under current parameters"""
        total_log_prob = 0.0
        
        for word, segmentations in data.items():
            word_log_prob = 0.0
            for seg in segmentations:
                seg_prob = 1.0
                if seg:
                    seg_prob *= self.segment_probs.get(seg[0], 1e-10)
                
                for i in range(len(seg) - 1):
                    seg_prob *= self.transition_probs.get((seg[i], seg[i+1]), 1e-10)
                
                word_log_prob += seg_prob
            
            if word_log_prob > 0:
                total_log_prob += math.log(word_log_prob)
        
        return total_log_prob

    def segment_word(self, word: str) -> Segmentation:
        """Segment a new word using Viterbi algorithm"""
        # Implement Viterbi algorithm to find most likely segmentation
        # This is a simplified version - a full implementation would be more complex
        
        # For now, just return the segments that exist in our vocabulary
        segments = []
        i = 0
        n = len(word)
        
        while i < n:
            found = False
            # Try to find the longest possible segment starting at i
            for j in range(min(i + 10, n), i, -1):  # Look for segments up to 10 chars
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


def load_data_from_pairs(word_pairs: List[Tuple[str, str]]) -> HyphenatedData:
    """Convert word-hyphenation pairs into training data format"""
    data = defaultdict(list)
    
    for word, hyphenated in word_pairs:
        segments = hyphenated.split('-')
        data[word].append(segments)
    
    return data


# Test data
test_data = [
    ("ультравысокочастотными", "уль-тра-вы-соко-ча-сто-тны-ми"),
    ("ультравысокочастотному", "уль-тра-вы-соко-ча-сто-тно-му"),
    ("ультравысокочастотного", "уль-тра-вы-соко-ча-сто-тно-го"),
    ("товарораспорядительными", "то-варо-ра-спо-ря-ди-тель-ны-ми"),
    ("товарораспорядительному", "то-варо-ра-спо-ря-ди-тель-но-му"),
    ("товарораспорядительного", "то-варо-ра-спо-ря-ди-тель-но-го"),
    ("самосовершенствовавшийся", "са-мо-со-вер-шен-ство-вав-ший-ся"),
    ("самопрограммировавшийся", "само-прог-рам-ми-ро-вав-ший-ся"),
]


if __name__ == "__main__":
    # Prepare data
    data = load_data_from_pairs(test_data)
    
    # Train model
    trainer = EMSegmentationTrainer(
        max_iterations=20,
        convergence_threshold=1e-4,
        min_segment_count=1,
        smoothing_alpha=0.1
    )
    trainer.train(data)
    
    # Show learned segments
    print("\nLearned segments and probabilities:")
    for seg, prob in sorted(trainer.segment_probs.items(), key=lambda x: -x[1]):
        if len(seg) > 1:  # Only show multi-character segments
            print(f"{seg}: {prob:.4f}")
    
    # Test segmentation on new words
    test_words = [
        "ультравысокочастотными",
        "товарораспорядительными",
        "самопрограммировавшийся",
        "новоеслово"
    ]
    
    print("\nTest segmentations:")
    for word in test_words:
        segmentation = trainer.segment_word(word)
        print(f"{word}: {'-'.join(segmentation)}")
