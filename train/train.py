from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Union, Iterable, Callable, Any
from collections import defaultdict, deque
import heapq
import math
from dataclasses import dataclass, field
from functools import total_ordering

# Types
Pair = Tuple[int, int]
Word = List[int]
AddedToken = Dict[str, Any]  # Simplified for translation

@dataclass
class Merge:
    pair: Pair
    count: int
    pos: Set[int]

    def __lt__(self, other: Merge) -> bool:
        if self.count != other.count:
            return self.count < other.count
        return self.pair > other.pair  # Reverse for ascending order

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Merge):
            return NotImplemented
        return self.count == other.count and self.pair == other.pair

@dataclass
class BpeTrainerConfig:
    min_frequency: int = 0
    vocab_size: int = 30000
    show_progress: bool = True
    special_tokens: List[AddedToken] = field(default_factory=list)
    limit_alphabet: Optional[int] = None
    initial_alphabet: Set[str] = field(default_factory=set)
    continuing_subword_prefix: Optional[str] = None
    end_of_word_suffix: Optional[str] = None
    max_token_length: Optional[int] = None

class BpeTrainerBuilder:
    def __init__(self):
        self.config = BpeTrainerConfig()

    def min_frequency(self, frequency: int) -> BpeTrainerBuilder:
        self.config.min_frequency = frequency
        return self

    def vocab_size(self, size: int) -> BpeTrainerBuilder:
        self.config.vocab_size = size
        return self

    def show_progress(self, show: bool) -> BpeTrainerBuilder:
        self.config.show_progress = show
        return self

    def special_tokens(self, tokens: List[AddedToken]) -> BpeTrainerBuilder:
        self.config.special_tokens = tokens
        return self

    def limit_alphabet(self, limit: int) -> BpeTrainerBuilder:
        self.config.limit_alphabet = limit
        return self

    def initial_alphabet(self, alphabet: Set[str]) -> BpeTrainerBuilder:
        self.config.initial_alphabet = alphabet
        return self

    def continuing_subword_prefix(self, prefix: str) -> BpeTrainerBuilder:
        self.config.continuing_subword_prefix = prefix
        return self

    def end_of_word_suffix(self, suffix: str) -> BpeTrainerBuilder:
        self.config.end_of_word_suffix = suffix
        return self

    def max_token_length(self, max_token_length: Optional[int]) -> BpeTrainerBuilder:
        self.config.max_token_length = max_token_length
        return self

    def build(self) -> BpeTrainer:
        return BpeTrainer(
            min_frequency=self.config.min_frequency,
            vocab_size=self.config.vocab_size,
            show_progress=self.config.show_progress,
            special_tokens=self.config.special_tokens,
            limit_alphabet=self.config.limit_alphabet,
            initial_alphabet=self.config.initial_alphabet,
            continuing_subword_prefix=self.config.continuing_subword_prefix,
            end_of_word_suffix=self.config.end_of_word_suffix,
            max_token_length=self.config.max_token_length,
        )

class BpeTrainer:
    def __init__(
        self,
        min_frequency: int = 0,
        vocab_size: int = 30000,
        show_progress: bool = True,
        special_tokens: Optional[List[AddedToken]] = None,
        limit_alphabet: Optional[int] = None,
        initial_alphabet: Optional[Set[str]] = None,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
        max_token_length: Optional[int] = None,
    ):
        self.min_frequency = min_frequency
        self.vocab_size = vocab_size
        self.show_progress = show_progress
        self.special_tokens = special_tokens or []
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = initial_alphabet or set()
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.max_token_length = max_token_length
        self.words: Dict[str, int] = {}

    @staticmethod
    def builder() -> BpeTrainerBuilder:
        return BpeTrainerBuilder()

    def setup_progress(self) -> Optional[Any]:
        # Placeholder for progress bar implementation
        if self.show_progress:
            print("[Progress bar would be shown]")
            return object()  # Dummy object
        return None

    def finalize_progress(self, p: Optional[Any], final_len: int) -> None:
        if p is not None:
            print(f"\nProgress completed: {final_len} items processed")

    def update_progress(self, p: Optional[Any], len: int, message: str) -> None:
        if p is not None:
            print(f"{message}: {len} items")

    def add_special_tokens(
        self, w2id: Dict[str, int], id2w: List[str]
    ) -> None:
        for token in self.special_tokens:
            content = token["content"]
            if content not in w2id:
                id2w.append(content)
                w2id[content] = len(id2w) - 1

    def compute_alphabet(
        self,
        wc: Dict[str, int],
        w2id: Dict[str, int],
        id2w: List[str],
    ) -> None:
        alphabet: Dict[str, int] = defaultdict(int)
        
        for word, count in wc.items():
            for c in word:
                alphabet[c] += count

        for c in self.initial_alphabet:
            alphabet[c] = math.inf

        kept = list(alphabet.items())

        to_remove = 0
        if self.limit_alphabet is not None and len(alphabet) > self.limit_alphabet:
            to_remove = len(alphabet) - self.limit_alphabet

        if to_remove > 0:
            kept.sort(key=lambda x: x[1])
            kept = kept[to_remove:]

        kept.sort(key=lambda x: ord(x[0]))
        for c, _ in kept:
            s = c
            if s not in w2id:
                id2w.append(s)
                w2id[s] = len(id2w) - 1

    def tokenize_words(
        self,
        wc: Dict[str, int],
        w2id: Dict[str, int],
        id2w: List[str],
        p: Optional[Any],
    ) -> Tuple[List[Word], List[int]]:
        words: List[Word] = []
        counts: List[int] = []

        for word, count in wc.items():
            current_word: Word = []
            counts.append(count)

            chars = list(word)
            for i, c in enumerate(chars):
                is_first = i == 0
                is_last = i == len(chars) - 1
                s = c
                
                if s in w2id:
                    if not is_first and self.continuing_subword_prefix:
                        s = f"{self.continuing_subword_prefix}{s}"
                    if is_last and self.end_of_word_suffix:
                        s = f"{s}{self.end_of_word_suffix}"
                    
                    if s not in w2id:
                        id2w.append(s)
                        w2id[s] = len(id2w) - 1
                    
                    current_word.append(w2id[s])
            
            words.append(current_word)
            
            if p is not None:
                pass  # Update progress

        return words, counts

    def count_pairs(
        self,
        words: List[Word],
        counts: List[int],
        p: Optional[Any],
    ) -> Tuple[Dict[Pair, int], Dict[Pair, Set[int]]]:
        pair_counts: Dict[Pair, int] = defaultdict(int)
        where_to_update: Dict[Pair, Set[int]] = defaultdict(set)

        for i, word in enumerate(words):
            for j in range(len(word) - 1):
                cur_pair: Pair = (word[j], word[j + 1])
                count = counts[i]
                
                where_to_update[cur_pair].add(i)
                pair_counts[cur_pair] += count

            if p is not None:
                pass  # Update progress

        return pair_counts, where_to_update

    def do_train(
        self,
        word_counts: Dict[str, int],
        model: "BPE",
    ) -> List[AddedToken]:
        word_to_id: Dict[str, int] = {}
        id_to_word: List[str] = []
        max_token_length = self.max_token_length or math.inf

        progress = self.setup_progress()

        # 1. Add special tokens
        self.add_special_tokens(word_to_id, id_to_word)

        # 2. Compute initial alphabet
        self.compute_alphabet(word_counts, word_to_id, id_to_word)

        # 3. Tokenize words
        self.update_progress(progress, len(word_counts), "Tokenize words")
        words, counts = self.tokenize_words(word_counts, word_to_id, id_to_word, progress)
        self.finalize_progress(progress, len(words))

        # 4. Count pairs
        self.update_progress(progress, len(words), "Count pairs")
        pair_counts, where_to_update = self.count_pairs(words, counts, progress)
        
        # Create priority queue
        queue: List[Merge] = []
        for pair, pos in where_to_update.items():
            count = pair_counts[pair]
            if count > 0:
                heapq.heappush(queue, Merge(pair, count, pos))
        self.finalize_progress(progress, len(words))

        # 5. Do merges
        self.update_progress(progress, self.vocab_size, "Compute merges")
        merges: List[Tuple[Pair, int]] = []
        
        while len(word_to_id) < self.vocab_size and queue:
            top = heapq.heappop(queue)
            
            if top.count != pair_counts[top.pair]:
                top.count = pair_counts[top.pair]
                heapq.heappush(queue, top)
                continue

            if top.count < 1 or self.min_frequency > top.count:
                break

            part_a = id_to_word[top.pair[0]]
            part_b = id_to_word[top.pair[1]]
            
            # Build new token
            if self.continuing_subword_prefix and part_b.startswith(self.continuing_subword_prefix):
                part_b = part_b[len(self.continuing_subword_prefix):]
            
            new_token = f"{part_a}{part_b}"
            
            # Insert new token if it doesn't exist
            new_token_id = word_to_id.get(new_token, len(id_to_word))
            if new_token not in word_to_id:
                id_to_word.append(new_token)
                word_to_id[new_token] = new_token_id
            
            merges.append((top.pair, new_token_id))

            # Merge the new pair in every word
            changes: List[Tuple[Pair, int]] = []
            for i in top.pos:
                word = words[i]
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == top.pair[0] and word[j + 1] == top.pair[1]:
                        new_word.append(new_token_id)
                        j += 2
                        
                        # Check for new pairs created by this merge
                        if len(new_word) > 1:
                            left_pair = (new_word[-2], new_word[-1])
                            changes.append((left_pair, 1))
                        
                        if j < len(word):
                            right_pair = (new_token_id, word[j])
                            changes.append((right_pair, 1))
                    else:
                        new_word.append(word[j])
                        j += 1
                
                words[i] = new_word

            # Update pair counts with changes
            for (pair, change) in changes:
                pair_counts[pair] += change * counts[i]
                where_to_update[pair].add(i)

            # Update queue with new/changed pairs
            for pair in where_to_update:
                count = pair_counts[pair]
                if count > 0:
                    heapq.heappush(queue, Merge(pair, count, where_to_update[pair]))

            if progress is not None:
                pass  # Update progress

        self.finalize_progress(progress, len(merges))

        # Update model with results
        model.vocab = word_to_id
        model.vocab_r = {v: k for k, v in word_to_id.items()}
        model.merges = {
            pair: (i, new_id) for i, (pair, new_id) in enumerate(merges)
        }
        model.continuing_subword_prefix = self.continuing_subword_prefix
        model.end_of_word_suffix = self.end_of_word_suffix

        return self.special_tokens.copy()

    def train(self, model: "BPE") -> List[AddedToken]:
        return self.do_train(self.words, model)

    def should_show_progress(self) -> bool:
        return self.show_progress

    def feed(
        self,
        iterator: Iterable[str],
        process: Callable[[str], List[str]],
    ) -> None:
        word_counts: Dict[str, int] = defaultdict(int)
        
        for sequence in iterator:
            for word in process(sequence):
                word_counts[word] += 1
        
        self.words = word_counts


# Simplified BPE model class for the translation
class BPE:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.vocab_r: Dict[int, str] = {}
        self.merges: Dict[Pair, Tuple[int, int]] = {}
        self.continuing_subword_prefix: Optional[str] = None
        self.end_of_word_suffix: Optional[str] = None

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab


# Test cases would be implemented similarly to the Rust version
if __name__ == "__main__":
    # Example usage
    word_counts = {
        "roses": 1,
        "are": 2,
        "red": 1,
        "voilets": 1,
        "blue": 1,
        "BERT": 1,
        "is": 2,
        "big": 1,
        "and": 1,
        "so": 1,
        "GPT-2": 1,
    }
    
    trainer = BpeTrainer(
        min_frequency=2,
        vocab_size=30000,
        show_progress=False
    )
    model = BPE()
    trainer.do_train(word_counts, model)
    
    print("Vocabulary:", model.vocab)
    print("Merges:", model.merges)
