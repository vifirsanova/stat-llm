from vectorizer import TextVectorizer
import torch

vectorizer = TextVectorizer("train/model.json")
batch = ["синхрофазотрон", "гипотенуза", "алфавит"]
tensor_batch = vectorizer.batch_vectorize(batch, 'tensor')

# Use in a PyTorch model
class SegmentClassifier(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.fc = torch.nn.Linear(vocab_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = SegmentClassifier(vectorizer.vocab_size)
output = model(tensor_batch)

print(output)
