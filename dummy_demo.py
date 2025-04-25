from vectorizer import TextVectorizer
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

model_id = "google/gemma-3-1b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

system_prompt = 'How many R\'s in STRAWBERRY?'
user_prompt = 'Return number only'

print('INITIAL PROMPTS')
print(system_prompt)
print(user_prompt)

def vectorizer(text):
    return TextVectorizer("train/model.json").segment_text(text)

vectorized_system = vectorizer(system_prompt)
vectorized_user = vectorizer(user_prompt)

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt},]
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=1024)

outputs = tokenizer.batch_decode(outputs)

print('RESULT BEFORE TOKENIZATION')
print(outputs)
print()
print('=======')
print()
print('TOKENIZED PROMPTS')
print(vectorized_system)
print(vectorized_user)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": '-'.join(vectorized_system)},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": '-'.join(vectorized_user)},]
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=1024)

outputs = tokenizer.batch_decode(outputs)

print('RESULT AFTER TOKENIZATION')
print(outputs)
