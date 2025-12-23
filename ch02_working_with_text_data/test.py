import re
import torch
from simpleTokenizer import SimpleTokenizer
from dataLoader import create_dataloder_v1
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = create_dataloder_v1(
    raw_text, batch_size = 8, max_length = max_length, stride = max_length, shuffle = False
)


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_tokens = list(sorted(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = { s : i for i, s in enumerate(all_tokens)}
print(len(vocab))

tokenizer = SimpleTokenizer(vocab)
print(tokenizer.encode("I love you !"))