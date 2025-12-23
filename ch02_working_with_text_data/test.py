import re
from SimpleTokenizer import SimpleTokenizer
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_tokens = list(sorted(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = { s : i for i, s in enumerate(all_tokens)}
print(len(vocab))

tokenizer = SimpleTokenizer(vocab)
print(tokenizer.encode("I love you !"))