import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
ids = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
texts = tokenizer.decode(ids)
context_length = 4
x = ids[: context_length]
y = ids[1 : context_length + 1]
print(f"x: {x}")
print(f"y:        {y}")