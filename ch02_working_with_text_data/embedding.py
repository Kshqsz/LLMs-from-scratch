import torch
from simpleTokenizer import SimpleTokenizer
from dataLoader import create_dataloder_v1
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = create_dataloder_v1(
    raw_text, batch_size = 8, max_length = max_length, stride = max_length, shuffle = False
)

for input_batch, output_batch in dataloader:
    break

vocab_size = 50257
output_dim = 256
torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(input_batch)
print(input_batch)
print(token_embeddings.shape)


context_length = max_length
torch.manual_seed(123)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(4))
print(pos_embeddings.shape)

print(pos_embeddings)
print(token_embeddings[0])
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings[0])