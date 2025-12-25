from previous_chapters import MultiHeadAttenion
import tiktoken
import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trk_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        # (batch, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trk_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / (torch.sqrt(var + self.eps))
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForWard(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttenion(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForWard(cfg)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx : (batch_size, num_tokens)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond) #(batch_size, num_tokens, vocab_size)
        logits = logits[:, -1, :] #(batch_size, vocab_size)
        probas = torch.softmax(logits, dim = -1) #(batch_size, vocab_size)
        idx_next = torch.argmax(probas, dim = -1, keepdim = True) #(batch_size, 1)
        idx = torch.cat((idx, idx_next), dim = -1) #(batch_size, num_tokens + 1) 
    return idx

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"
model = GPTModel(GPT_CONFIG_124M)

tokenizer = tiktoken.get_encoding("gpt2")
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim = 0)
logits = model(batch)
print(logits.shape)


# torch.manual_seed(123)
# tokenizer = tiktoken.get_encoding("gpt2")
# start_context = "Hello, I am"

# encoded = tokenizer.encode(start_context)
# print("encoded: ", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print(encoded_tensor.shape)

# out = generate_text_simple(
#     model = model,
#     idx = encoded_tensor,
#     max_new_tokens = 6,
#     context_size = GPT_CONFIG_124M["context_length"]
# )
# print(out)
# out = out.squeeze(0).tolist()
# print(tokenizer.decode(out))


# torch.manual_seed(123)


# batch = []

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# tokenizer = tiktoken.get_encoding("gpt2")
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))

# batch = torch.stack(batch, dim = 0)
# out = model(batch)
# total_params = sum(p.numel() for p in model.parameters())
# print(out.shape)
# print(total_params)












# torch.manual_seed(123)
# ffn = FeedForWard(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)
# print(ffn(x).shape)
# print(ffn.layers[0].weight)

# torch.manual_seed(123)
# batch_example = torch.randn(2, 5) 

# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch_example)
# print(out)
# ln = DummyLayerNorm(6)
# print(ln(out))










# tokenizer = tiktoken.get_encoding("gpt2")


# batch = []

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))

# batch = torch.stack(batch, dim = 0)
# # print(batch.shape)
# # print(batch)



# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# # print(logits)

