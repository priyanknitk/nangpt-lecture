# pylint: disable=C0103,W0621,C3001
"""
A simple language model using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------------

torch.manual_seed(1337)

# ------------------------------------------------------------------
# 1. Load raw training text
# ------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ------------------------------------------------------------------
# 2. Build vocabulary and helper encode / decode
# ------------------------------------------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = dict(enumerate(chars))  # reverse mapping from indices to characters
encode = lambda s: [stoi[c] for c in s]  # noqa: E731
decode = lambda indices: ''.join([itos[i] for i in indices])  # noqa: E731

# ------------------------------------------------------------------
# 3. Data tensors and train/val split
# ------------------------------------------------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------------------------------------------------------------
# 5. Batch helper
# ------------------------------------------------------------------


def get_batch(split: str):
    """
    Generate a small batch of data of inputs x and targets
    y.
    """
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss on the training and validation set.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------
# Attention head
# -----------------------------------------------------------------
class Head(nn.Module):
    """A single attention head."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # calculate attention scores
        wei = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out



# ------------------------------------------------------------------
# 6. Bigram language model
# ------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model.
    """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)  # (B, T, C)
        logits = self.lm_head(x)                   # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        Generate new tokens from the model.
        """
        for _ in range(max_new_tokens):
            # crop the context if needed
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]          # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ------------------------------------------------------------------
# 7. Instantiate model
# ------------------------------------------------------------------
model = BigramLanguageModel()
m = model.to(device)
# ------------------------------------------------------------------
# 8. Optimizer and training loop
# ------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ------------------------------------------------------------------
# 9. Final text generation
# ------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
