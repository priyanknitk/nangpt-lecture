"""
This module demonstrates basic file handling in Python.
It reads the contents of 'input.txt' and prints the length of the dataset in
characters.
"""
import torch

BLOCK_SIZE = 8  # The length of the input sequence to be processed at once
BATCH_SIZE = 4  # The number of sequences to process in parallel

# read it in to inspect it
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = dict(enumerate(chars))
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337) # fix seed for reproducibility


def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y, depending on the split.
    split: 'train' or 'val'
    """
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(BATCH_SIZE):  # batch dimension
    for t in range(BLOCK_SIZE):  # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")
