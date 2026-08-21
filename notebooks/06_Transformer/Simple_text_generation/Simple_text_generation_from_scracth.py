#!/usr/bin/env python
# coding: utf-8

# Limitations compared to SOTA LLMs:
# This is a small, from-scratch character-level Transformer, not a production-grade LLM.
# Notable gaps: single-head attention per block (no multi-head/GQA to capture diverse
# relations in parallel); character-level tokenization instead of a learned subword/BPE
# vocabulary, making sequences far longer and less semantically dense per token; a tiny
# embed_dim (128) and few layers, versus billions of parameters and dozens of layers in
# SOTA models; no rotary/ALiBi positional encoding (uses simple learned absolute position
# embeddings capped at seq_length); trained on a ~10K-sample slice of Nemotron-CC for a
# handful of epochs, versus trillions of tokens; no instruction tuning, RLHF/DPO alignment,
# or safety filtering; greedy argmax decoding only (no beam search, nucleus/top-k sampling,
# or temperature control); and no KV-caching, so generation recomputes the full forward
# pass at every step rather than scaling efficiently to long outputs.

# Character-level text generation from scratch -- single-head variant


import argparse

parser = argparse.ArgumentParser(description="Character-level text generation from scratch (multi-GPU, Nemotron-CC, single-head)")
parser.add_argument("--num_layers", type=int, default=3, help="Number of self-attention blocks")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
parser.add_argument("--epochs", type=int, default=4, help="Number of passes over the training data")
args = parser.parse_args()


# ## Hyperparameters
# Sequence length, batch size, and model dimensions used for training.

# Parameters
seq_length = args.seq_length
batch_size = args.batch_size
embed_dim = 128
hidden_dim = 4 * embed_dim  # standard transformer MLP expansion ratio (was 32: a bottleneck, not an expansion)
num_layers = args.num_layers
lr = args.lr
epochs = args.epochs

dataset_name = "nemotron_cc_singlehead"


# ## Data loading and tokenization
# Load the Nemotron-CC text sample, build a character-level vocabulary, and define the `encode`/`decode`
# helpers. The text is also split into train/validation sets for language modeling.

import torch
from datetime import datetime

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

run_date = datetime.now().strftime("%Y%m%d")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

from datasets import load_dataset

# Load the dataset (pre-downloaded from spyysalo/nemotron-cc-10K-sample and written to parquet
# under the data directory; see download_nemotron_cc.py)
dataset = load_dataset(
    "parquet",
    data_files={
        "train": "/users/staff/dmi-dmi/lucchi0001/data/nemotron_cc/train.parquet",
        "validation": "/users/staff/dmi-dmi/lucchi0001/data/nemotron_cc/validation.parquet",
        "test": "/users/staff/dmi-dmi/lucchi0001/data/nemotron_cc/test.parquet",
    },
)


# Extract the text
# Concatenate all text entries in the dataset
text = ''.join(dataset['train']['text'])


# Nemotron-CC is noisy web-crawl text with a very long tail of rare unicode characters
# (thousands of distinct glyphs each occurring only a handful of times). Left uncapped,
# vocab_size explodes to 4000+, which blows up the [batch, seq_length, vocab_size] logits
# tensor and OOMs the GPUs. Cap the vocabulary to the most frequent characters (covering
# ~99.9% of all characters) and map the rare remainder to a single <unk> token.
from collections import Counter

max_vocab_chars = 256
char_counts = Counter(text)
chars = [ch for ch, _ in char_counts.most_common(max_vocab_chars)]
unk_token = '<unk>'

vocab_size = len(chars) + 1  # + 1 for <unk>

# create a mapping from characters to integers ((max_vocab_chars) frequent chars + <unk>)
stoi = {ch: i for i, ch in enumerate(chars)}
stoi[unk_token] = len(chars)
itos = {i: ch for ch, i in stoi.items()}
unk_id = stoi[unk_token]

# encoder: take a string, output a list of integers (rare chars map to <unk>)
encode = lambda s: [stoi.get(c, unk_id) for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

num_unk = sum(v for k, v in char_counts.items() if k not in stoi or k == unk_token)
print(f"Dataset: {dataset_name}")
print(f"Vocabulary size: {vocab_size} (capped at {max_vocab_chars} most frequent chars + <unk>; "
      f"{len(char_counts)} distinct chars seen, {num_unk} chars ({100*num_unk/len(text):.4f}%) mapped to <unk>)")
print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")


# ## Batch sampler
# Samples random fixed-length windows of `seq_length` characters from the training text. The target `y` is the same window shifted by one character, i.e. next-character prediction.

def get_batch(batch_size):
    data = train_data  # could use validation, pass an argument?
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i+seq_length] for i in ix])
    y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ## Model definition
# Embedding layer followed by a single-head causal self-attention block with an MLP sublayer.
# Unlike a sequence-to-one-label setup, the output projection `Wo` is applied per position,
# producing a next-character distribution at every timestep.

import torch.nn as nn


def causal_mask(seq_len, device):
    # True where attention must be blocked (future positions)
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class AttentionBlock(nn.Module):
    """ One causal single-head self-attention layer with pre-LayerNorm and a residual
    connection, followed by a pre-LayerNorm MLP sublayer with its own residual connection. """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wproj = nn.Linear(embed_dim, embed_dim)  # output projection after attention
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    # x has shape [batch_size, sequence_length, embed_dim]
    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.ln(x)
        Q = self.Wq(x_norm)
        K = self.Wk(x_norm)
        V = self.Wv(x_norm)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)
        scores = scores.masked_fill(causal_mask(T, x.device), float('-inf'))
        attn = torch.matmul(torch.softmax(scores, dim=-1), V)
        attn = self.Wproj(attn)
        x = x + attn  # residual connection
        x = x + self.mlp(self.ln2(x))  # residual connection
        return x


class MyNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_length, num_layers=1):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Embedding(num_embeddings=seq_length, embedding_dim=embed_dim)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)  # final norm before output projection
        self.Wo = nn.Linear(embed_dim, vocab_size)  # output

    # input_x has shape [batch_size, sequence_length]
    # output has size [batch_size, seq_length, vocab_size]
    def forward(self, input_x):
        seq_len = input_x.shape[1]
        positions = torch.arange(seq_len, device=input_x.device)
        X = self.token_embedding(input_x) + self.position_embedding(positions)  # Shape: [batch_size, sequence_length, embed_dim]

        for block in self.blocks:
            X = block(X)

        X = self.ln_f(X)

        out = self.Wo(X)

        #print('out shapes', out.shape)  # [batch_size, seq_length, vocab_size]

        return out

    def get_scores(self, x):
        # Returns the (post-causal-mask) attention scores of the last layer.
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        X = self.token_embedding(x) + self.position_embedding(positions)  # Shape: [batch_size, sequence_length, embed_dim]
        mask = causal_mask(T, x.device)

        scores = None
        for block in self.blocks:
            x_norm = block.ln(X)
            Q = block.Wq(x_norm)
            K = block.Wk(x_norm)
            V = block.Wv(x_norm)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (X.shape[-1] ** 0.5)
            scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.matmul(torch.softmax(scores, dim=-1), V)
            X = X + block.Wproj(attn)
            X = X + block.mlp(block.ln2(X))

        return scores


import torch.optim as optim
import time

steps_per_epoch = max(1, len(train_data) // (batch_size * seq_length))
num_steps = epochs * steps_per_epoch
print_step = max(1, steps_per_epoch // 10)

# raw_model holds the plain, unwrapped module: used for saving weights, and for the
# post-training inference cells below (get_scores, single-sequence generation) where
# DataParallel's batch-splitting has no benefit.
raw_model = MyNN(vocab_size, embed_dim, hidden_dim, seq_length, num_layers=num_layers)
raw_model.to(device)

model = raw_model
if num_gpus > 1:
    model = nn.DataParallel(raw_model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()

print(f"Using device: {device}")
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs via nn.DataParallel")
else:
    print(f"Using {num_gpus} GPU(s); DataParallel not needed")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

train_start = time.time()
losses = []
for step in range(num_steps):
    total_loss = 0.0

    inputs_idx, labels_idx = get_batch(batch_size)
    #print('inputs', inputs_idx.shape, labels_idx.shape)

    optimizer.zero_grad()

    logits = model(inputs_idx)  # logits has size [batch_size, seq_length, vocab_size]

    logits = logits.view(-1, vocab_size)
    #print('logits', logits.shape)

    labels_idx = labels_idx.flatten()
    #print('labels', labels_idx.shape)


    loss = criterion(logits, labels_idx)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    losses.append(loss.item())

    if step % print_step == 0:
        print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}", flush=True)

train_elapsed = time.time() - train_start
print(f"Training time: {train_elapsed:.2f}s")
if torch.cuda.is_available():
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory: {peak_mem_mb:.1f} MB")
else:
    print("Peak GPU memory: N/A (no CUDA device)")


import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
loss_path = f"loss_{dataset_name}_seed{seed}_{run_date}_nl{num_layers}_lr{lr}_bs{batch_size}_sl{seq_length}_ep{epochs}.png"
plt.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.close()


weights_path = f'model_weights_{dataset_name}_seed{seed}_{run_date}_nl{num_layers}_lr{lr}_bs{batch_size}_sl{seq_length}_ep{epochs}.pt'
torch.save(raw_model.state_dict(), weights_path)
print(f'Saved model weights to {weights_path}')

# To reload later:
# model = MyNN(vocab_size, embed_dim, hidden_dim, seq_length, num_layers=num_layers)
# model.load_state_dict(torch.load(weights_path))
# model.eval()


# ## Qualitative check: next-character prediction
# Feed one sampled window through the trained model and decode the predicted next character at each position.
# Uses `raw_model` directly (no DataParallel wrapping needed for single-sequence inference).

raw_model.eval()

inputs_idx, labels_idx = get_batch(1)
print('inputs', decode(inputs_idx.squeeze(0).tolist()))
logits = raw_model(inputs_idx).squeeze(0)
pred_indices = torch.argmax(logits, dim=-1)  # [batch_size, seq_length]
predicted_text = decode(pred_indices.tolist())  # Convert IDs to characters/tokens
print("Predicted text:", predicted_text)


m = 40  # number of characters to predict

context_idx, _ = get_batch(1)
context = context_idx.squeeze(0).tolist()

generated = []
with torch.no_grad():
    for _ in range(m):
        window = context[-seq_length:]
        window_tensor = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        logits = raw_model(window_tensor)
        next_idx = torch.argmax(logits[0, -1]).item()
        generated.append(next_idx)
        context.append(next_idx)

print('Seed:', decode(context_idx.squeeze(0).tolist()))
print(f'Predicted next {m} characters:', decode(generated))


# ## Attention score visualization
# Visualize the raw (pre-softmax) attention scores for a single example sequence.

example = data[0:seq_length].to(device)
decode(example.tolist())
score = raw_model.get_scores(example.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

plt.figure(figsize=(100, 100))
plt.imshow(score, cmap='viridis', interpolation='nearest')
plt.colorbar()
attn_path = f"attn_scores_{dataset_name}_seed{seed}_{run_date}_nl{num_layers}_lr{lr}_bs{batch_size}_sl{seq_length}_ep{epochs}.png"
plt.savefig(attn_path, dpi=150, bbox_inches="tight")
plt.close()
print(f'Saved attention score plot to {attn_path}')
