import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'
blockSize = 8
batchSize = 4
epochs = 100000
learningRate = 1e-4
evalEpoch = 350
n_embed=32


with open("pg22566.txt", "r", encoding='utf-8') as f:
    content = f.read()

chars = sorted(list(set(content)))
vocab_size = len(chars)
# print(chars)

stringTo_Int = {ch:i for i,ch in enumerate(chars)}
intTo_String = {i:ch for i,ch in enumerate(chars)}
# print(stringTo_Int)
encode = lambda s: [stringTo_Int[c] for c in s]
decode = lambda l: ''.join([intTo_String[i] for i in l])

data = encode(content)
n = int(len(data)*0.8)
train_Data = data[:n]
val_Data = data[n:]

def getBatch(division):
    data = train_Data if division=='train' else val_Data
    x_indexes = random.randn(len(data)-blockSize, (batchSize, ))
    x = torch.stack([data[i: i+blockSize] for i in x_indexes])
    y = torch.stack([data[i+1: blockSize+1] for i in x_indexes])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def lossEstimation():
    out={}
    model.eval()
    for portion in ['train', 'test']:
        losses = torch.zeros(evalEpoch)
        for k in range(evalEpoch):
            X,Y = getBatch(portion)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[portion] = losses.mean()
    model.train()
    return out


class BigramLangModel(nn.Module):
    def __init__(self):
        super().__init__()

        # -----------------new stuff-------------
        self.tokenEmbeddingTable = nn.Embedding(vocab_size, n_embed)
        self.positionEmbedding_table = nn.Embedding(blockSize, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # -----------------------------------------
    
    def forward(self, index, target=None):
        # -----------new stuff----------------
        B, T = index.shape
        token_embd = self.tokenEmbeddingTable(index)   # (B, T, C)
        pos_emd = self.positionEmbedding_table(torch.arange(T, device=device))
        x = token_embd + pos_emd
        logits = self.lm_head(x)  # (B, T, vocabSize)
        # ----------------------------------------
        
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)   # converting both to the same dimmentionality so we can perform the cross_entropy loss function on them
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self.forward(index)[0] # since it returns logits and loss

            # The purpose of this is to focus only on the logits corresponding to the last token in each sequence, effectively ignoring the past sequence.
            logits = logits[:,-1,:]
            prob = F.softmax(logits, dim=-1) # seperate to groups by looking to the last dim in 2d it'll be column wise
            nextIndex = torch.multinomial(prob, num_samples=1)
            index = torch.cat((index, nextIndex), dim=1)
        return index
    

model = BigramLangModel()
print(decode(model.generate(index=torch.tensor([[34]]), max_new_tokens=5000).tolist()[0]))