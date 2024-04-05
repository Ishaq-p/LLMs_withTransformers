import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
blockSize = 128
batchSize = 32
epochs = 5000
learningRate = 3e-4
evalEpoch = 300
n_embed=16
n_head = 3
n_layer = 3
dropout = 0.2


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

data = torch.tensor(encode(content), dtype=torch.long)
n = int(len(data)*0.8)
train_Data = data[:n]
val_Data = data[n:]

def getBatch(division):
    data = train_Data if division=='train' else val_Data
    x_indexes = torch.randint(len(data)-blockSize, (batchSize, ))
    x = torch.stack([data[i: i+blockSize] for i in x_indexes])
    y = torch.stack([data[i+1: i+blockSize+1] for i in x_indexes])
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


class Head(nn.Module):
    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(n_embed, headSize, bias=False)
        self.query = nn.Linear(n_embed, headSize, bias=False)
        self.value = nn.Linear(n_embed, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize,blockSize)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([Head(headSize) for _ in range(numHeads)])
        self.proj = nn.Linear(headSize*numHeads, n_embed)
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        headSize = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, headSize)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLangModel(nn.Module):
    def __init__(self):
        super().__init__()

        # -----------------new stuff-------------
        self.tokenEmbeddingTable = nn.Embedding(vocab_size, n_embed)
        self.positionEmbedding_table = nn.Embedding(blockSize, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # -----------------------------------------
    
    def forward(self, index, target=None):
        # -----------new stuff----------------
        # print(index.shape)
        B, T = index.shape    # Batch, Token
        token_embd = self.tokenEmbeddingTable(index)   # (B, T, C)  (Batch, Token, Channel)
        pos_emd = self.positionEmbedding_table(torch.arange(T, device=device))
        x = token_embd + pos_emd
        x = self.blocks(x)
        x = self.ln_f(x)
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
            index_ = index[:, -blockSize:]
            logits, loss = self(index_) # since it returns logits and loss
            # The purpose of this is to focus only on the logits corresponding to the last token in each sequence, effectively ignoring the past sequence.
            logits = logits[:,-1,:]
            prob = F.softmax(logits, dim=-1) # seperate to groups by looking to the last dim in 2d it'll be column wise
            nextIndex = torch.multinomial(prob, num_samples=1)
            index = torch.cat((index, nextIndex), dim=1)
        return index
        

model = BigramLangModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
lossVisualData={'Epoch':[],   'Test': [],    'Train':[]}

for epoch in range(epochs):
  X, Y = getBatch('train')
  if epoch % evalEpoch ==0:
    losses = lossEstimation()
    lossVisualData['Epoch'].append(epoch);  lossVisualData['Train'].append(losses['train']);  lossVisualData['Test'].append(losses['test'])

    if epoch%evalEpoch==0:
      print(f"epoch: {epoch}\t trainLoss: {losses['train']:.3f}\t testLoss: {losses['test']:.3f}")

  logits, loss = model.forward(X, Y)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
print(loss.item())
print(decode(model.generate(index=torch.tensor([[34]]), max_new_tokens=500).tolist()[0]))

plt.Figure(figsize=(15,15))
plt.plot(lossVisualData['Epoch'], lossVisualData['Train'], color='r', label='train')
plt.plot(lossVisualData['Epoch'], lossVisualData['Test'], color='b', label='test')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()