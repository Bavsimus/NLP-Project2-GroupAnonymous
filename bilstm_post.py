import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

try:
    import conllu
except ImportError:
    # !pip install conllu
    from conllu import parse_incr

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print("settings done.")

if not os.path.exists("tr_imst-ud-train.conllu"):
    print("downloading data...")
    # !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu
    # !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-test.conllu
    print("downloading done.")
else:
    print("u have data already.")

def build_vocab(filepath):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1} 
    tag_to_ix = {"<PAD>": 0}
    
    with open(filepath, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            for token in sentence:
                word = token['form'].lower()
                tag = token['upos']
                
                if tag is None: continue
                
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
                    
    return word_to_ix, tag_to_ix

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w.lower(), to_ix["<UNK>"]) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

print("Vocabulary building...")
word_to_ix, tag_to_ix = build_vocab("tr_imst-ud-train.conllu")
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

print(f"Vocab Size: {len(word_to_ix)}")
print(f"Tag Size: {len(tag_to_ix)}")

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        
        tag_scores = torch.log_softmax(tag_space, dim=1)
        return tag_scores

print("bi-lstm model declared.")

EMBEDDING_DIM = 64
HIDDEN_DIM = 64
LEARNING_RATE = 0.1
EPOCHS = 3

model = BiLSTMTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

print(f"Training starts ({EPOCHS} Epoch)...")

TRAIN_LIMIT = 1000 

with open("tr_imst-ud-train.conllu", "r", encoding="utf-8") as f:
    training_data = list(parse_incr(f))[:TRAIN_LIMIT]

for epoch in range(EPOCHS): 
    total_loss = 0
    for i, sentence in enumerate(training_data):
        tokens = [t['form'] for t in sentence if t['upos'] is not None]
        tags = [t['upos'] for t in sentence if t['upos'] is not None]
        
        if not tokens: continue
            
        sentence_in = prepare_sequence(tokens, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        model.zero_grad()
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1} Done. Loss: {total_loss/len(training_data):.4f}")

print("✅ training done!")

print("\n--- OUTPUT ---")
sample_sentence = ["Yarın", "okula", "erkenden", "gideceğim", "."]

with torch.no_grad():
    inputs = prepare_sequence(sample_sentence, word_to_ix)
    tag_scores = model(inputs)
    
    _, predicted_idxs = torch.max(tag_scores, 1)
    
    for word, idx in zip(sample_sentence, predicted_idxs):
        tag_name = ix_to_tag[idx.item()]
        print(f"{word} : {tag_name}")