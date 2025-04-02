import os, re, datetime, torch
from torch import optim, nn
from collections import defaultdict
import pickle

class Tokenizer:
    def __init__(self, vocab_size=50000):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = vocab_size
        
    def fit(self, texts):
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_freq[word] += 1
        for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode(self, text):
        return [self.word_to_idx.get(word, 0) for word in text.lower().split()]
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ' '.join(self.idx_to_word.get(idx, '') for idx in indices)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=12, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        pos = self.pos_embedding[:, :x.size(1)]
        x = self.embedding(x) + pos
        x = self.transformer(x)
        return self.decoder(x)

def loadData(folder_path):
    all_text = ""
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                with open(os.path.join(dirpath, filename), "r", encoding="utf-8") as file:
                    content = file.read() + " "
                    content = re.sub(r'[^a-zA-Z\s\?\!\.\,\:\-\'\"\<\>]', '', content)
                    content = re.sub(r'\s+', ' ', content)
                    all_text += content
    return all_text.strip().lower()

def train():
    try:
        text = loadData("data")
        sentences = [line for line in text.splitlines() if line.strip()]
        
        dataset = " ".join(sentences)
        print(f"Dataset: {len(dataset):,} [{dataset[:100]}]")
        
        tokenizer = Tokenizer()
        tokenizer.fit(sentences)
        vocab_size = len(tokenizer.word_to_idx)
        print(f"Vocabulary size: {vocab_size:,}")
        
        model = Transformer(
            vocab_size=vocab_size,
            d_model=1024,
            nhead=16,
            num_layers=24,
            dim_feedforward=4096
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        dataset = []
        for sent in sentences:
            tokens = tokenizer.encode(sent)
            if len(tokens) > 512:
                tokens = tokens[:512]
            input_ids = torch.tensor(tokens, dtype=torch.long)
            dataset.append((input_ids, input_ids))
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        
        total_epochs = 500
        startTime = datetime.datetime.now()
        
        for epoch in range(total_epochs):
            model.train()
            total_loss = 0
            
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{total_epochs}] Loss: {avg_loss:.4f}")
        
        endTime = datetime.datetime.now()
        hours, remainder = divmod((endTime - startTime).seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining Time: {hours}:{minutes}:{seconds}")
        
        print(f"Training complete! Loss: {total_loss/len(dataloader):.4f}")
        print("Saving model...")
        os.makedirs("model_weights", exist_ok=True)
        torch.save(model, "model_weights/model.pt")
        with open("model_weights/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        print("Model saved successfully!")
                    
    except Exception as e:
        print(f"Error during training: {str(e)}", flush=True)

if __name__ == "__main__":
    train()