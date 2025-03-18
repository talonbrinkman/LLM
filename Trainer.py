import os, re, datetime, torch
from torch import optim
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def loadData(folder_path):
    all_text = ""
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read() + " "
                    content = re.sub(r'[^a-zA-Z\s\?\!\.\,\:\-\'\"\<\>]', '', content)
                    content = re.sub(r'\s+', ' ', content)
                    all_text += content
    return all_text.strip().lower()

# Load the tokenizer and model for GPT-2 Medium
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

MAX_SEQ_LEN = 1024

def encode(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    return tokens[:, :MAX_SEQ_LEN]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.inputs = [encode(text) for text in texts]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        return {'input_ids': input_ids.squeeze(0), 'attention_mask': attention_mask.squeeze(0),
                'labels': input_ids.squeeze(0)}

text = loadData("data")
sentences = [line for line in text.splitlines() if line.strip()]
dataset = " ".join(sentences)

print(f"Dataset: {len(dataset):,} [{dataset[:100]}]")
print(f"Vocabulary size: {len(tokenizer):,}")

dataset = TextDataset(dataset.splitlines(), tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

total_epochs = 20
startTime = datetime.datetime.now()
for epoch in range(total_epochs):
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{total_epochs}] Loss: {loss.item():.4f}")

endTime = datetime.datetime.now()
hours, remainder = divmod((endTime - startTime).seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Training Time: {hours}:{minutes}:{seconds}")

model.save_pretrained("model_weights")
tokenizer.save_pretrained("model_weights")
print("Model and tokenizer saved in 'model_weights' folder.")