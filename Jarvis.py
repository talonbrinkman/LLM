import torch, os, re, time
from torch import optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_data_from_folder(folder_path):
    all_text = ""
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read() + " "
                    if os.path.basename(dirpath) == "qa":
                        content = re.sub(r'[^a-zA-Z\s\?\!\.\,\:\-\'\"]', '', content)
                        content = re.sub(r'\s+', ' ', content)
                    else:
                        content = re.sub(r'[^a-zA-Z\s\?\!\.\,\:\-\'\"]', '', content)
                        content = re.sub(r'\s+', ' ', content)
                    all_text += content

    all_text = all_text.strip()
    all_text = all_text.lower()
    return all_text

text = load_data_from_folder("data")
sentences = text.splitlines()
sentences = [line for line in sentences if line.strip() != ""]
dataset = " ".join(sentences)
print("Dataset (" + str(len(dataset)) + "): " + str(dataset[:100]))

model = GPT2LMHeadModel.from_pretrained("GPT2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("GPT2-medium")
model.train()

MAX_SEQ_LEN = 1024

def encode(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    if tokens.size(1) > MAX_SEQ_LEN:
        tokens = tokens[:, :MAX_SEQ_LEN]
    return tokens

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, seq_len=MAX_SEQ_LEN):
        self.inputs = [encode(text) for text in texts]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        attention_mask = torch.ones(input_ids.size(), dtype=torch.long)
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': input_ids.squeeze(0)
        }

dataset = TextDataset(dataset.splitlines(), tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.00001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_time = time.time()
totalEpochs = 50
for epoch in range(totalEpochs):
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{totalEpochs}] Loss: {loss.item():.4f}")

end_time = time.time()
training_time = end_time - start_time
hours, remainder = divmod(training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Training took {hours:.0f}:{minutes:.0f}:{seconds:.0f}")

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, words_per_line=15):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=pad_token_id,
        temperature=temperature,
        do_sample=True,
        attention_mask=attention_mask
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = re.sub(r'\s+', ' ', generated_text.replace("\r", " ").replace("\n", " ").strip())
    words = generated_text.split(' ')
    lines = [' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
    formatted_text = '\n     '.join(lines)
    return formatted_text

while True:
    user_input = input("> ")
    print("     " + str(generate_text(model, tokenizer, user_input, temperature=1.0)))