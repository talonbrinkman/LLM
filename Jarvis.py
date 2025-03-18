import torch, re, time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained("model_weights").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("model_weights")

tokenizer.add_tokens(["<eos>"])

def generate_text(prompt, max_length=100, temperature=1.0, words_per_line=15):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        do_sample=True,
        eos_token_id=tokenizer.encode("<eos>")[0]
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_text = re.sub(r'\s+', ' ', generated_text.strip())
    words = generated_text.split(' ')
    return '\n     '.join([' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)])

while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break
    print("     " + generate_text(user_input, temperature=1.0))