import torch
import pickle
from collections import defaultdict
from Trainer import Transformer

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

class Jarvis:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the model and tokenizer
        try:
            # Load the full model with weights_only=False
            self.model = torch.load("model_weights/model.pt", weights_only=False)
            self.model.to(self.device)
            self.model.eval()
            
            with open("model_weights/tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
                
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
        
    def generate(self, prompt, max_length=250, temperature=0.5, top_k=50):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(1, next_token_idx)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return self.tokenizer.decode(input_ids[0])

if __name__ == "__main__":
    jarvis = Jarvis()
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == 'quit':
            break
        response = jarvis.generate(user_input)
        print(f"\n     {response}")