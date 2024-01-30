import torch
from torch.nn.functional import log_softmax, softmax, one_hot

class StdModels:
    def __init__(self, models, tokenizers, device=None):
        self.models = models
        self.tokenizers = tokenizers
        
        self.scales = []
        for model in self.models:
            self.scales.append(sum(p.numel() for p in model.parameters()))
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
    def __iter__(self):
        return self.models
    
    def __len__(self):
        return len(self.models)
    
    def sample(self, p, prompts, k):
        tokenizer = self.tokenizers[p]
        model = self.models[p]
        model.to(self.device).eval()
        
        batch_size = len(prompts)
        max_length=64
        inputs = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True, 
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=k,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_tokens = outputs.sequences[:, max_length:]
            scores = torch.stack([log_softmax(step_score, dim=1) for step_score in outputs.scores])
            scores = scores.transpose(0, 1)
            
            generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            logits = torch.gather(scores, -1, generated_tokens.unsqueeze(-1)).squeeze()
            logits = logits.where(generated_tokens != tokenizer.pad_token_id, 0).sum(dim=-1)
                
        return (
            [generated[i*k : (i+1)*k] for i in range(batch_size)],
            [logits[i*k : (i+1)*k] for i in range(batch_size)]
        )