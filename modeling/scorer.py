from typing import Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import itertools
import os
import pickle
import math
import torch
import torch.nn as nn
from torch.nn.functional import logsigmoid, log_softmax
from transformers import (
    GPT2Model, 
    GPT2PreTrainedModel,
    logging
)

logger = logging.get_logger(__name__)

class Scorer(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, 1, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None
    ) -> Tuple:

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            attention_mask is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no attention mask is provided."
        if attention_mask is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (torch.eq(attention_mask, 1).long().argmax(-1) - 1).to(
                logits.device
            )

        logits_reg = logits[torch.arange(batch_size, device=logits.device), sequence_lengths, 0]
        
        lm_logits = log_softmax(lm_logits, dim=-1)
        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        lm_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        lm_logits = torch.cumsum(lm_logits, dim=-1)
        logits_prob = lm_logits[torch.arange(batch_size, device=logits.device), sequence_lengths - 1]

        output = (logits_reg, logits_prob) + transformer_outputs[1:]
        return output
    
    def compute(self, tokenizer, prompt, texts, modeling="reg"):
        texts = [prompt + text + tokenizer.eos_token for text in texts]
        tokenized = tokenizer(
            texts, 
            pad_to_max_length=True, 
            padding="max_length", 
            max_length=512, 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        outputs = self.forward(**tokenized)
        if modeling == "reg":
            return outputs[0]
        elif modeling == "prob":
            return outputs[1]
        else:
            raise NotImplementedError()
        
    def generate(self, tokenizer, generated, modeling="reg"):
        self.eval()
        tokenized = tokenizer(
            generated, 
            pad_to_max_length=True, 
            padding="max_length", 
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = self.forward(**tokenized)
        if modeling == "reg":
            return outputs[0]
        elif modeling == "prob":
            return outputs[1]
        else:
            raise NotImplementedError()
    
    def pre_sample(self, args, train_dataset, std_models, cache_dir=".cache"):
        print("Sampling")
        prompts = train_dataset["prompts"]
        cache_file = os.path.join(cache_dir, f"scorer_samples_{args.sample_k}_{len(std_models)}.pt")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.samples = pickle.load(f)
        else:
            self.samples = {}
            for start_idx in tqdm(range(0, len(prompts), args.per_device_sample_batch_size), desc="Batch: "):
                end_idx = min(len(prompts), start_idx + args.per_device_sample_batch_size)
                
                for prompt in prompts[start_idx:end_idx]:
                    self.samples[prompt] = []
                for p1, p2 in itertools.combinations(range(len(std_models)), 2):
                    texts1, logits1 = std_models.sample(p1, prompts[start_idx:end_idx], args.sample_k)
                    texts2, logits2 = std_models.sample(p2, prompts[start_idx:end_idx], args.sample_k)
                    for i, prompt in enumerate(prompts[start_idx:end_idx]):
                        self.samples[prompt].append((
                            texts1[i], texts2[i], logits1[i], logits2[i], p1, p2
                        ))
                for p in range(len(std_models)):
                    gold = train_dataset["gold"][start_idx:end_idx]
                    k = min(args.sample_k, max([len(x) for x in gold]))
                    texts, logits = std_models.sample(p, prompts[start_idx:end_idx], k)
                    for i, prompt in enumerate(prompts[start_idx:end_idx]):
                        self.samples[prompt].append((
                            texts[i], gold[i][:k], logits[i], None, p1, p2
                        ))
            
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.samples, f)
                
    # config: {key: val}
    # dataset: {"promtps": list[str]}
    def train_loop(self, args, tokenizer, train_dataset, std_models):
        print("Training")
        assert hasattr(self, "samples"), "Run pre_sample() for sampling first"
        
        prompts = train_dataset["prompts"]
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=args.learning_rate)
        self.train()
        
        for epoch in tqdm(range(int(args.num_train_epochs)), desc="Epoch: "):
            np.random.shuffle(prompts)
            loss = 0
            losses = []
            for batch_idx, start_idx in enumerate(tqdm(range(0, len(prompts), args.per_device_train_batch_size), desc="Batch: ")):
                end_idx = min(len(prompts), start_idx + args.per_device_train_batch_size)
                
                batch_size = sum([len(self.samples[p]) for p in prompts[start_idx:end_idx]])
                for p_idx in range(start_idx, end_idx):
                    for ptr in range(len(self.samples[prompts[p_idx]])):
                        texts1, texts2, _, _, p1, p2 = self.samples[prompts[p_idx]][ptr]
                        
                        scores1 = self.compute(tokenizer, prompts[p_idx], texts1, args.modeling)
                        scores2 = self.compute(tokenizer, prompts[p_idx], texts2, args.modeling)
                        
                        s = torch.tensor(0, dtype=torch.float).to(self.device)
                        margin = args.margin * (math.log(std_models.scales[p2]) - math.log(std_models.scales[p1]))
                        for k in range(min(len(scores1), len(scores2))):
                            s += -logsigmoid(scores2[k] - scores1[k] - margin)
                        s /= (k + 1) * batch_size * args.gradient_accumulation_steps
                        s.backward()
                        loss += s.item()
                
                if batch_idx % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    losses.append(loss)
                    loss = 0
            
            logger.info(f"Epoch: {epoch}, Loss: {np.mean(losses)}")
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"epoch-{epoch}"))
            self.save_pretrained(os.path.join(args.output_dir, f"epoch-{epoch}"))
        