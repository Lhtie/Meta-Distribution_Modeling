from typing import Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import (
    GPT2Model, 
    GPT2PreTrainedModel,
    logging
)

logger = logging.get_logger(__name__)

class EnergyFunc(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, 1, bias=False)

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

        output = (logits,) + transformer_outputs[1:]
        return output
    
    def decoder_only_sample(self, tokenizer, model, prompt, k):
        model.to(self.device).eval()
        
        inputs = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = len(tokenizer(prompt)["input_ids"])
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=k,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_tokens = outputs.sequences
            scores = outputs.scores
            scores = [softmax(step_score, dim=1) for idx, step_score in enumerate(scores)]
            
            generated = []
            logits = []
            for i in range(generated_tokens.shape[0]):
                generated.append(tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip())    
                logit = torch.tensor(0, dtype=torch.float).to(self.device)
                for j in range(len(scores)):
                    logit += scores[j][i, generated_tokens[i, prompt_len+j]]
                logits.append(logit)
                
        return generated, logits

    # config: {key: val}
    # dataset: {"promtps": list[str]}
    def train_loop(self, args, tokenizer, train_dataset, eval_tokenizer, eval_model):
        prompts = train_dataset["prompts"]
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=args.learning_rate)
        
        self.train()
        for epoch in tqdm(range(int(args.num_train_epochs)), desc="Epoch: "):
            losses = []
            for start_idx in tqdm(range(0, len(prompts), args.per_device_train_batch_size), desc="Batch: "):
                end_idx = min(len(prompts), start_idx + args.per_device_train_batch_size)
                
                loss = torch.tensor(0, dtype=torch.float).to(self.device)
                sample_cnt = 0
                for prompt in prompts[start_idx:end_idx]:
                    texts, scores = self.decoder_only_sample(eval_tokenizer, eval_model, prompt, args.sample_k)
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
                    logits = self.forward(**tokenized)[0]
                    
                    # for i, logit in enumerate(logits):
                        
                loss /= sample_cnt
                losses.append(loss.item)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            logger.info(f"Epoch: {epoch}, Loss: {np.mean(losses)}")
        