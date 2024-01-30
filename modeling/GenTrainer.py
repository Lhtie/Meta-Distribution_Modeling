import torch
import torch.nn as nn
import inspect
from copy import deepcopy

from transformers import (
    Trainer
)

class GenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("weight_mask")
        labels = deepcopy(inputs["input_ids"])
        assert "attention_mask" in inputs
        labels[inputs["attention_mask"] == 0] = -100
        
        # forward pass
        outputs = model(**inputs)
        lm_logits = outputs.get("logits")
        
        # compute ce loss, average over batches
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = weights[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        lm_loss = torch.mean(torch.stack([
            torch.sum(loss_fct(logit, label) * weight)
            for logit, label, weight in zip(shift_logits, shift_labels, shift_weights)
        ]))
        
        return (lm_loss, outputs) if return_outputs else lm_loss
    
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids", "weight_mask"] + self.label_names))