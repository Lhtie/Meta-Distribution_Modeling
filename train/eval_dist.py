import torch
from dataclasses import dataclass, field
from typing import Optional
import logging
import pickle
import json
from tqdm import tqdm
from datasets import Dataset
import scipy.stats
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    GPTNeoXForCausalLM, 
    AutoTokenizer, 
    AutoConfig
)

import sys
sys.path.append("..")
from modeling import StdModels, Scorer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    eval_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The evaluation model checkpoint for weights initialization."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_input_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_output_file: Optional[str] = field(
        default=None, metadata={"help": "The output training data file (a text file)."}
    )
    eval_data_input_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    eval_data_output_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional output evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "Evaluation batch size per device"},
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    modeling: Optional[str] = field(
        default="reg",
        metadata={"help": "Either reg or prob"},
    )
    sample_k: Optional[int] = field(
        default=10,
        metadata={"help": "Number of samples per standard model to evaluate expectations"},
    )
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset(args: DataTrainingArguments, tokenizer, evaluate=False):
    file_input_path = args.eval_data_input_file if evaluate else args.train_data_input_file
    file_output_path = args.eval_data_output_file if evaluate else args.train_data_output_file
    mapping = {}
    if args.line_by_line:
        with open(file_input_path, encoding="utf-8") as f:
            inputs = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        with open(file_output_path, encoding="utf-8") as f:
            outputs = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        assert len(inputs) == len(outputs)
        for i, o in zip(inputs, outputs):
            i += " ="
            if not i in mapping:
                mapping[i] = []
            mapping[i].append(o)
        dataset = {"prompts": list(mapping.keys()), "text": list(mapping.values())}

    else:
        raise NotImplementedError()
    
    def tokenize(examples):
        inputs = [doc for doc in examples["prompts"]]
        return tokenizer(
            inputs, 
            max_length=64, 
            truncation=True, 
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )

    dataset = Dataset.from_dict(dataset)
    input_data = Dataset.from_dict({"prompts": dataset["prompts"]})
    input_data = input_data.map(tokenize, batched=True)
    input_data.set_format(columns=["input_ids", "attention_mask"], type="pytorch")
    return dataset, input_data

def load_eval_model(args):
    config = AutoConfig.from_pretrained(args.eval_model_path)
    model = GPTNeoXForCausalLM.from_pretrained(args.eval_model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_path, padding_side='left')
    return tokenizer, model

def sample(tokenizer, model, inputs, k):
    model.to(device).eval()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    max_length=64
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
        generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        generated = [g + tokenizer.eos_token for g in generated]
            
    return [generated[i*k : (i+1)*k] for i in range(len(inputs))]

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    scorer = Scorer.from_pretrained(model_args.model_name_or_path, config=config)
    
    tokenizer.pad_token = tokenizer.eos_token
    scorer.config.pad_token_id = tokenizer.eos_token_id
    scorer.to(device)
    
    eval_tokenizer, eval_model = load_eval_model(model_args)
    eval_dataset, eval_inputs = get_dataset(data_args, eval_tokenizer, evaluate=True)
    
    prompts = eval_dataset["prompts"]
    scores = []
    for start_idx in tqdm(range(0, len(prompts), data_args.per_device_eval_batch_size), desc="Batch: "):
        end_idx = min(len(prompts), start_idx + data_args.per_device_eval_batch_size)
        
        generated = sample(eval_tokenizer, eval_model, eval_inputs[start_idx:end_idx], data_args.sample_k)
        for gen in generated:
            scores.append(np.mean([s.item() for s in scorer.generate(tokenizer, gen, modeling=data_args.modeling)]))
            
    print(np.mean(scores))