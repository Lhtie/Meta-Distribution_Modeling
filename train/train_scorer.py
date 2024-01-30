import torch
from dataclasses import dataclass, field
from typing import Optional
import logging
import random
import json
import numpy as np
from tqdm import tqdm

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

@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_sample_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for sampling."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    sample_k: Optional[int] = field(
        default=10,
        metadata={"help": "Number of samples per standard model to evaluate expectations"},
    )
    margin: Optional[float] = field(
        default=1.0
    )
    modeling: Optional[str] = field(
        default="reg",
        metadata={"help": "Either reg or prob"},
    )
    seed: Optional[str] = field(
        default=42
    )
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model_paths = {
    "pythia-14m": "/root/autodl-tmp/cnn_dailymail_pythia-14m",
    "pythia-31m": "/root/autodl-tmp/cnn_dailymail_pythia-31m",
    "pythia-70m": "/root/autodl-tmp/cnn_dailymail_pythia-70m",
    # "pythia-160m": "/root/autodl-tmp/commongen_pythia-160m",
    "pythia-410m": "/root/autodl-tmp/cnn_dailymail_pythia-410m"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_std_models():
    models, tokenizers = [], []
    for model_path in model_paths.values():
        config = AutoConfig.from_pretrained(model_path)
        model = GPTNeoXForCausalLM.from_pretrained(model_path, config=config)
        models.append(model)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        tokenizers.append(tokenizer)
    return StdModels(models, tokenizers, device)

def get_dataset(args: DataTrainingArguments, evaluate=False):
    file_input_path = args.eval_data_input_file if evaluate else args.train_data_input_file
    file_output_path = args.eval_data_output_file if evaluate else args.train_data_output_file
    mapping = {}
    if args.line_by_line:
        with open(file_input_path, encoding="utf-8") as f:
            inputs = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        with open(file_output_path, encoding="utf-8") as f:
            outputs = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        # prompts, gold = [], []
        # with open(file_input_path, "r") as f:
        #     for pack in tqdm(json.load(f)):
        #         prompts.append(pack["article"] + "\n\nThe article can be summarized as:")
        #         gold.append(pack["abstracts"])
        for i, o in zip(inputs, outputs):
            i += " ="
            if not i in mapping:
                mapping[i] = []
            mapping[i].append(o)
        dataset = {"prompts": list(mapping.keys()), "gold": list(mapping.values())}
        # dataset = {"prompts": prompts, "gold": gold}
    else:
        raise NotImplementedError()
    return dataset

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    std_models = load_std_models()
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    scorer = Scorer.from_pretrained(model_args.model_name_or_path, config=config)
    
    tokenizer.pad_token = tokenizer.eos_token
    scorer.config.pad_token_id = tokenizer.eos_token_id
    
    train_dataset = get_dataset(data_args) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, evaluate=True) if training_args.do_eval else None
    set_seed(seed=training_args.seed)
    
    scorer.pre_sample(
        training_args,
        train_dataset,
        std_models,
        cache_dir=".cache/commongen"
        # cache_dir=".cache/cnn_dailymail"
    )
    scorer.to(device)
    scorer.train_loop(
        training_args, 
        tokenizer, 
        train_dataset,
        std_models
    )
    tokenizer.save_pretrained(training_args.output_dir)
    scorer.save_pretrained(training_args.output_dir)