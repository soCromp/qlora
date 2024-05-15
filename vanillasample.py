import os
from qlora import *
from collections import defaultdict
import copy
import json
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse
import warnings
import numpy as np

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
)
from datasets import load_dataset, Dataset, load_from_disk
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from peft import PeftModel

from sklearn.metrics.pairwise import cosine_similarity


hfparser = transformers.HfArgumentParser((
    ModelArguments, DataArguments, TrainingArguments, GenerationArguments
))
model_args, data_args, training_args, generation_args, extra_args = \
    hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
args = argparse.Namespace(
    **vars(model_args), **vars(data_args), **vars(training_args)
)
print(args)

datapath = os.path.join(args.output_dir, 'synth.dat')

checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
if completed_training:
    print('Detected that training was already completed!')

model, tokenizer = get_accelerate_model(args, checkpoint_dir)

preds = []
batch_size = 100
num_samples = 10000
inputs = tokenizer(batch_size*[f"{tokenizer.bos_token}This person's"], return_tensors="pt", add_special_tokens=False)
inputs={c:inputs[c].to(model.device) for c in inputs}
print(inputs['input_ids'][:,0])

print('beginning generation')

for batch in tqdm(range(num_samples//batch_size)):
    out = model.generate(**inputs, generation_config=args.generation_config) # batch_size x num_cols x max_column_len
    res = tokenizer.batch_decode(out)
    # print(res)
    preds.extend(res)
        
    if len(preds)%500 == 0:
        hp = Dataset.from_pandas(pd.DataFrame(preds).T)
        hp.save_to_disk(datapath)
