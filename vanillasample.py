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

tokenizer = AutoTokenizer.from_pretrained('/zoo/llama2/llama2-7b-hf/',
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        )
data_module = make_data_module(tokenizer=tokenizer, args=args)
collator = data_module['data_collator']
print('data loaded')


print('loading base model')
modelpath = get_last_checkpoint(args.output_dir)[0]
datapath = os.path.join(modelpath, 'samples.dat')
config = LlamaConfig(**vars(args))
# model = LlamaForCausalLM.from_pretrained(modelpath, config=config)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    device_map='auto'
    # torch_dtype=torch.bfloat16,
    # device_map={"": 0},
    # load_in_4bit=True,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type='nf4',
    # )
)
model = PeftModel.from_pretrained(model, join(modelpath, 'adapter_model'), is_trainable=False).merge_and_unload()
print('peft model unloaded')
tokenizer = AutoTokenizer.from_pretrained("/zoo/llama2/llama2-7b-hf/")

preds = []
batch_size = 100
num_samples = 10000
inputs = tokenizer(batch_size*[f"{tokenizer.bos_token}This person's"], return_tensors="pt", add_special_tokens=False)
inputs={c:inputs[c].to(model.device) for c in inputs}
print(inputs['input_ids'][:,0])

print('beginning generation')

for batch in tqdm(range(num_samples//batch_size + 1)):
    out = model.generate(**inputs, generation_config=args.generation_config) # batch_size x num_cols x max_column_len
    res = tokenizer.batch_decode(out)
    # print(res)
    preds.extend(res)
        
    if len(preds)%100 == 0:
        hp = Dataset.from_pandas(pd.DataFrame(preds).T)
        hp.save_to_disk(datapath)
