import os
import random

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModelForCausalLM
from trl import SFTTrainer
import pandas as pd
from tqdm import tqdm


# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
refined_model = "llama-2-7b-nli" #You can give it your own name

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_model = PeftModelForCausalLM.from_pretrained(base_model, refined_model)

df = pd.read_json('learned-NQ-test.jsonl', lines=True)
nli_df = pd.read_json('../data/NQ-nli-gpt35.json')
nli_df.set_index(['qid', 'system'], inplace=True)

template = """<s>[INST] Here is a question, a set of golden answers (split with /), an AI-generated answer.
Can you judge whether the AI-generated answer is correct according to the question and golden answers, simply answer Yes or No.

Question: {question}

Golden answers: {golden_answer}

AI answer: {system}

Can golden answers be inferred from AI answer: {a2astar}

Can AI answer be inferred from golden answers: {astar2a}
[/INST]"""


text_gen = pipeline(task="text-generation", model=peft_model, tokenizer=llama_tokenizer, max_new_tokens=20, do_sample=True)

for index, row in tqdm(df.iterrows(), total=len(df)):
    try:
        nli = nli_df.loc[int(row['qid']), row['system']]
        a2astar = nli['a2astar'].lower()
        astar2a = nli['astar2a'].lower()
    except KeyError:
        print('unk')
        a2astar = 'unknown'
        astar2a = 'unknown'


    question = row['question']
    golden_answer = row['golden_answer']
    system = row['system_answer']
    query = template.format(
        question=question,
        golden_answer=golden_answer,
        system=system,
        a2astar=a2astar,
        astar2a=astar2a,
        )
    output = text_gen(query)[0]['generated_text']
    prediction = output.split('[/INST]')[1].strip().lower()

    try:
        yes_index = prediction.index('yes')
    except ValueError:
        yes_index = 100000
    try:
        no_index = prediction.index('no')
    except ValueError:
        no_index = 100000
    
    pred = 1 if yes_index < no_index else 0
    if yes_index == no_index:
        print('nothing')
        pred = random.choice([1, 0])

    df.at[index, 'system_judge'] = pred

df.to_json('learned-NQ-test-nli-predicted.jsonl', lines=True, orient='records')