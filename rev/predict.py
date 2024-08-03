import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModelForCausalLM
from tqdm import tqdm
import math

import json
import sys

mode = sys.argv[1]
assert mode in ("rationale", "norationale")


# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
refined_model = f"llama-2-7b-{mode}"  # You can give it your own name

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, quantization_config=quant_config, device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_model = PeftModelForCausalLM.from_pretrained(base_model, refined_model)


# Load datasets
full_df = pd.read_json('../data/NQ-qa2s-gpt35.json')
test_df = pd.read_json('../trained-eval/learned-NQ-test.jsonl', orient='records', lines=True)

full_df.set_index(['qid', 'system'], inplace=True)


# next, convert train_df to llama finetune format
rationale_template = (
    "<s> [INST] Given the fact: {fact}\n"
    "answer this question: {question}\n"
    "[/INST] {golden_answer} </s>"
)
norationale_template = (
    "<s> [INST] Answer this question: {question}\n"
    "[/INST] {golden_answer} </s>"
)

def generate_text(df):
    for i, row in df.iterrows():
        item = row.to_dict()
        try:
            full_row = full_df.loc[item['qid'], item['system']]
        except KeyError:
            print(f"Missing {item['qid']} {item['system']}")
            continue
        golden_answer = item['golden_answer'].replace('||', ' or ') 
        if mode == "norationale":
            text = norationale_template.format(
                question=item['question'],
                golden_answer=golden_answer
            )
        else:
            text = rationale_template.format(
                question=item['question'],
                golden_answer=golden_answer,
                fact=full_row['system_statement']
            )
        yield i, text, golden_answer


def compute_tail_logprob(peft_model):
    for i, text, golden_answer in generate_text(test_df):
        input_ids = llama_tokenizer.encode(text, return_tensors="pt")
        outputs = peft_model(input_ids=input_ids)

        tail_ids = llama_tokenizer.encode(f'{golden_answer} </s>', return_tensors="pt")
        tail_length = tail_ids.shape[1]
        tail_logits = outputs.logits[0, -tail_length:]

        log_probs = []
        tail_logprobs = torch.nn.functional.log_softmax(tail_logits, dim=-1)
        for j, token_id in enumerate(tail_ids[0]):
            token_logprob = tail_logprobs[j, token_id]
            if token_logprob.isnan().any():
                print(f'Nan logprob at {i} for {token_id}', text, golden_answer)
            log_probs.append(token_logprob.item())
        log_prob = math.fsum(log_probs)
        assert not math.isnan(log_prob)
        yield i, log_prob


if __name__ == '__main__':
    output_df = test_df.copy()
    for i, log_prob in tqdm(compute_tail_logprob(peft_model), total=len(test_df)):
        output_df.at[i, f'{mode}_log_prob'] = log_prob
    output_df.to_json(f'learned-NQ-test-{mode}-predicted.jsonl', orient='records', lines=True)