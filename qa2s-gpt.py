from openai import OpenAI
import openai
import backoff
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import os
import json
from httpx import HTTPStatusError

from llm_tools import LLMCache

client = OpenAI()

prompt = """Convert a question answer pair to a declarative statement, following these two examples:
Q: where is the tv show the curse of oak island filmed
A: Oak Island
S: The TV show the Curse of Oak Island is filmed on Oak Island.

Q: who wrote the first declaration of human rights
A: Cyrus
S: Cyrus wrote the first declaration of human rights

Do not provide explanations. Provide the statement only. Follow the above examples and convert this pair:
Q: {question}
A: {answer}
S:"""


@backoff.on_exception(backoff.expo, (openai.RateLimitError,), max_time=5)
def call_openai_backoff(question, answer):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        seed=args.seed,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt.format(question=question, answer=answer),
            },
        ],
    )
    return response.choices[0].message.content


def call_openai_withcache(cache, question, answer):
    key = f'S{args.seed}--{str(question).strip()}---->{str(answer).strip()}'
    statement = cache.get(key)
    if statement is None:
        statement = call_openai_backoff(question, answer)
        cache.set(key, statement)
    return statement


def call_openai(args):
    cache = LLMCache('cache/qa2s_cache.sqlite')
    i, row = args
    golden_answers = row['golden_answer'].split('||')
    golden_statements = []
    for ans in golden_answers:
        golden_statement = call_openai_withcache(cache, row['question'], ans)
        golden_statements.append(golden_statement)
    golden_statements = '||'.join(golden_statements) 

    system_statement = call_openai_withcache(cache, row['question'], row['system_answer'])

    return row, golden_statements, system_statement


def run_by_dataset(dataset):
    df = pd.read_json(f'data/{dataset}-reformatted.jsonl', lines=True)

    if args.samplesize:
        df = df.sample(args.samplesize, random_state=42)

    print(f'Computing {df.shape[0]} rows...')

    df_unprocessed = df
    df_computed = None

    print(f'Computing {df_unprocessed.shape[0]} rows...')

    inputs = df_unprocessed.iterrows()

    if df_computed is not None:
        with open(f'data/_tmp_{dataset}_qa2s.jsonl', 'w') as f:
            for _, row in df_computed.iterrows():
                f.write(json.dumps(row.to_dict()) + '\n')
    
    if args.nprocs == 1:
        with open(f'data/_tmp_{dataset}_s{args.seed}_qa2s.jsonl', 'w') as f:
            for input_args in tqdm(inputs):
                row, golden_statements, system_statement = call_openai(input_args)
                row['golden_statement'] = golden_statements
                row['system_statement'] = system_statement
                f.write(json.dumps(row.to_dict()) + '\n')
    
    else:
        with mp.Pool(args.nprocs) as pool, open(f'data/_tmp_{dataset}_s{args.seed}_qa2s.jsonl', 'w') as f:
            for row, golden_statements, system_statement in tqdm(pool.imap_unordered(call_openai, inputs), total=df_unprocessed.shape[0]):
                row['golden_statement'] = golden_statements
                row['system_statement'] = system_statement
                f.write(json.dumps(row.to_dict()) + '\n')
    
    # merge rows
    with open(f'data/_tmp_{dataset}_s{args.seed}_qa2s.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)
        if 'id' in df:
            df = df.drop(columns=['id'])
        df.to_json(f'data/{dataset}-qa2s-gpt35-s{args.seed}.json', orient='records', indent=2)


def run_nq():
    run_by_dataset('NQ')


def run_tq():
    run_by_dataset('TQ')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NQ')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nprocs', type=int, default=8)
    parser.add_argument('--samplesize', required=False, type=int)
    args = parser.parse_args()

    if args.dataset == 'NQ':
        run_nq()
    elif args.dataset == 'TQ':
        run_tq()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
