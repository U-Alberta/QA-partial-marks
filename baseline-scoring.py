from openai import OpenAI
import openai
import backoff
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import os
import json

from llm_tools import LLMCache

client = OpenAI()

prompt = (
    "Here is a question, a set of golden answers "
    "(split with /), an AI-generated answer. "
    "Can you judge whether the AI-generated answer is correct according to the question and golden answers? Simply give a score from 1 to 5.\n"
    "1: The AI-generated answer is completely wrong.\n"
    "2: The AI-generated answer is mostly wrong.\n"
    "3: The AI-generated answer is neither wrong nor right.\n"
    "4: The AI-generated answer is mostly right.\n"
    "5: The AI-generated answer is completely right.\n"
    "\n"
    "Question: {question}\n"
    "Golden answers: {golden_answer}\n"
    "AI answer: {system_answer}\n"
)


@backoff.on_exception(backoff.expo, (openai.RateLimitError,), max_time=5)
def call_openai_backoff(question, golden_answer, system_answer):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=42,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt.format(question=question, golden_answer=golden_answer, system_answer=system_answer),
            },
        ],
    )
    return response.choices[0].message.content


def call_openai_withcache(cache, question, golden_answer, system_answer):
    key = f'{str(question).strip()}---->{str(golden_answer).strip()}---->{str(system_answer).strip()}'
    statement = cache.get(key)
    if statement is None:
        statement = call_openai_backoff(question, golden_answer, system_answer)
        cache.set(key, statement)
    return statement


def call_openai(args):
    cache = LLMCache('/dev/shm/baseline_score.sqlite')
    i, row = args
    if row['a2astar'] == row['astar2a']:
        return row, -1
    score = call_openai_withcache(cache, row['question'], row['golden_answer'], row['system_answer'])

    return row, score


def run_by_dataset(dataset):
    df = pd.read_json(f'data/{dataset}-nli-gpt35.json')
    print(f'Computing {df.shape[0]} rows...')


    print(f'Computing {df.shape[0]} rows...')

    inputs = df.iterrows()


    with mp.Pool(6) as pool, open(f'cache/_tmp_{dataset}_baselinescore.jsonl', 'w') as f:
        for row, score in tqdm(pool.imap_unordered(call_openai, inputs, chunksize=16), total=df.shape[0]):
            row['baseline_score'] = score
            f.write(json.dumps(row.to_dict()) + '\n')
    
    # merge rows
    with open(f'cache/_tmp_{dataset}_baselinescore.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)
        if 'id' in df:
            df = df.drop(columns=['id'])
        df.to_json(f'data/{dataset}-baselinescore-gpt35.json', orient='records', indent=2)


def run_nq():
    run_by_dataset('NQ')


def run_tq():
    run_by_dataset('TQ')

if __name__ == '__main__':
    run_nq()