from openai import OpenAI
import openai
import backoff
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from llm_tools import LLMCache

client = OpenAI()

prompt = """Please identify whether the premise entails or contradicts the hypothesis in the following premise and hypothesis. The answer should be exact “entailment”, “contradiction”, or “neutral”. Provide only the answer from the three options. Do not provide explanations.

Premise: {premise}
Hypothesis: {hypothesis}

Is it entailment, contradiction, or neutral?"""


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def call_openai_backoff(premise, hypothesis):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        seed=42 if args.seed is None else args.seed,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt.format(premise=premise, hypothesis=hypothesis),
            },
        ],
    )
    return response.choices[0].message.content


def check_cache(cache, premise, hypothesis):
    key = f"s{args.seed}||{premise}-->{hypothesis}"
    value = cache.get(key)
    if value is None:
        value = call_openai_backoff(premise, hypothesis)
        cache.set(key, value)
    return value


cache = LLMCache('/dev/shm/py/nli-gpt35.sqlite')
def call_openai(args):
    i, row = args
    golden_statements = row['golden_statement'].split('||')
    astar_to_a = []
    for ans in golden_statements:
        astar_to_a.append(check_cache(cache, ans, row['system_statement']))
    a_to_astar = []
    for ans in golden_statements:
        a_to_astar.append(check_cache(cache, row['system_statement'], ans))
    return i, a_to_astar, astar_to_a

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NQ')
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--nprocs', type=int, default=8)

    args = parser.parse_args()

    if args.dataset not in ['NQ', 'TQ']:
        raise ValueError('Invalid dataset')

    seed_suffix = '' if args.seed is None else f'-s{args.seed}'
    print("seed suffix", seed_suffix)
    df = pd.read_json(f'data/{args.dataset}-qa2s-gpt35{seed_suffix}.json')
    inputs = df.iterrows()

    with mp.Pool(args.nprocs) as pool:
        for i, a_to_astar, astar_to_a in tqdm(pool.imap_unordered(call_openai, inputs), total=df.shape[0]):
            df.at[i, 'a2astar'] = '||'.join(a_to_astar)
            df.at[i, 'astar2a'] = '||'.join(astar_to_a)

    df.to_json(f'data/{args.dataset}-nli-gpt35{seed_suffix}.json', orient='records', indent=2)
