import json
import multiprocessing as mp

from openai import OpenAI
import openai
import backoff
import pandas as pd
from tqdm import tqdm

from llm_tools import LLMCache

client = OpenAI()

prompt = (
    "We have two statements S1 (the premise) and S2 (the hypothesis). S1 entails S2.\n"
    "\n"
    "S1: {s1}\n\n"
    "S2: {s2}\n\n"
    "Now, list the reasoning process step by step to show how S2 can be deduced from S1.\n"
    "List the steps as numbered statements, starting from 1.\n"
    "If a step involves information not mentioned in S1 and S2, append [[INFO]] after the step.\n"
    "If an assumption must be made to deduce S2 from S1, append [[ASSUMPTION]] after the step.\n"
    "Provide the reasoning steps only. Do not include any other information.\n"
)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def call_openai_backoff(s1, s2):
    message = prompt.format(s1=s1, s2=s2)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=42,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant and an expert in reasoning.",
            },
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def call_cot_score(s1, s2, chain):
    message = prompt.format(s1=s1, s2=s2)
    score_prompt = (
        "Based on the reasoning steps, rate how hard it is to deduce S2 from S1.\n"
        "1: Very easy\n"
        "2: Easy\n"
        "3: Neither easy nor hard\n"
        "4: Hard\n"
        "5: Very hard\n"
        "Consider how many assumptions are needed, how much information is needed, and how much reasoning is needed.\n"
        "Return a number from 1 to 5 only. Do not include any other information.\n"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=42,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant and an expert in reasoning.",
            },
            {"role": "user", "content": message},
            {"role": "assistant", "content": chain},
            {"role": "user", "content": score_prompt},
        ],
    )
    return response.choices[0].message.content


def cached_call_score_gpt35(cache: LLMCache, s1, s2, chain):
    key = f"{s1}---->{s2}"
    result = cache.get(key)
    if result is None:
        result = call_cot_score(s1, s2, chain)
        cache.set(key, result)
    return result


def cached_call_cot_gpt35(cache: LLMCache, s1, s2):
    key = f"{s1}---->{s2}"
    result = cache.get(key)
    if result is None:
        result = call_openai_backoff(s1, s2)
        cache.set(key, result)
    return result


s1 = """One of the most well-known punk poets is Patti Smith. Smith has been a major influence in the punk and alternative rock music scenes, and her work often incorporates poetry and spoken word. Her 1975 single "Gloria" is an example of her punk poetry in action. Other punk poets include John Cooper Clarke, Jim Carroll, and Richard Hell."""
s2 = """John Cooper Clarke is known as the punk poet who used poetry in their music"""

#chains = call_openai_backoff(s1, s2)
#print(chains)
#core = call_cot_score(s1, s2, chains)
#print(score)


def cot_single_gpt35(args):
    i, row = args
    golden_statements = row["golden_statement"].split("||")

    if row["ainf"] == 1 and row["asup"] == 0:
        direction = "astar2a"
    elif row["ainf"] == 0 and row["asup"] == 1:
        direction = "a2astar"
    else:
        return i, row, []

    cache = LLMCache("cache/cot-cache-gpt35.sqlite")

    which_entails = ["entailment" in x.lower() for x in row[direction].split("||")]

    chains = []
    for gs, entails in zip(golden_statements, which_entails):
        if not entails:
            continue
        if direction == "astar2a":
            s1 = gs
            s2 = row["system_statement"]
            chains.append(cached_call_cot_gpt35(cache, s1, s2))
        else:
            s2 = gs
            s1 = row["system_statement"]
            chains.append(cached_call_cot_gpt35(cache, s1, s2))
    return i, row, chains


def score_single_gpt35(args):
    i, row = args
    golden_statements = row["golden_statement"].split("||")

    if row["ainf"] == 1 and row["asup"] == 0:
        direction = "astar2a"
    elif row["ainf"] == 0 and row["asup"] == 1:
        direction = "a2astar"
    else:
        return i, row, []

    cot_cache = LLMCache("cache/cot-cache-gpt35.sqlite")
    score_cache = LLMCache("cache/cot-score-gpt35.sqlite")

    which_entails = ["entailment" in x.lower() for x in row[direction].split("||")]

    scores = []
    for gs, entails in zip(golden_statements, which_entails):
        if not entails:
            continue
        if direction == "astar2a":
            s1 = gs
            s2 = row["system_statement"]
        else:
            s2 = gs
            s1 = row["system_statement"]
        chain = cached_call_cot_gpt35(cot_cache, s1, s2)
        score = cached_call_score_gpt35(score_cache, s1, s2, chain)
        scores.append(score)
        
    return i, row, scores


def run_cot_gpt35():
    df = pd.read_json("data/NQ-nli-gpt35.json")

    inputs = [(i, row) for i, row in df.iterrows()]
    print(len(inputs))

    with mp.Pool(8) as pool, open("data/cot-gpt35.jsonl", "w") as f:
        for i, row, chains in tqdm(
            pool.imap_unordered(cot_single_gpt35, inputs), total=df.shape[0]
        ):
            output = {}
            for k, v in row.items():
                output[k] = v
            output["chains"] = chains
            f.write(json.dumps(output) + "\n")


def run_score_gpt35():
    df = pd.read_json("data/NQ-nli-gpt35.json")

    inputs = [(i, row) for i, row in df.iterrows()]
    print(len(inputs))

    with mp.Pool(8) as pool, open("data/cot-score-gpt35.jsonl", "w") as f:
        for i, row, scores in tqdm(
            pool.imap_unordered(score_single_gpt35, inputs), total=df.shape[0]
        ):
            output = {}
            for k, v in row.items():
                output[k] = v
            output["scores"] = scores
            f.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    run_cot_gpt35()
    print("Done!")
