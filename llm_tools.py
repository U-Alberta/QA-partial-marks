import sqlite3
import asyncio

import os
import aiohttp


class LLMCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=120)
        self.cursor = self.conn.cursor()

        # create table if not exists
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def __del__(self):
        self.conn.close()

    def get(self, key):
        self.cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        if row is None:
            return None
        return row[0]

    def set(self, key, value):
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def delete(self, key):
        self.cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        self.conn.commit()

    def clear(self):
        self.cursor.execute("DELETE FROM cache")
        self.conn.commit()

    def list(self):
        self.cursor.execute("SELECT key FROM cache")
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]


async def make_openai_request(session, prompt, system=None, **kwargs):
    api_key = os.environ.get("OPENAI_API_KEY")
    header = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    req = {
        "model": "gpt-3.5-turbo",
        "seed": 42,
        "temperature": 0.0,
        "max_tokens": 300,
    }
    for k in req:
        if k in kwargs:
            req[k] = kwargs[k]

    if system is None:
        system = "You are a helpful assistant."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    req["messages"] = messages

    async with session.post(
        "https://api.openai.com/v1/chat/completions", json=req, headers=header
    ) as resp:
        resp.raise_for_status()
        resp = await resp.json()
        return resp["choices"][0]["message"]['content']


async def make_openai_request_batch(session, prompts, system=None, **kwargs):
    queue = asyncio.Queue()
    for prompt in prompts:
        queue.put_nowait(prompt)
    results = asyncio.Queue() 
    async def worker():
        while True:
            prompt = await queue.get()
            try:
                resp = await make_openai_request(session, prompt, system, **kwargs)
                queue.task_done()
                print(f"Done: {prompt}")
                await results.put(resp)
            except Exception as e:
                print("Exception:", e)
                queue.task_done()
    
    # run 8 workers
    workers = [worker() for _ in range(8)]
    # gather all responses
    responses = []
    # wait until all tasks are done
    await queue.join()
    # cancel all workers
    for w in workers:
        w.cancel()
    return responses


if __name__ == "__main__":
    prompts = [
        "Q: What is the capital of California?"
    for _ in range(10)]

    async def test():
        async with aiohttp.ClientSession() as session:
            resps = await make_openai_request_batch(session, prompts)
            print(resps) 

    asyncio.run(test())
