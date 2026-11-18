"""Lesson 3: bytes per token of an LLM server that is already running.

pravaha watches the GPU while one request is served, then divides the traffic
by the number of generated tokens. In decoding, a dense model reads all of its
weights once per token, so GiB per token should be close to the weight size.

Start a server first (any OpenAI compatible server, for example vLLM):
    vllm serve Qwen/Qwen3-4B-Instruct-2507
Then:
    python examples/03_llm_bytes_per_token.py [--url http://localhost:8000] [--tokens 400]
"""
import argparse
import json
import urllib.request

import pravaha_mem as pv

GIB = 2 ** 30
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--tokens", type=int, default=400)
a = ap.parse_args()


def call(path, body=None):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


model = call("/v1/models")["data"][0]["id"]
print(f"model: {model}")

with pv.observe() as m:
    m.mark("idle_before")
    __import__("time").sleep(2)
    m.mark("request")
    r = call("/v1/chat/completions", {
        "model": model, "max_tokens": a.tokens, "ignore_eos": True,
        "messages": [{"role": "user", "content": "Write a long story about rivers."}]})
    m.mark("idle_after")
    __import__("time").sleep(2)

print(m.summary())
req = m.phases()["request"]
n = r["usage"]["completion_tokens"]
print(f"\ngenerated tokens     {n}")
print(f"kernel reads / token {req['kernel_read'] / GIB / n:.3f} GiB")
print(f"tokens per second    {n / req['seconds']:.1f}")
print(f"read bandwidth       {req['kernel_read'] / req['seconds'] / 1e9:.1f} GB/s")
print("compare GiB per token with the model's weight size on disk (dense models read all weights per token)")
