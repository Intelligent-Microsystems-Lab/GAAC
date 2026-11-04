"""Calibrate the per-(layer, kv head, channel) exponent bases of a model on WikiText-2 train.

  python examples/calibrate.py --model ~/models/Qwen3-8B --prompt-len 32768 --num-prompts 4 --out qwen3_8b_bases.pt

Use prompts as long as the deployment context (RoPE shifts the K exponents with position).  --text takes any UTF-8
file instead of WikiText-2.
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanovllm import LLM, SamplingParams
from gaac import BaseCollector

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--prompt-len", type=int, default=4096); ap.add_argument("--num-prompts", type=int, default=4)
ap.add_argument("--text")
a = ap.parse_args()

if a.text:
    text = open(a.text).read()
else:
    from datasets import load_dataset
    text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
llm = LLM(os.path.expanduser(a.model), enforce_eager=True, max_model_len=a.prompt_len + 1,
          max_num_batched_tokens=max(16384, a.prompt_len + 1))
ids = llm.tokenizer(text[:20_000_000], add_special_tokens=False).input_ids
assert len(ids) >= a.num_prompts * a.prompt_len, f"need {a.num_prompts * a.prompt_len} tokens, text has {len(ids)}"

col = BaseCollector()
layers = [m for m in llm.model_runner.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
for i, m in enumerate(layers):
    m.observer = (lambda i: lambda k, v: col.observe(i, k, v))(i)
llm.generate([ids[i * a.prompt_len:(i + 1) * a.prompt_len] for i in range(a.num_prompts)], SamplingParams(max_tokens=1), use_tqdm=False)
print(f"saved bases for {col.save(a.out)} layers -> {a.out}")
