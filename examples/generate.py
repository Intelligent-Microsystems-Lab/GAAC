"""End-to-end generation with nano-vllm: stock paged BF16 KV vs the GAAC compressed KV cache.

  python examples/generate.py --model ~/models/Qwen3-8B                                  # GAAC, per-sequence bases
  python examples/generate.py --model ~/models/Qwen3-8B --bases qwen3_8b_bases.pt        # GAAC, calibrated bases
  python examples/generate.py --model ~/models/Qwen3-8B --kv none                        # stock nano-vllm
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanovllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--kv", default="gaac", choices=["gaac", "none"])
ap.add_argument("--bases"); ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--eager", action="store_true")
a = ap.parse_args()

llm = LLM(os.path.expanduser(a.model), enforce_eager=a.eager, max_model_len=a.max_model_len,
          kv_compression=a.kv, gaac_bases=a.bases)
prompts = ["Introduce yourself.", "List all prime numbers within 100.", "Explain how a KV cache speeds up LLM decoding."]
prompts = [llm.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
t0 = time.perf_counter()
outs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=a.max_tokens))
print(f"{time.perf_counter() - t0:.1f} s")
for p, o in zip(prompts, outs):
    print("\n" + "-" * 80 + f"\n{p!r}\n-> {o['text']!r}")
if a.kv == "gaac":
    print("\nKV cache:", llm.model_runner.gaac_stats())   # sequences have finished; tiles stay until the slot is reused
