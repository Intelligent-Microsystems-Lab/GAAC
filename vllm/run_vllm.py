"""vLLM 0.17.1 + GAAC: calibrate the exponent bases, then generate with the compressed KV cache (or the stock FlashAttention backend).

  # 1. calibration (one long prompt; writes per-layer bases)
  python vllm/run_vllm.py --model Qwen/Qwen3.8-27B --calib bases_qwen38_27b.pt --prompt-len 32768
  # 2. GAAC vs FlashAttention, same prompts, greedy
  python vllm/run_vllm.py --model Qwen/Qwen3.8-27B --bases bases_qwen38_27b.pt --backend gaac --prompt-len 8192 --num-seqs 4
  python vllm/run_vllm.py --model Qwen/Qwen3.8-27B --backend fa --prompt-len 8192 --num-seqs 4

Prompts are WikiText-2 (test split for generation, train split for calibration) or --text <file>.  Prints the KV block count
(capacity at the same gpu_memory_utilization) and the generated text; --out saves the token ids for comparisons.
"""
import argparse, json, os, sys, time
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ROOT)
os.environ["PYTHONPATH"] = _ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")    # vLLM worker processes import gaac.vllm.backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--backend", default="gaac", choices=["gaac", "fa"])
    ap.add_argument("--bases"); ap.add_argument("--calib", help="calibration run: write the bases file here")
    ap.add_argument("--prompt-len", type=int, default=4096); ap.add_argument("--num-seqs", type=int, default=1); ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.85); ap.add_argument("--eager", action="store_true"); ap.add_argument("--text")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.calib:
        os.environ.update(GAAC_VLLM="1", GAAC_CALIB=a.calib); os.environ.pop("GAAC_BASES", None); a.eager = True; a.num_seqs = 1; a.gen = 1
    elif a.backend == "gaac":
        assert a.bases, "--bases (run --calib first)"
        os.environ.update(GAAC_VLLM="1", GAAC_BASES=os.path.abspath(a.bases))
    else:
        os.environ.pop("GAAC_VLLM", None)          # stock vLLM attention backend (FlashAttention)

    from vllm import LLM, SamplingParams, TokensPrompt

    if os.path.isdir(a.model): cfg_path = os.path.join(a.model, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(a.model, "config.json")
    cfg = json.load(open(cfg_path))   # read directly: older transformers do not know qwen3_5 (vLLM has its own config)
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}     # Qwen3.5-family VL checkpoints: text only
    llm = LLM(model=a.model, dtype="bfloat16", block_size=16, enable_prefix_caching=False, max_num_seqs=max(1, a.num_seqs),
              max_model_len=a.prompt_len + a.gen + 64, max_num_batched_tokens=max(8192, a.prompt_len), gpu_memory_utilization=a.gpu_util,
              enforce_eager=a.eager, seed=0, **mm)
    tok = llm.get_tokenizer()
    if a.text:
        text = open(a.text).read()
    else:
        from datasets import load_dataset
        text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train" if a.calib else "test")["text"])
    ids = tok(text[: a.prompt_len * 8 * a.num_seqs], add_special_tokens=False)["input_ids"]
    assert len(ids) >= a.prompt_len * a.num_seqs, "not enough text"
    prompts = [TokensPrompt(prompt_token_ids=ids[i * a.prompt_len:(i + 1) * a.prompt_len]) for i in range(a.num_seqs)]
    cc = llm.llm_engine.vllm_config.cache_config
    print(f"[{a.backend if not a.calib else 'calib'}] KV cache: {cc.num_gpu_blocks} blocks of {cc.block_size} tokens", flush=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=a.gen, ignore_eos=True), use_tqdm=False)
    print(f"generate: {time.perf_counter() - t0:.1f} s")
    if a.calib:
        print(f"bases written to {a.calib}" if os.path.exists(a.calib) else "calibration file missing"); return
    print("text[0]:", repr(outs[0].outputs[0].text[:300]))
    if a.out:
        json.dump(dict(backend=a.backend, blocks=cc.num_gpu_blocks, block_size=cc.block_size, tokens=[o.outputs[0].token_ids for o in outs]), open(a.out, "w"))


if __name__ == "__main__":      # vLLM may start its engine with multiprocessing spawn
    main()
