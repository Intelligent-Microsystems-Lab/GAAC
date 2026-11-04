# GAAC in vLLM (NVIDIA container `vllm:26.03-py3`)

GAAC runs inside vLLM as an attention backend: vLLM's paged KV cache holds GAAC-compressed blocks, and decoding reads
them directly. vLLM keeps its scheduler, block manager, chunked prefill, prefix caching and CUDA graphs.

| | |
| --- | --- |
| Platform | NVIDIA DGX Spark |
| Container | `nvcr.io/nvidia/vllm:26.03-py3` |
| Models | decoder attention with head_dim 128 or 256 and GQA group ≤ 8: Qwen3 dense and the Qwen3.5/3.6/3.8 hybrids (only their full-attention layers use GAAC) |
| vLLM changes | `vllm-0.17.1-gaac.patch` (for the vLLM in this container), 5 small edits that do nothing unless `GAAC_VLLM=1` |

## Setup

```bash
docker run --gpus all -it --rm -v $PWD:/gaac -w /gaac nvcr.io/nvidia/vllm:26.03-py3 bash
# inside the container:
cd $(python -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") \
  && patch -p1 < /gaac/vllm/vllm-0.17.1-gaac.patch && cd /gaac
pip install -e /gaac --no-deps
```

## Run

```bash
# 1. calibrate the exponent bases once per model (use a prompt about as long as your deployment context)
python vllm/run_vllm.py --model Qwen/Qwen3.8-27B --calib bases_qwen38_27b.pt --prompt-len 32768

# 2. generate with GAAC
python vllm/run_vllm.py --model Qwen/Qwen3.8-27B --backend gaac --bases bases_qwen38_27b.pt --prompt-len 8192 --num-seqs 2

# 3. the same with stock vLLM attention, for comparison
python vllm/run_vllm.py --model Qwen/Qwen3.8-27B --backend fa --prompt-len 8192 --num-seqs 2
```

The script prints the number of KV blocks vLLM allocated, which shows the extra capacity at the same
`--gpu-util`, and the generated text. `--out` saves the token ids.

To use GAAC from your own code, set the environment before creating `vllm.LLM`, and use `block_size=16` and
`dtype="bfloat16"`:

```python
import os
os.environ["GAAC_VLLM"] = "1"
os.environ["GAAC_BASES"] = "bases_qwen38_27b.pt"
from vllm import LLM
llm = LLM(model="Qwen/Qwen3.8-27B", dtype="bfloat16", block_size=16)
```

| Variable | Meaning |
| --- | --- |
| `GAAC_VLLM=1` | enable GAAC (patched vLLM only) |
| `GAAC_BASES` | bases file written by the calibration step |
| `GAAC_OVF_SLOT` | optional: escape bytes reserved per compressed tile (default 128) |
| `GAAC_RAW_TILES` | optional: size of the per-layer pool for blocks kept as raw BF16 |
