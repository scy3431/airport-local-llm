# Bedrock Project — MLX LoRA Fine-Tuning Setup

This guide covers everything needed to go from a fresh clone of this repo to prompting the fine-tuned model, on Apple Silicon (macOS).

## Prerequisites

- macOS on Apple Silicon
- Homebrew
- Python 3.13
- A Hugging Face-accessible base model (this project uses `Qwen/Qwen3-0.6B`)

## 1. Environment Setup

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate

pip install mlx mlx-lm
```

Verify install:

```bash
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
```

## 2. Known Issue: MPI / MPICH Conflict

If you have Anaconda/conda installed alongside Homebrew, MLX's distributed-training
init can accidentally load Anaconda's MPICH library instead of Homebrew's real
Open MPI, and **hard-crashes training** with an error like:

```
[mpi] MPI found but it does not appear to be Open MPI. MLX requires Open MPI...
```

**Fix — scope your `PATH` and `DYLD_LIBRARY_PATH` for the shell session so
Anaconda is never on the library search path:**

```bash
export PATH="$PWD/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib"

# sanity check â€” these should resolve inside your .venv / Homebrew, not Anaconda
which python
which mpirun
python --version
mpirun --version
```

Run this in every new terminal session before training (or add it to a project
`.envrc` / shell alias â€” see "Optional: Make This Permanent" below).

## 3. Data

Training data lives in `data/` as JSONL files in chat format:

```
data/
â”œâ”€â”€ train.jsonl
â”œâ”€â”€ valid.jsonl
â””â”€â”€ test.jsonl
```

Each line is a chat-formatted example, e.g.:

```json
{"messages": [{"role": "system", "content": "You are an expert geographical assistant specializing in airport metadata. Answer airport questions using the information provided in your training."}, {"role": "user", "content": "Show data concerning airport code CLT."}, {"role": "assistant", "content": "The airport name is Charlotte Douglas International, located in Charlotte, US. Latitude: 35.2144, Longitude: -80.9473."}]}
```

## 4. Training (LoRA Fine-Tune)

```bash
rm -rf adapters
mkdir adapters

mlx_lm.lora \
    --model Qwen/Qwen3-0.6B \
    --train \
    --data data \
    --iters 10 \
    --batch-size 1 \
    --num-layers 4 \
    --adapter-path adapters \
    --mask-prompt
```

Confirm the adapter weights were actually written (this is the file that
matters — if training crashes partway through, only `adapter_config.json`
will exist and `adapters.safetensors` will be missing):

```bash
ls -lah adapters
```

You should see both `adapter_config.json` and `adapters.safetensors`.

> **Note on training strength:** 10 iterations / 4 layers / batch size 1 is a
> smoke test to confirm the pipeline works end-to-end, not a real training
> run. To get the model to actually learn the output format, increase:
> - `--iters` to 200-1000+
> - `--num-layers` to 8-16 (or `-1` for all layers)
> - `--batch-size` to 4-8 (120 training examples can support this)

## 5. Prompting the Fine-Tuned Model

```bash
mlx_lm.generate \
    --model Qwen/Qwen3-0.6B \
    --adapter-path adapters \
    --prompt "Give me the details for LAX." \
    --max-tokens 300
```

- `--adapter-path adapters` loads your LoRA weights on top of the base model.
- `--max-tokens` defaults to 100, which is too short for full responses —
  bump it up (300 is a reasonable starting point).
- Qwen3 is a thinking model by default, so part of the token budget goes
  toward a `<think>...</think>` block before the final answer â€” factor that
  into your `--max-tokens` budget.

## Quick Reference: Full Startup Sequence

```bash
cd bedrock-project
source .venv/bin/activate

export PATH="$PWD/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib"

mlx_lm.generate \
    --model Qwen/Qwen3-0.6B \
    --adapter-path adapters \
    --prompt "Your prompt here." \
    --max-tokens 300
```

## Optional: Make the PATH/DYLD Fix Permanent

Rather than exporting these every session, you can add a project-local
`.envrc` (if using [direnv](https://direnv.net/)) or a small wrapper script,
e.g. `bin/run.sh`:

```bash
#!/bin/bash
export PATH="$(dirname "$0")/../.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib"
exec "$@"
```

Usage: `./bin/run.sh mlx_lm.generate --model Qwen/Qwen3-0.6B --adapter-path adapters --prompt "..."`
