# RAMP: Recursive Adapter-based Model Expansion

RAMP studies whether a pretrained Transformer can trade **unique parameter depth** for **recursive compute depth**.

The first experiment uses `google/functiongemma-270m-it`:

```text
Teacher:  T1 -> T2 -> ... -> T18

Student:  S1 -> S2 -> ... -> S36
          \___________________/
             one shared W

Si = shared W + depth-specific LoRA_i
```

The hardware motivation is weight-stationary inference: keep the large shared decoder weight in on-chip SRAM and stream only small per-depth adapters while hidden states iterate through the same physical compute block.

## Prototype design

The repository intentionally stays small:

- `ramp.py`: recursive shared decoder and depth-specific LoRA
- `train.py`: FunctionGemma teacher-student experiment
- `requirements.txt`: pinned dependencies

FunctionGemma has 18 Gemma3 decoder layers with a repeating **5 sliding-attention + 1 full-attention** structure. RAMP does not replace this with arbitrary full attention. The teacher role of each layer is stretched across the deeper student. With depth 36, every teacher layer is represented by two recursive student steps.

```text
Teacher layer Ti  ->  Student steps S(2i-1), S(2i)
Teacher hidden Ti ->  distillation target at S(2i)
```

The physical shared block is initialized by the parameter-wise mean of the 18 pretrained decoder blocks. For stable depth expansion, recursive residual updates are scaled by

```text
residual_scale = teacher_depth / student_depth
```

so the default 18 -> 36 conversion starts at `0.5` residual strength per recursive step.

Each linear projection in the shared attention/MLP block receives an independent LoRA delta for every logical depth. Embedding and LM-head weights remain frozen. The shared base can either be trained with a small learning rate or frozen for an adapter-only ablation.

Training uses two losses:

1. supervised causal-LM loss on correctly formatted FunctionGemma tool calls,
2. hidden-state distillation between every teacher layer and its corresponding RAMP boundary.

Training examples are formatted through FunctionGemma's own `apply_chat_template(..., tools=...)`; tool-call strings are not manually concatenated.

## Install

Accept the Gemma license for `google/functiongemma-270m-it` on Hugging Face first.

```bash
git clone https://github.com/HigherIdeal/RAMP.git
cd RAMP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

The code is pinned to `transformers==4.57.3` because RAMP directly follows that Gemma3 decoder interface.

## Run

Default experiment:

```bash
python train.py --depth 36 --rank 8 --steps 200
```

Adapter-only smoke test:

```bash
python train.py --depth 36 --rank 4 --steps 20 --freeze-shared-base
```

The default maximum sequence length is 512. The script refuses to truncate a formatted function call; if a tool schema makes a sample longer, increase `--max-length` instead.

The teacher is loaded first, evaluated, and used to precompute layer-wise distillation targets on CPU. It is then removed from GPU memory before the RAMP student is loaded, which keeps the experiment practical on a small GPU such as an RTX 3050.

The output checkpoint `ramp.pt` stores only the shared decoder block, all depth-specific adapters, and the RAMP structural metadata. The unchanged FunctionGemma embedding/LM head are not duplicated in the checkpoint.

## Immediate experiments

The first useful comparison is:

| Model | Logical depth | Physical decoder blocks | Adapter rank |
|---|---:|---:|---:|
| FunctionGemma teacher | 18 | 18 | - |
| RAMP-18 | 18 | 1 | 4 / 8 / 16 |
| RAMP-36 | 36 | 1 | 4 / 8 / 16 |
| RAMP-54 | 54 | 1 | 4 / 8 / 16 |

Measure tool-call accuracy, physical parameter count, adapter parameter count, latency, FLOPs, and estimated off-chip weight traffic. The central question is whether deeper recursive compute can recover the capability lost by replacing many unique decoder blocks with one SRAM-resident shared block.

## Current limitation

KV cache is intentionally disabled. A normal Transformer cache is indexed by physical layer, whereas RAMP reuses one physical attention block across many logical depths. Correct cache semantics therefore require a separate logical-depth-aware cache design and should not be hidden behind the first proof-of-concept.
