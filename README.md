# RAMP: Recursive Adapter-based Model Expansion

RAMP tests a simple hypothesis: **a shallow Transformer with unique weights can be distilled into a deeper model that repeatedly reuses one physical decoder block, while lightweight depth-specific LoRA adapters provide per-layer variation.**

The first prototype uses `google/functiongemma-270m-it` because it is small and has an explicit tool-calling task. FunctionGemma has 18 decoder layers; the default RAMP student replaces them with **36 recursive steps**.

```text
Teacher:  L1 -> L2 -> ... -> L18

Student:  (W + A1B1) -> (W + A2B2) -> ... -> (W + A36B36)
                    same shared W at every step
```

The long-term hardware target is weight-stationary inference: keep the large shared weight `W` in on-chip SRAM and change only the much smaller depth-specific adapters.

## Current prototype

Only four files are used intentionally. `ramp.py` contains the architecture conversion and `train.py` contains the toy teacher-student experiment.

The implementation currently:

- averages the pretrained 18 decoder blocks to initialize one shared block,
- reuses that block for 36 recursive steps,
- attaches a separate low-rank delta to every linear projection at each step,
- freezes embeddings and the LM head,
- trains the shared block + adapters using supervised tool-call loss and teacher-logit KD,
- uses full attention at every recursive step,
- disables KV cache because one physical attention block is reused across logical depths.

The last two points are deliberate prototype simplifications, not final architectural decisions.

## Run

FunctionGemma may require accepting Google's Gemma license on Hugging Face and logging in first.

```bash
git clone https://github.com/HigherIdeal/RAMP.git
cd RAMP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
python train.py
```

Default settings are chosen for a first RTX 3050 experiment:

```bash
python train.py --depth 36 --rank 8 --steps 200 --max-length 128
```

For a smaller first smoke test:

```bash
python train.py --depth 24 --rank 4 --steps 20 --max-length 96 --freeze-shared-base
```

`train.py` evaluates four held-out tool-selection prompts before and after training and saves only the physical shared decoder block plus its step adapters to `ramp_shared_layer.pt`.

## Research questions

The immediate experiment is not intended to prove SRAM efficiency by itself. It tests whether parameter capacity can be traded for recursive compute without destroying tool-use capability. The next comparisons should be:

1. original 18-layer FunctionGemma teacher,
2. 18-step shared-weight RAMP,
3. 36-step RAMP,
4. 54/72-step RAMP,
5. rank 0/4/8/16 adapters.

Primary metrics should be tool-call accuracy, trainable/physical parameter count, FLOPs, latency, and estimated off-chip weight traffic. If the deeper shared student retains teacher capability, SRAM residency and accelerator mapping become the next stage.
