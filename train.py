import argparse
import gc
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ramp import convert_to_ramp, parameter_counts


MODEL_ID = "google/functiongemma-270m-it"
SYSTEM_MSG = "You are a model that can do function calling with the following functions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a timer in seconds.",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "integer", "description": "Timer duration"}},
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

TRAIN_EXAMPLES = [
    ("What's the weather in Seoul?", "get_weather", {"location": "Seoul"}),
    ("Check the weather in Busan.", "get_weather", {"location": "Busan"}),
    ("How is the weather in Tokyo?", "get_weather", {"location": "Tokyo"}),
    ("What's the weather in London?", "get_weather", {"location": "London"}),
    ("Search the web for SRAM accelerator papers.", "web_search", {"query": "SRAM accelerator papers"}),
    ("Look up FunctionGemma tool calling.", "web_search", {"query": "FunctionGemma tool calling"}),
    ("Search for recursive Transformer research.", "web_search", {"query": "recursive Transformer research"}),
    ("Find papers about weight sharing in LLMs.", "web_search", {"query": "weight sharing LLM papers"}),
    ("Set a timer for 30 seconds.", "set_timer", {"seconds": 30}),
    ("Set a two minute timer.", "set_timer", {"seconds": 120}),
    ("Start a timer for five minutes.", "set_timer", {"seconds": 300}),
    ("Give me a 45 second timer.", "set_timer", {"seconds": 45}),
    ("Add 3 and 7.", "add_numbers", {"a": 3, "b": 7}),
    ("What is 25 plus 17?", "add_numbers", {"a": 25, "b": 17}),
    ("Add 128 and 256.", "add_numbers", {"a": 128, "b": 256}),
    ("Sum 1.5 and 2.25.", "add_numbers", {"a": 1.5, "b": 2.25}),
]

EVAL_EXAMPLES = [
    ("Tell me the weather in Incheon.", "get_weather"),
    ("Search online for weight sharing in LLMs.", "web_search"),
    ("Set a timer for 90 seconds.", "set_timer"),
    ("Add 11 and 31.", "add_numbers"),
]


def prompt_messages(user_text: str):
    return [
        {"role": "developer", "content": SYSTEM_MSG},
        {"role": "user", "content": user_text},
    ]


def encode_example(tokenizer, sample, max_length: int):
    user_text, tool_name, arguments = sample
    prompt = tokenizer.apply_chat_template(
        prompt_messages(user_text),
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_messages = prompt_messages(user_text) + [
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": tool_name, "arguments": arguments}}
            ],
        }
    ]
    full = tokenizer.apply_chat_template(
        full_messages,
        tools=TOOLS,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_ids = prompt["input_ids"]
    input_ids = full["input_ids"]
    if input_ids.shape[1] > max_length:
        raise ValueError(
            f"Formatted sample needs {input_ids.shape[1]} tokens but --max-length={max_length}. "
            "Increase --max-length; truncating a tool call would corrupt the target."
        )
    if input_ids.shape[1] < prompt_ids.shape[1] or not torch.equal(
        input_ids[:, : prompt_ids.shape[1]], prompt_ids
    ):
        raise RuntimeError("FunctionGemma chat template prefix changed; prompt/target masking is unsafe")

    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


@torch.no_grad()
def evaluate(model, tokenizer, device: str) -> int:
    model.eval()
    correct = 0
    for text, expected_tool in EVAL_EXAMPLES:
        inputs = tokenizer.apply_chat_template(
            prompt_messages(text),
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=model.config.use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        match = re.search(r"call:([a-zA-Z0-9_]+)", generated)
        predicted = match.group(1) if match else "<none>"
        correct += int(predicted == expected_tool)
        print(f"[{predicted:>12}] expected={expected_tool:>12} | {text}")
    return correct


@torch.no_grad()
def build_teacher_targets(teacher, encoded, device: str):
    """Store each teacher decoder-layer output on CPU, then the teacher can be freed."""
    all_targets = []
    for example in encoded:
        captured = []
        handles = [
            layer.register_forward_hook(lambda _module, _inputs, output: captured.append(output[0].detach()))
            for layer in teacher.model.layers
        ]
        teacher.model(
            input_ids=example["input_ids"].to(device),
            attention_mask=example["attention_mask"].to(device),
            use_cache=False,
        )
        for handle in handles:
            handle.remove()
        if len(captured) != len(teacher.model.layers):
            raise RuntimeError("Failed to capture all teacher decoder states")
        all_targets.append([state.squeeze(0).cpu().half() for state in captured])
    return all_targets


def normalized_hidden_mse(student_state: torch.Tensor, teacher_state: torch.Tensor) -> torch.Tensor:
    student_state = student_state.float()
    teacher_state = teacher_state.float()
    student_state = student_state / torch.sqrt(student_state.square().mean(dim=-1, keepdim=True) + 1e-6)
    teacher_state = teacher_state / torch.sqrt(teacher_state.square().mean(dim=-1, keepdim=True) + 1e-6)
    return F.mse_loss(student_state, teacher_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=36)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--base-lr", type=float, default=1e-5, help="Shared base learning rate")
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--freeze-shared-base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("RAMP training currently expects an NVIDIA GPU")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    encoded = [encode_example(tokenizer, sample, args.max_length) for sample in TRAIN_EXAMPLES]
    print("formatted lengths:", [x["input_ids"].shape[1] for x in encoded])

    print("Loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    print("Teacher evaluation:")
    teacher_score = evaluate(teacher, tokenizer, device)
    print(f"teacher tool selection: {teacher_score}/{len(EVAL_EXAMPLES)}")
    teacher_targets = build_teacher_targets(teacher, encoded, device)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading RAMP student...")
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    student = convert_to_ramp(
        student,
        depth=args.depth,
        rank=args.rank,
        alpha=args.alpha,
        train_shared_base=not args.freeze_shared_base,
    )
    counts = parameter_counts(student)
    print("RAMP parameters:", {k: f"{v / 1e6:.2f}M" for k, v in counts.items()})
    print(
        f"teacher depth={student.model.ramp_teacher_depth}, student depth={args.depth}, "
        f"residual_scale={student.model.ramp_residual_scale:.3f}"
    )

    print("Student before training:")
    before = evaluate(student, tokenizer, device)
    print(f"student tool selection: {before}/{len(EVAL_EXAMPLES)}")

    shared = student.model.ramp_shared_layer
    adapter_params = []
    base_params = []
    for name, param in shared.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_a" in name or "lora_b" in name:
            adapter_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": args.lr})
    if base_params:
        param_groups.append({"params": base_params, "lr": args.base_lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters. Use rank > 0 or unfreeze the shared base.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    boundaries = student.model.ramp_boundary_steps

    student.train()
    for step in range(1, args.steps + 1):
        sample_idx = random.randrange(len(encoded))
        example = encoded[sample_idx]
        input_ids = example["input_ids"].to(device)
        attention_mask = example["attention_mask"].to(device)
        labels = example["labels"].to(device)

        captured = [None] * len(boundaries)
        handles = []
        for teacher_idx, student_step in enumerate(boundaries):
            def save_state(_module, _inputs, output, idx=teacher_idx):
                captured[idx] = output[0]
            handles.append(student.model.layers[student_step].register_forward_hook(save_state))

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            model_out = student.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        for handle in handles:
            handle.remove()

        if any(state is None for state in captured):
            raise RuntimeError("Failed to capture RAMP boundary states")

        # Causal LM: hidden state at t-1 predicts the target token at t.
        target_positions = torch.where(labels[0] != -100)[0]
        target_positions = target_positions[target_positions > 0]
        pred_positions = target_positions - 1
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = student.lm_head(model_out.last_hidden_state[:, pred_positions, :])
        ce_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels[:, target_positions].reshape(-1),
        )

        kd_loss = torch.zeros((), device=device)
        for teacher_idx, student_state in enumerate(captured):
            teacher_state = teacher_targets[sample_idx][teacher_idx].to(device)
            kd_loss = kd_loss + normalized_hidden_mse(student_state.squeeze(0), teacher_state)
        kd_loss = kd_loss / len(captured)

        loss = ce_loss + args.kd_weight * kd_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for group in param_groups for p in group["params"]], 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 20 == 0:
            print(
                f"step={step:4d} loss={loss.item():.4f} "
                f"ce={ce_loss.item():.4f} hidden_kd={kd_loss.item():.4f}"
            )

    print("Student after training:")
    after = evaluate(student, tokenizer, device)
    print(f"student tool selection: {after}/{len(EVAL_EXAMPLES)}")

    torch.save(
        {
            "model_id": MODEL_ID,
            "depth": args.depth,
            "rank": args.rank,
            "alpha": args.alpha,
            "residual_scale": student.model.ramp_residual_scale,
            "layer_types": student.config.layer_types,
            "shared_state_dict": student.model.ramp_shared_layer.state_dict(),
        },
        "ramp.pt",
    )
    print("saved: ramp.pt")


if __name__ == "__main__":
    main()
