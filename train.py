import argparse
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ramp import convert_to_ramp, parameter_counts


MODEL_ID = "google/functiongemma-270m-it"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
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
                "properties": {"seconds": {"type": "integer"}},
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
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]

TRAIN_EXAMPLES = [
    ("What's the weather in Seoul?", "<start_function_call>call:get_weather{location:<escape>Seoul<escape>}<end_function_call>"),
    ("Check the weather in Busan.", "<start_function_call>call:get_weather{location:<escape>Busan<escape>}<end_function_call>"),
    ("How is the weather in Tokyo?", "<start_function_call>call:get_weather{location:<escape>Tokyo<escape>}<end_function_call>"),
    ("Search the web for SRAM accelerator papers.", "<start_function_call>call:web_search{query:<escape>SRAM accelerator papers<escape>}<end_function_call>"),
    ("Look up FunctionGemma tool calling.", "<start_function_call>call:web_search{query:<escape>FunctionGemma tool calling<escape>}<end_function_call>"),
    ("Search for recursive transformer research.", "<start_function_call>call:web_search{query:<escape>recursive transformer research<escape>}<end_function_call>"),
    ("Set a timer for 30 seconds.", "<start_function_call>call:set_timer{seconds:30}<end_function_call>"),
    ("Set a two minute timer.", "<start_function_call>call:set_timer{seconds:120}<end_function_call>"),
    ("Start a timer for five minutes.", "<start_function_call>call:set_timer{seconds:300}<end_function_call>"),
    ("Add 3 and 7.", "<start_function_call>call:add_numbers{a:3,b:7}<end_function_call>"),
    ("What is 25 plus 17?", "<start_function_call>call:add_numbers{a:25,b:17}<end_function_call>"),
    ("Add 128 and 256.", "<start_function_call>call:add_numbers{a:128,b:256}<end_function_call>"),
]

EVAL_EXAMPLES = [
    ("Tell me the weather in Incheon.", "get_weather"),
    ("Search online for weight sharing in LLMs.", "web_search"),
    ("Set a timer for 90 seconds.", "set_timer"),
    ("Add 11 and 31.", "add_numbers"),
]


def make_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {
            "role": "developer",
            "content": "You are a model that can do function calling with the following functions",
        },
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)


def make_batch(tokenizer, user_text: str, target: str, max_length: int, device: str):
    prompt = make_prompt(tokenizer, user_text)
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        prompt + target + eos,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    return input_ids, attention_mask, labels


def evaluate(model, tokenizer, device: str) -> int:
    model.eval()
    correct = 0
    for text, expected_tool in EVAL_EXAMPLES:
        prompt = make_prompt(tokenizer, text)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        match = re.search(r"call:([a-zA-Z0-9_]+)", generated)
        predicted = match.group(1) if match else "<none>"
        correct += int(predicted == expected_tool)
        print(f"[{predicted:>12}] expected={expected_tool:>12} | {text}")
    return correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=36)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--freeze-shared-base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This first prototype expects an NVIDIA GPU.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    student = convert_to_ramp(
        student,
        depth=args.depth,
        rank=args.rank,
        alpha=args.alpha,
        train_shared_base=not args.freeze_shared_base,
    )

    counts = parameter_counts(student)
    print("RAMP parameters:", {k: f"{v / 1e6:.2f}M" for k, v in counts.items()})
    print("Before training:")
    before = evaluate(student, tokenizer, device)
    print(f"tool selection: {before}/{len(EVAL_EXAMPLES)}")

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    student.train()
    for step in range(1, args.steps + 1):
        user_text, target = random.choice(TRAIN_EXAMPLES)
        input_ids, attention_mask, labels = make_batch(tokenizer, user_text, target, args.max_length, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        with torch.autocast(device_type="cuda", dtype=dtype):
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            ce_loss = student_out.loss

        target_mask = labels[:, 1:] != -100
        student_target = student_out.logits[:, :-1][target_mask].float()
        teacher_target = teacher_logits[:, :-1][target_mask].float()
        temperature = args.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_target / temperature, dim=-1),
            F.softmax(teacher_target / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        loss = ce_loss.float() + args.kd_weight * kd_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 20 == 0:
            print(
                f"step={step:4d} loss={loss.item():.4f} "
                f"ce={ce_loss.item():.4f} kd={kd_loss.item():.4f}"
            )

    print("After training:")
    after = evaluate(student, tokenizer, device)
    print(f"tool selection: {after}/{len(EVAL_EXAMPLES)}")
    torch.save(student.model.ramp_shared_layer.state_dict(), "ramp_shared_layer.pt")
    print("saved: ramp_shared_layer.pt")


if __name__ == "__main__":
    main()
