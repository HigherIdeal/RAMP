import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepLoRALinear(nn.Module):
    """One shared linear weight plus a different low-rank delta per recursive step."""

    def __init__(self, base: nn.Linear, steps: int, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.step = 0

        self.lora_a = nn.Parameter(
            torch.empty(steps, rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(steps, base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.normal_(self.lora_a, mean=0.0, std=1.0 / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = F.linear(F.linear(x, self.lora_a[self.step]), self.lora_b[self.step])
        return y + delta * self.scale


def _inject_step_lora(module: nn.Module, steps: int, rank: int, alpha: float) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, StepLoRALinear(child, steps, rank, alpha))
        else:
            _inject_step_lora(child, steps, rank, alpha)


def _set_step(module: nn.Module, step: int) -> None:
    for child in module.modules():
        if isinstance(child, StepLoRALinear):
            child.step = step


class StepLayer(nn.Module):
    """Calls one physical decoder block with the adapter assigned to this depth."""

    def __init__(self, shared_layer: nn.Module, step: int):
        super().__init__()
        object.__setattr__(self, "_shared_layer", shared_layer)
        self.step = step

    def forward(self, *args, **kwargs):
        shared_layer = object.__getattribute__(self, "_shared_layer")
        _set_step(shared_layer, self.step)
        return shared_layer(*args, **kwargs)


@torch.no_grad()
def _average_layers_(target: nn.Module, source_layers: list[nn.Module]) -> None:
    sources = [dict(layer.named_parameters()) for layer in source_layers]
    for name, param in target.named_parameters():
        acc = torch.zeros_like(param, dtype=torch.float32)
        for source in sources:
            acc.add_(source[name].detach().float())
        param.copy_((acc / len(sources)).to(param.dtype))


def convert_to_ramp(
    model,
    depth: int = 36,
    rank: int = 8,
    alpha: float = 16.0,
    train_shared_base: bool = True,
):
    """Convert FunctionGemma/Gemma3 into a deeper shared-weight RAMP student.

    The first prototype uses full attention for every recursive step and disables
    KV cache. This makes one physical decoder block reusable at every depth.
    """
    if depth < 1 or rank < 1:
        raise ValueError("depth and rank must be positive")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a Hugging Face Gemma3ForCausalLM-like model")

    original_layers = list(model.model.layers)
    layer_types = list(model.config.layer_types)
    try:
        full_idx = layer_types.index("full_attention")
    except ValueError as exc:
        raise ValueError("RAMP prototype expects at least one full-attention Gemma3 layer") from exc

    for param in model.parameters():
        param.requires_grad_(False)

    shared_layer = copy.deepcopy(original_layers[full_idx])
    _average_layers_(shared_layer, original_layers)

    for param in shared_layer.parameters():
        param.requires_grad_(train_shared_base)
    _inject_step_lora(shared_layer, depth, rank, alpha)

    # Remove the original unique stack and register one physical decoder block.
    model.model.layers = nn.ModuleList()
    model.model.ramp_shared_layer = shared_layer
    model.model.layers = nn.ModuleList([StepLayer(shared_layer, step) for step in range(depth)])

    model.config.num_hidden_layers = depth
    model.config.layer_types = ["full_attention"] * depth
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    return model


def parameter_counts(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, StepLoRALinear)
        for p in (module.lora_a, module.lora_b)
    )
    return {"total": total, "trainable": trainable, "adapter": adapter}
