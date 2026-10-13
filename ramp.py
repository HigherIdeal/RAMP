import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepLoRALinear(nn.Module):
    """One shared linear projection with a separate LoRA delta for each logical depth."""

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
        a = self.lora_a[self.step]
        b = self.lora_b[self.step]
        return y + F.linear(F.linear(x, a), b) * self.scale


def _inject_step_lora(module: nn.Module, steps: int, rank: int, alpha: float) -> None:
    if rank == 0:
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, StepLoRALinear(child, steps, rank, alpha))
        else:
            _inject_step_lora(child, steps, rank, alpha)


def _set_step(module: nn.Module, step: int) -> None:
    for child in module.modules():
        if isinstance(child, StepLoRALinear):
            child.step = step


@torch.no_grad()
def _mean_initialize_(target: nn.Module, source_layers: list[nn.Module]) -> None:
    """Initialize the shared block with the parameter-wise mean of teacher blocks."""
    sources = [dict(layer.named_parameters()) for layer in source_layers]
    for name, param in target.named_parameters():
        if any(name not in source for source in sources):
            raise RuntimeError(f"Cannot average decoder parameter: {name}")
        mean = torch.zeros_like(param, dtype=torch.float32)
        for source in sources:
            mean.add_(source[name].detach().float())
        mean.div_(len(sources))
        param.copy_(mean.to(device=param.device, dtype=param.dtype))


class RampStep(nn.Module):
    """A logical Gemma3 decoder depth backed by one shared physical decoder block."""

    def __init__(
        self,
        shared_layer: nn.Module,
        step: int,
        attention_type: str,
        residual_scale: float,
    ):
        super().__init__()
        # The shared block is registered exactly once as model.ramp_shared_layer.
        object.__setattr__(self, "_shared_layer", shared_layer)
        self.step = step
        self.attention_type = attention_type
        self.residual_scale = residual_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global,
        position_embeddings_local,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        if use_cache or past_key_values is not None:
            raise RuntimeError("RAMP does not support KV cache yet; use use_cache=False")

        shared = object.__getattribute__(self, "_shared_layer")
        _set_step(shared, self.step)

        is_sliding = self.attention_type == "sliding_attention"
        shared.self_attn.is_sliding = is_sliding
        shared.self_attn.sliding_window = shared.config.sliding_window if is_sliding else None
        position_embeddings = position_embeddings_local if is_sliding else position_embeddings_global

        residual = hidden_states
        x = shared.input_layernorm(hidden_states)
        x, attn_weights = shared.self_attn(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            output_attentions=output_attentions,
            use_cache=False,
            cache_position=cache_position,
            **kwargs,
        )
        x = shared.post_attention_layernorm(x)
        hidden_states = residual + x * self.residual_scale

        residual = hidden_states
        x = shared.pre_feedforward_layernorm(hidden_states)
        x = shared.mlp(x)
        x = shared.post_feedforward_layernorm(x)
        hidden_states = residual + x * self.residual_scale

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


def convert_to_ramp(
    model,
    depth: int = 36,
    rank: int = 8,
    alpha: float = 16.0,
    train_shared_base: bool = True,
):
    """Convert pretrained FunctionGemma into a deeper shared-weight RAMP student.

    This implementation targets transformers==4.57.3. It preserves Gemma3's
    sliding/full-attention roles while expanding each teacher layer into one or
    more recursive student steps.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected Gemma3ForCausalLM / FunctionGemma")
    if rank < 0:
        raise ValueError("rank must be >= 0")

    original_layers = list(model.model.layers)
    teacher_depth = len(original_layers)
    if depth < teacher_depth:
        raise ValueError(f"RAMP depth must be >= teacher depth ({teacher_depth})")

    original_types = list(model.config.layer_types)
    if len(original_types) != teacher_depth:
        raise RuntimeError("config.layer_types does not match decoder depth")

    for param in model.parameters():
        param.requires_grad_(False)

    shared_layer = copy.deepcopy(original_layers[0])
    _mean_initialize_(shared_layer, original_layers)
    for param in shared_layer.parameters():
        param.requires_grad_(train_shared_base)
    _inject_step_lora(shared_layer, depth, rank, alpha)

    # Stretch each teacher layer's role across the deeper student.
    teacher_index_for_step = [min((step * teacher_depth) // depth, teacher_depth - 1) for step in range(depth)]
    student_layer_types = [original_types[i] for i in teacher_index_for_step]
    boundary_steps = [
        max(step for step, teacher_idx in enumerate(teacher_index_for_step) if teacher_idx == i)
        for i in range(teacher_depth)
    ]
    residual_scale = teacher_depth / depth

    model.model.layers = nn.ModuleList()
    model.model.ramp_shared_layer = shared_layer
    model.model.layers = nn.ModuleList(
        [
            RampStep(shared_layer, step, student_layer_types[step], residual_scale)
            for step in range(depth)
        ]
    )
    model.model.ramp_boundary_steps = boundary_steps
    model.model.ramp_teacher_depth = teacher_depth
    model.model.ramp_residual_scale = residual_scale

    model.config.num_hidden_layers = depth
    model.config.layer_types = student_layer_types
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    return model


def parameter_counts(model) -> dict[str, int]:
    adapter = 0
    for module in model.modules():
        if isinstance(module, StepLoRALinear):
            adapter += module.lora_a.numel() + module.lora_b.numel()
    return {
        "physical": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter": adapter,
    }
