from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration
from .config import CITY_SHAPES, MAX_VALUE_DIM
LANGUAGE_LORA_SUFFIXES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')

def discover_language_lora_targets(model: nn.Module):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith('model.layers.'):
            continue
        if any((name.endswith(suffix) for suffix in LANGUAGE_LORA_SUFFIXES)):
            targets.append(name)
    targets = sorted(targets)
    num_layers = int(model.config.num_hidden_layers)
    expected = num_layers * len(LANGUAGE_LORA_SUFFIXES)
    if len(targets) != expected:
        raise RuntimeError(f'Unexpected Qwen2.5-VL language-module structure.\nFound {len(targets)} LoRA targets, expected {expected} ({num_layers} layers x {len(LANGUAGE_LORA_SUFFIXES)} modules).\nFirst targets: {targets[:20]}')
    bad = [name for name in targets if name.startswith('visual.')]
    if bad:
        raise RuntimeError('Vision modules were accidentally selected for LoRA:\n' + '\n'.join(bad[:20]))
    return targets

def find_unique_module(model: nn.Module, suffix: str):
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(f"Expected exactly one module ending with '{suffix}', found {len(matches)}: {names[:10]}")
    return matches[0]

def unfreeze_visual_merger(vlm: nn.Module):
    name, merger = find_unique_module(vlm, 'visual.merger')
    for p in merger.parameters():
        p.requires_grad = True
    return (name, merger)

def find_language_final_norm(vlm: nn.Module):
    return find_unique_module(vlm, 'model.norm')

def get_language_hidden_size(vlm: nn.Module) -> int:
    cfg = vlm.config
    text_cfg = getattr(cfg, 'text_config', None)
    if text_cfg is not None and hasattr(text_cfg, 'hidden_size'):
        return int(text_cfg.hidden_size)
    if hasattr(cfg, 'hidden_size'):
        return int(cfg.hidden_size)
    raise RuntimeError('Cannot resolve Qwen language hidden_size from config.')

class ValueHead(nn.Module):

    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, head_dim), nn.SiLU(), nn.Linear(head_dim, MAX_VALUE_DIM))

    def forward(self, x):
        return self.net(x)

class TensorHead(nn.Module):

    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, head_dim), nn.SiLU())
        self.city_heads = nn.ModuleDict()
        for city, (h, w) in CITY_SHAPES.items():
            key = self._key(city)
            self.city_heads[key] = nn.Linear(head_dim, 4 * 2 * h * w)

    @staticmethod
    def _key(city: str) -> str:
        return city.replace('-', '_')

    def forward_city(self, x: torch.Tensor, city: str):
        h, w = CITY_SHAPES[city]
        z = self.trunk(x)
        y = self.city_heads[self._key(city)](z)
        return y.view(*x.shape[:-1], 4, 2, h, w)

@dataclass
class Stage1Output:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    value_loss: torch.Tensor
    tensor_loss: torch.Tensor
    value_predictions: Optional[torch.Tensor] = None
    tensor_predictions: Optional[list] = None
    readout_states: Optional[torch.Tensor] = None

class Stage1UrbanVLM(nn.Module):

    def __init__(self, vlm: nn.Module, value_head_dim: int=512, tensor_head_dim: int=1024, lambda_value: float=1.0, lambda_tensor: float=1.0):
        super().__init__()
        self.vlm = vlm
        self.lambda_value = float(lambda_value)
        self.lambda_tensor = float(lambda_tensor)
        hidden_size = get_language_hidden_size(self.vlm)
        self.value_head = ValueHead(hidden_size, value_head_dim)
        self.tensor_head = TensorHead(hidden_size, tensor_head_dim)
        _, final_norm = find_language_final_norm(self.vlm)
        self._captured_hidden = None
        self._hidden_hook_handle = final_norm.register_forward_hook(self._capture_hidden_hook)

    def _capture_hidden_hook(self, module, inputs, output):
        self._captured_hidden = output

    def _readout_states(self, readout_positions: torch.Tensor):
        hidden = self._captured_hidden
        if hidden is None:
            raise RuntimeError('Final language hidden state was not captured.')
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, readout_positions]

    @staticmethod
    def _masked_value_mse(pred, target, mask):
        active_samples = mask.any(dim=1)
        if not active_samples.any():
            return pred.sum() * 0.0
        pred32 = pred.float()
        target32 = target.float()
        mask32 = mask.float()
        sq = (pred32 - target32).pow(2) * mask32
        per_sample = sq.sum(dim=1) / mask32.sum(dim=1).clamp_min(1.0)
        return per_sample[active_samples].mean()

    def _tensor_mse(self, readout, tensor_targets, return_predictions=False):
        sample_losses = []
        all_predictions = [] if return_predictions else None
        for i, targets in enumerate(tensor_targets):
            if not targets:
                if return_predictions:
                    all_predictions.append({})
                continue
            city_losses = []
            sample_preds = {}
            for city, target in targets.items():
                pred = self.tensor_head.forward_city(readout[i:i + 1], city)[0]
                loss = F.mse_loss(pred.float(), target.to(pred.device, dtype=torch.float32), reduction='mean')
                city_losses.append(loss)
                if return_predictions:
                    sample_preds[city] = pred.detach()
            sample_losses.append(torch.stack(city_losses).mean())
            if return_predictions:
                all_predictions.append(sample_preds)
        if not sample_losses:
            loss = readout.sum() * 0.0
        else:
            loss = torch.stack(sample_losses).mean()
        return (loss, all_predictions)

    def forward(self, input_ids, attention_mask, labels, pixel_values, image_grid_thw, readout_positions, value_target, value_mask, tensor_targets, return_predictions: bool=False):
        self._captured_hidden = None
        outputs = self.vlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False, output_hidden_states=False, return_dict=True)
        ce_loss = outputs.loss.float()
        readout = self._readout_states(readout_positions)
        value_pred = self.value_head(readout)
        value_loss = self._masked_value_mse(value_pred, value_target, value_mask)
        tensor_loss, tensor_predictions = self._tensor_mse(readout, tensor_targets, return_predictions=return_predictions)
        total = ce_loss + self.lambda_value * value_loss + self.lambda_tensor * tensor_loss
        return Stage1Output(loss=total, ce_loss=ce_loss, value_loss=value_loss, tensor_loss=tensor_loss, value_predictions=value_pred.detach() if return_predictions else None, tensor_predictions=tensor_predictions, readout_states=readout.detach() if return_predictions else None)

def build_trainable_stage1_model(cfg):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    base.config.use_cache = False
    lora_targets = discover_language_lora_targets(base)
    print(f'Discovered {len(lora_targets)} language LoRA target modules across {base.config.num_hidden_layers} decoder layers.')
    print('First LoRA targets:')
    for name in lora_targets[:7]:
        print('  ', name)
    lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias='none', task_type=TaskType.CAUSAL_LM, target_modules=lora_targets)
    vlm = get_peft_model(base, lora_cfg)
    merger_name, _ = unfreeze_visual_merger(vlm)
    if hasattr(vlm, 'gradient_checkpointing_enable'):
        try:
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        except TypeError:
            vlm.gradient_checkpointing_enable()
    if hasattr(vlm, 'enable_input_require_grads'):
        vlm.enable_input_require_grads()
    model = Stage1UrbanVLM(vlm=vlm, value_head_dim=cfg.value_head_dim, tensor_head_dim=cfg.tensor_head_dim, lambda_value=cfg.lambda_value, lambda_tensor=cfg.lambda_tensor)
    return (model, merger_name)

def trainable_parameter_report(model: Stage1UrbanVLM) -> dict:
    groups = {'lora': [0, 0], 'visual_merger': [0, 0], 'value_head': [0, 0], 'tensor_head': [0, 0], 'other_trainable': [0, 0]}
    for name, p in model.named_parameters():
        n = p.numel()
        if 'lora_' in name:
            key = 'lora'
        elif 'visual.merger' in name:
            key = 'visual_merger'
        elif name.startswith('value_head.'):
            key = 'value_head'
        elif name.startswith('tensor_head.'):
            key = 'tensor_head'
        elif p.requires_grad:
            key = 'other_trainable'
        else:
            continue
        groups[key][0] += n
        if p.requires_grad:
            groups[key][1] += n
    return {k: {'total': total, 'trainable': trainable} for k, (total, trainable) in groups.items()}
