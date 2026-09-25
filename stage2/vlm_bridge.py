from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from stage1.checkpoint import load_stage1_checkpoint
from stage1.model import get_language_hidden_size

@dataclass
class FrozenContext:
    s_qa: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prompt_lengths: torch.Tensor

def _unwrap_qwen(vlm):
    if hasattr(vlm, 'get_base_model'):
        base = vlm.get_base_model()
    else:
        base = vlm
    required = ['visual', 'model', 'get_rope_index', 'get_input_embeddings', 'config']
    missing = [x for x in required if not hasattr(base, x)]
    if missing:
        raise RuntimeError(f'Could not unwrap Qwen2.5-VL base model; missing {missing}')
    return base

class FrozenStage1Bridge(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        if cfg.stage1_source == 'grounded':
            stage1, processor, stage1_cfg = load_stage1_checkpoint(checkpoint_dir=cfg.stage1_checkpoint, model_name=cfg.model_name, torch_dtype=torch.bfloat16, device=str(device), trainable=False)
            qwen_owner = stage1.vlm
        elif cfg.stage1_source == 'base_pretrained':
            stage1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
            stage1.config.use_cache = False
            stage1.to(device)
            qwen_owner = stage1
            ref_cfg_path = Path(cfg.stage1_checkpoint) / 'stage1_config.json'
            if ref_cfg_path.exists():
                stage1_cfg = json.loads(ref_cfg_path.read_text(encoding='utf-8'))
            else:
                stage1_cfg = {'min_pixels': 512 * 28 * 28, 'max_pixels': 1280 * 28 * 28, 'max_seq_length': 8192}
            processor = AutoProcessor.from_pretrained(cfg.model_name, min_pixels=int(stage1_cfg.get('min_pixels', 512 * 28 * 28)), max_pixels=int(stage1_cfg.get('max_pixels', 1280 * 28 * 28)), use_fast=False)
        else:
            raise ValueError(f'Unknown stage1_source={cfg.stage1_source!r}')
        for p in stage1.parameters():
            p.requires_grad_(False)
        stage1.eval()
        self.stage1 = stage1
        self.processor = processor
        self.stage1_cfg = stage1_cfg
        object.__setattr__(self, '_qwen_ref', _unwrap_qwen(qwen_owner))
        self.hidden_size = get_language_hidden_size(qwen_owner)
        self._assert_frozen()

    def _assert_frozen(self):
        bad = [name for name, p in self.stage1.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f'Stage-1 must be completely frozen; trainable examples: {bad[:10]}')

    def train(self, mode: bool=True):
        super().train(mode)
        self.stage1.eval()
        self.qwen.eval()
        return self

    @property
    def qwen(self):
        return self._qwen_ref

    @property
    def trainable_parameter_count(self):
        return sum((p.numel() for p in self.stage1.parameters() if p.requires_grad))

    def _fuse_multimodal_embeddings(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        image_grid_thw = batch['image_grid_thw']
        embeds = self.qwen.get_input_embeddings()(input_ids)
        pixel_values = pixel_values.to(dtype=self.qwen.visual.dtype)
        image_embeds = self.qwen.visual(pixel_values, grid_thw=image_grid_thw)
        n_tokens = int((input_ids == self.qwen.config.image_token_id).sum().item())
        if n_tokens != int(image_embeds.shape[0]):
            raise RuntimeError(f'Image token/features mismatch: tokens={n_tokens}, features={image_embeds.shape[0]}')
        mask = (input_ids == self.qwen.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(mask, image_embeds.to(embeds.device, embeds.dtype))
        position_ids, _ = self.qwen.get_rope_index(input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None, second_per_grid_ts=None, attention_mask=attention_mask)
        return (embeds, position_ids)

    @torch.no_grad()
    def prepare_context(self, batch) -> FrozenContext:
        self.stage1.eval()
        embeds, position_ids = self._fuse_multimodal_embeddings(batch)
        attention_mask = batch['attention_mask'].to(embeds.device)
        lengths = batch['prompt_lengths'].to(embeds.device)
        out = self.qwen.model(input_ids=None, inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
        hidden = out.last_hidden_state
        idx = torch.arange(hidden.shape[0], device=hidden.device)
        s = hidden[idx, lengths - 1]
        return FrozenContext(s_qa=s.detach(), inputs_embeds=embeds.detach(), attention_mask=attention_mask.detach(), position_ids=position_ids.detach(), prompt_lengths=lengths.detach())

    def _reenter_impl(self, context: FrozenContext, z: torch.Tensor):
        if z.ndim != 2 or z.shape[-1] != self.hidden_size:
            raise ValueError(f'Expected Z [B,{self.hidden_size}], got {tuple(z.shape)}')
        ctx = context.inputs_embeds
        b, l, d = ctx.shape
        if z.shape[0] != b:
            raise ValueError('Context/Z batch mismatch')
        z = z.to(dtype=ctx.dtype)
        positions = context.prompt_lengths
        ext = F.pad(ctx, (0, 0, 0, 1))
        slot = F.one_hot(positions, num_classes=l + 1).to(ctx.dtype).unsqueeze(-1)
        ext = ext * (1.0 - slot) + slot * z.unsqueeze(1)
        mask_ext = F.pad(context.attention_mask, (0, 1), value=0).clone()
        mask_ext.scatter_(1, positions.unsqueeze(-1), 1)
        pos_ext = F.pad(context.position_ids, (0, 1), value=0).clone()
        gather_idx = (context.prompt_lengths - 1).view(1, b, 1).expand(3, b, 1)
        last_pos = torch.gather(context.position_ids, 2, gather_idx).squeeze(-1)
        latent_pos = last_pos + 1
        for axis in range(3):
            pos_ext[axis].scatter_(1, positions.unsqueeze(-1), latent_pos[axis].unsqueeze(-1))
        out = self.qwen.model(input_ids=None, inputs_embeds=ext, attention_mask=mask_ext, position_ids=pos_ext, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
        hidden = out.last_hidden_state
        rows = torch.arange(b, device=hidden.device)
        return hidden[rows, positions]

    def reenter(self, context: FrozenContext, z: torch.Tensor):
        return self._reenter_impl(context, z)
