import json
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct'
JSONL_PATH = '../../outputs/QA_source_chi_v3_finegrain/train/QA_train.jsonl'
processor = AutoProcessor.from_pretrained(MODEL_NAME)
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    sample = json.loads(next(f))
msgs = sample['messages']
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
image_inputs, video_inputs = process_vision_info(msgs)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
print(sample['template_id'], sample['target_type'], sample.get('numeric_target'), sample.get('tensor_target_ref'))
print({k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')})
