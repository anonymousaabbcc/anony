import json
from pathlib import Path
from urllib.parse import urlparse, unquote
from PIL import Image
from transformers import AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False
MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct'
QA_PATH = './outputs/qa/source/train/QA_train.jsonl'

def load_samples(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def pick_one(samples, template_id):
    for x in samples:
        if x['template_id'] == template_id:
            return x
    raise ValueError(f'Template {template_id} not found')

def uri_to_pil(uri):
    parsed = urlparse(uri)
    if parsed.scheme == 'file':
        path = unquote(parsed.path)
    else:
        path = uri
    return Image.open(path).convert('RGB')

def fallback_process_vision_info(messages):
    images = []
    for msg in messages:
        for item in msg.get('content', []):
            if isinstance(item, dict) and item.get('type') == 'image':
                images.append(uri_to_pil(item['image']))
    return (images, None)

def build_inputs(processor, sample):
    messages = sample['messages']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if HAS_QWEN_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = fallback_process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
    return (text, image_inputs, inputs)

def inspect_sample(processor, sample):
    print('=' * 100)
    print('Sample ID :', sample['sample_id'])
    print('Template  :', sample['template_id'])
    print('Cities    :', sample['cities'])
    print('Level     :', sample['region_level'])
    print('Question  :', sample['question'])
    print('Image num :', len(sample.get('image_paths', [])))
    text, image_inputs, inputs = build_inputs(processor, sample)
    print('\nRendered prompt length:', len(text))
    print('Loaded image count    :', len(image_inputs) if image_inputs is not None else 0)
    print('\nProcessor output tensors:')
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={tuple(v.shape)} dtype={v.dtype}')
        else:
            print(f'  {k}: {type(v)}')
    print('\nPASS')

def main():
    samples = load_samples(QA_PATH)
    test_templates = ['P3', 'P7', 'P8']
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=512 * 28 * 28, max_pixels=1280 * 28 * 28, use_fast=False)
    for tid in test_templates:
        sample = pick_one(samples, tid)
        inspect_sample(processor, sample)
if __name__ == '__main__':
    main()
