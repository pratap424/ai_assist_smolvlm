import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import decord
import numpy as np

MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model once
torch.set_float32_matmul_precision('high')  # Optional performance tweak
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).to(DEVICE)

@torch.inference_mode()
def describe_image(image_path: str, max_new_tokens: int = 128) -> str:
    """
    Describe a single image in detail.
    """
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "path": image_path},
            {"type": "text",  "text": "Please describe this image in detail."}
        ]
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return outputs[0]

@torch.inference_mode()
def describe_video(
    video_path: str,
    num_keyframes: int = 4,
    max_frame_tokens: int = 32,
    max_summary_tokens: int = 128
) -> str:
    """
    Generate a cohesive description of the entire video by:
      1. Sampling fewer keyframes for speed.
      2. Captioning each keyframe briefly.
      3. Summarizing all captions into one paragraph.
    """
    def to_bf16(inputs):
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)
        return inputs

    def caption_frame(frame: np.ndarray) -> str:
        pil_img = Image.fromarray(frame).convert("RGB")
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": "Describe this frame briefly."}
            ]
        }]
        inputs = processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(DEVICE)
        inputs = to_bf16(inputs)
        gen_ids = model.generate(**inputs, max_new_tokens=max_frame_tokens)
        return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    def summarize_captions(captions: list) -> str:
        bulleted = "\n".join(f"- {c}" for c in captions)
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Here are brief captions for key moments of a video:"},
                {"type": "text", "text": bulleted},
                {"type": "text", "text": "Now please write a single, coherent description of the entire video."}
            ]
        }]
        inputs = processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(DEVICE)
        inputs = to_bf16(inputs)
        gen_ids = model.generate(**inputs, max_new_tokens=max_summary_tokens)
        return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    # Fewer frames sampled for speed
    indices = np.linspace(0, total_frames - 1, num=num_keyframes, dtype=int)

    captions = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        try:
            captions.append(caption_frame(frame))
        except Exception:
            # Skip problematic frames
            continue

    if not captions:
        return "Unable to generate video description."  

    return summarize_captions(captions)
