import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # pre-trained vision–language model


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model {MODEL_ID} on {device} ({dtype})...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def describe_image(model, processor, image_path: str, question: str):
    image = Image.open(image_path).convert("RGB")

    # LLaVA expects a chat-like prompt with <image> token
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device, torch.float16 if next(model.parameters()).dtype == torch.float16 else torch.float32)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    # LLaVA responses sometimes echo the prompt; this strips special tokens.
    answer = processor.decode(output_ids[0], skip_special_tokens=True)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Image → text with LLaVA")
    parser.add_argument("image", help="Path to image file (png/jpg/jpeg)")
    parser.add_argument(
        "-q",
        "--question",
        default="Describe this image in detail.",
        help="Question to ask about the image",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"Image not found: {img_path}")

    model, processor = load_model()
    caption = describe_image(model, processor, str(img_path), args.question)

    print("\n=== MODEL OUTPUT ===\n")
    print(caption)


if __name__ == "__main__":
    main()
