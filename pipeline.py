import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM
)
from PIL import Image
from config import CONFIG


class ImageToTextPipeline:
    def __init__(self, device="cuda"):
        self.device = device

        # ===== Load image encoder ======
        print("Loading Image Encoder:", CONFIG["image_encoder"])
        self.processor = AutoProcessor.from_pretrained(CONFIG["image_encoder"])
        self.img_encoder = AutoModel.from_pretrained(CONFIG["image_encoder"]).to(device)

        # ===== Create adapter layer ======
        print("Creating Adapter Layer...")
        self.adapter = torch.nn.Linear(
            CONFIG["adapter_dim_in"],
            CONFIG["adapter_dim_out"]
        ).to(device)

        # ===== Load LLaMA (or any LLM) =====
        print("Loading LLM:", CONFIG["llm"])
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm"])
        self.llm = AutoModelForCausalLM.from_pretrained(
            CONFIG["llm"],
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def encode_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.img_encoder(**inputs)
            # CLIP outputs last_hidden_state + pooled_output
            if hasattr(outputs, "pooler_output"):
                img_emb = outputs.pooler_output
            else:
                img_emb = outputs.last_hidden_state.mean(dim=1)

        return img_emb

    def adapt(self, img_emb):
        return self.adapter(img_emb)

    def generate_text(self, adapted_emb):
        # Expand into prompt tokens
        prompt = "Describe this image in detail: "
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids

        # Inject adapted embeddings as prefix
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        adapted_emb = adapted_emb.unsqueeze(1)  # (1, 1, hidden)
        fused = torch.cat([adapted_emb, inputs_embeds], dim=1)

        with torch.no_grad():
            output = self.llm.generate(
                inputs_embeds=fused,
                max_length=120,
                num_beams=5
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def run(self, image_path):
        img_emb = self.encode_image(image_path)
        adapted = self.adapt(img_emb)
        caption = self.generate_text(adapted)
        return caption


if __name__ == "__main__":
    pipeline = ImageToTextPipeline()
    result = pipeline.run("demo.jpg")
    print("\n=== IMAGE DESCRIPTION ===")
    print(result)
