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
        self.img_encoder = AutoModel.from_pretrained(
            CONFIG["image_encoder"]
        ).to(device)

        # ===== Create adapter layer ======
        print("Creating Adapter Layer...")
        self.adapter = torch.nn.Linear(
            CONFIG["adapter_dim_in"],
            CONFIG["adapter_dim_out"]
        ).to(device)

        # ===== Load LLaMA (or any LLM) =====
        print("Loading LLM:", CONFIG["llm"])
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            CONFIG["llm"],
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def encode_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            # for CLIP: get only image features
            img_emb = self.img_encoder.get_image_features(
                pixel_values=pixel_values
            )  # shape (1, 512)

        return img_emb

    def adapt(self, img_emb):
        return self.adapter(img_emb)

    def generate_text(self, adapted_emb):
        prompt = "Describe this image in detail: "
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = toks.input_ids

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # make dtypes match
        adapted_emb = adapted_emb.to(inputs_embeds.dtype).unsqueeze(1)  # (1, 1, 4096)
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


CONFIG = {
    "image_encoder": "openai/clip-vit-base-patch32",
    "adapter_dim_in": 512,     # output dim of encoder
    "adapter_dim_out": 4096,   # LLaMA hidden size
    "llm": "meta-llama/Llama-3.1-8B-Instruct",
}


if __name__ == "__main__":
    pipeline = ImageToTextPipeline()
    result = pipeline.run("test3.jpeg")
    print("\n=== IMAGE DESCRIPTION ===")
    print(result)
