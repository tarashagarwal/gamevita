CONFIG = {
    "image_encoder": "openai/clip-vit-base-patch32",
    "adapter_dim_in": 512,     # output dim of encoder
    "adapter_dim_out": 4096,   # LLaMA hidden size
    "llm": "meta-llama/Llama-3.1-8B-Instruct",
}
