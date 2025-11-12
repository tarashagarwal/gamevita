CONFIG = {
    "image_encoder": "openai/clip-vit-base-patch32",
    "llm": "meta-llama/Llama-3.1-8B-Instruct",

    # how many image tokens to feed into the LLM (prefix tokens)
    "num_image_tokens": 8,

    # generation settings
    "max_new_tokens": 120,
    "num_beams": 5,
}
