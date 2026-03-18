from transformers import AutoTokenizer

path_to_tokenizer = "/workspace/datasets/tokenizer/llama2_tok"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading tokenizer from {model_id} (Non-gated Llama 2 mirror)...")
tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
print(f"Vocabulary Size: {tok.vocab_size}") # Should be 32000
# Save to your desired folder
tok.save_pretrained(path_to_tokenizer)
print(f"Saved to {path_to_tokenizer}")
print("----------------------------------------------------------------")
print(f"IMPORTANT: For Megatron-LM preprocessing, point to the .model file:")
print(f"Argument: --tokenizer-model {path_to_tokenizer}/tokenizer.model")