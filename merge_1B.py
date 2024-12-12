from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths for the base model and LoRA weights
base_model = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
lora_weights_path = "llama3_1B_pubmedqa"

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

# Load the LoRA weights
lora_model = PeftModel.from_pretrained(model, lora_weights_path)

# Merge LoRA weights into the base model
lora_model = lora_model.merge_and_unload()

# Save the complete model
save_path = "merged_model_1B"
lora_model.save_pretrained(save_path)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(save_path)

print(f"Model has been merged and saved to: {save_path}")
