from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

# model_path = "merged_model_3B"

model_path = "merged_model_1B"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example input prompt
prompt = "Tell me how to keep healthy?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(inputs["input_ids"], max_length=500)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
