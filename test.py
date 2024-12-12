from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

# Configure quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

# Paths for the base model and LoRA weights
base_model = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
lora_weights_path = "llama3_3B_pubmedqa"

# base_model = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
# lora_weights_path = "llama3_1B_pubmedqa"

# Load the base model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Ensure LoRA weights are correctly loaded
model.load_adapter(lora_weights_path)

# Example input prompt
prompt = "Tell me how to keep healthy?\n\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


# Generate text
outputs = model.generate(inputs["input_ids"], max_length=500)

# Decode the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
