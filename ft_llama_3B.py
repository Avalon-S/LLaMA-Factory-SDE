import json
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

args = {
    "stage": "sft",                        # do supervised fine-tuning
    "do_train": True,
    "model_name_or_path": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "dataset": "pubmedqa",             # use PubMedQA datasets
    "template": "llama3",                     # use llama3 prompt template
    "finetuning_type": "lora",                   # use LoRA adapters to save memory
    "lora_target": "all",                     # attach LoRA adapters to all linear layers
    "output_dir": "llama3_3B_pubmedqa",                  # the path to save LoRA adapters
    "per_device_train_batch_size": 2,               # the batch size
    "gradient_accumulation_steps": 4,               # the gradient accumulation steps
    "lr_scheduler_type": "cosine",                 # use cosine learning rate scheduler
    "logging_steps": 10,                      # log every 10 steps
    "warmup_ratio": 0.1,                      # use warmup scheduler
    "save_steps": 1000,                      # save checkpoint every 1000 steps
    "learning_rate": 1e-6,                     # the learning rate
    "num_train_epochs": 3.0,                    # the epochs of training
    "max_samples": 500,                      # use 500 examples in each dataset
    "max_grad_norm": 1.0,                     # clip gradient norm to 1.0
    "loraplus_lr_ratio": 16.0,                   # use LoRA+ algorithm with lambda=16.0
    "fp16": True,                         # use float16 mixed precision training
    "use_liger_kernel": True,                   # use liger kernel for efficient training
}

json_file_path = "train_llama3_3B_pubmedqa.json"
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(args, f, indent=2)

#os.chdir("")

os.system(f"llamafactory-cli train {json_file_path}")
