# -*- coding: utf-8 -*-
"""GEN_ATC_LLM_Mistral-7B-v0.1_v1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w0Wzrb_V93uwy5JN4eujnaeEbXN7Bz5r
"""

# Install required libraries
!pip install datasets trl peft uuid pandas evaluate transformers bitsandbytes torch huggingface_hub

# Imports
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
import time
import os
from uuid import uuid4

# Login to Hugging Face Hub if needed
from huggingface_hub import notebook_login
notebook_login()

# Define helper functions for calculating parameters and memory usage
def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_lora_parameters(lora_model):
    return sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

def estimate_full_model_memory(model, batch_size, seq_length):
    model_params = calculate_model_parameters(model)
    model_memory_bytes = model_params / 2  # INT4
    gradient_memory_bytes = 2 * model_params * 2  # FP16
    optimizer_memory_bytes = 2 * gradient_memory_bytes
    activation_memory_bytes = 4 * batch_size * seq_length * 2  # FP16
    total_memory_bytes = (model_memory_bytes + gradient_memory_bytes +
                          optimizer_memory_bytes + activation_memory_bytes)
    return total_memory_bytes / (1024 ** 3)

def estimate_lora_memory(model, lora_model, batch_size, seq_length):
    """
    Estimate the memory required for LoRA fine-tuning with additional factors.
    """
    print("\nLoRA training model parameters")
    model_params = calculate_model_parameters(model)
    lora_params = calculate_lora_parameters(lora_model)

    # Memory for model parameters (INT4) and LoRA parameters (INT4)
    model_memory_bytes = model_params / 2  # INT4: 4 bits per parameter
    lora_memory_bytes = lora_params / 2  # INT4: 4 bits per parameter
    print(f"Model Memory: {model_memory_bytes / (1024 ** 3)} GB")
    print(f"LoRA Memory: {lora_memory_bytes / (1024 ** 3)} GB")

    # Memory for gradients (FP16)
    gradient_memory_bytes = 2 * lora_params * 2  # FP16: 16 bits per parameter
    print(f"Gradient Memory: {gradient_memory_bytes / (1024 ** 3)} GB")

    # Optimizer state memory (FP16, assuming Adam optimizer)
    optimizer_memory_bytes = 2 * gradient_memory_bytes
    print(f"Optimizer State Memory: {optimizer_memory_bytes / (1024 ** 3)} GB")

    # Activation memory (FP16, estimate)
    activation_memory_bytes = 4 * batch_size * seq_length * 2  # FP16: 16 bits per activation
    print(f"Activation Memory: {activation_memory_bytes / (1024 ** 3)} GB")

    # Total memory
    total_memory_bytes = model_memory_bytes + lora_memory_bytes + gradient_memory_bytes + optimizer_memory_bytes + activation_memory_bytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    print(f"Total Estimated Memory: {total_memory_gb} GB")
    return total_memory_gb

# Function to calculate the maximum token length in the dataset
def max_token_len(dataset, tokenizer):
    """
    Calculate the maximum token length of text entries in a dataset using a specified tokenizer.
    """
    max_seq_length = 0
    for row in dataset:
        tokens = len(tokenizer(row['text'])['input_ids'])
        if tokens > max_seq_length:
            max_seq_length = tokens
    return max_seq_length

# Define model and tokenizer
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Define project parameters
username = 'smgreen1'  # Your Hugging Face username
repo_name = 'gen_atc_mistral_7b_v1'  # Fixed repository name for Hugging Face

# Load and process dataset
dataset_name = 'ai-aerospace/ams_data_train_mistral_v0.1_100'
dataset = load_dataset(dataset_name)

# Calculate maximum token length for training and validation data
max_token_length_train = max_token_len(dataset['train'], tokenizer)
max_token_length_val = max_token_len(dataset['validation'], tokenizer)
max_token_length = min(tokenizer.model_max_length, max(max_token_length_train, max_token_length_val))
print(f"Max token length to use: {max_token_length}")

# Define LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)

# Model parameters
model_params = {
    "project_name": './llms/' + repo_name,  # Local project directory
    "model_name": model_name,
    "repo_id": username + '/' + repo_name,  # Repository ID for Hugging Face Hub
    "block_size": 2 * max_token_length,
    "model_max_length": max_token_length,
    "logging_steps": -1,
    "evaluation_strategy": "epoch",
    "save_total_limit": 1,
    "save_strategy": "epoch",
    "mixed_precision": "fp16",
    "lr": 3e-5,
    "epochs": 3,
    "batch_size": 1,
    "warmup_ratio": 0.1,
    "gradient_accumulation": 1,
    "optimizer": "adamw_torch",
    "scheduler": "linear",
    "weight_decay": 0,
    "max_grad_norm": 1,
    "seed": 42,
    "quantization": "int4",
}

# Load the base model and apply LoRA
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
lora_model = get_peft_model(base_model, peft_config)
lora_model.print_trainable_parameters()

# Estimate LoRA memory usage
lora_memory = estimate_lora_memory(base_model, lora_model, model_params['batch_size'], model_params['model_max_length'])
print(f"Estimated LoRA fine-tuning memory: {lora_memory} GB")

# Training arguments
args_custom = TrainingArguments(
    per_device_train_batch_size=model_params['batch_size'],
    per_device_eval_batch_size=model_params['batch_size'],
    gradient_accumulation_steps=model_params['gradient_accumulation'],
    warmup_ratio=model_params['warmup_ratio'],
    num_train_epochs=model_params['epochs'],
    learning_rate=model_params['lr'],
    fp16=True,
    logging_steps=model_params['logging_steps'],
    save_total_limit=model_params['save_total_limit'],
    evaluation_strategy=model_params['evaluation_strategy'],
    metric_for_best_model="f1",
    output_dir=model_params["GEN_ATC_Minstral_7B_V1"],   # Use unique output directory
    logging_dir=model_params["GEN_ATC_Minstral_7B_V1"],  # Set logging directory
    optim=model_params['optimizer'],
    max_grad_norm=model_params['max_grad_norm'],
    weight_decay=model_params['weight_decay'],
    lr_scheduler_type=model_params['scheduler'],
    remove_unused_columns=False,
)

# Initialize the trainer
trainer = SFTTrainer(
    model=base_model,
    peft_config=peft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field='text',
    max_seq_length=model_params['model_max_length'],
    tokenizer=tokenizer,
    args=args_custom,
    packing=False
)

# Start training and track time
start_time = time.perf_counter()
trainer.train()
end_time = time.perf_counter()
print(f'Elapsed time for training: {end_time - start_time} seconds')

# Save the trained model with unique project name
trainer.model.save_pretrained(model_params["GEN_ATC_Minstral_7B_V1"])


# Merge the model with LoRA weights and save
merged_model = PeftModel.from_pretrained(base_model, model_params["GEN_ATC_Minstral_7B_V1"]).merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")

# Push to Hugging Face Hub with specified repository name
merged_model.push_to_hub(f"{username}/{repo_name}")