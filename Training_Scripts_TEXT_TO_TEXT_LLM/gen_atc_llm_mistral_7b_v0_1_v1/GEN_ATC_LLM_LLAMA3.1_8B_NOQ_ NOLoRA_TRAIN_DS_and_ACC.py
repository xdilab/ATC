import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import os
from PyPDF2 import PdfReader

# Path to the PDF file
pdf_path = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Pilot’s Handbook of Aeronautical Knowledge.pdf"

# Dataset from the PDF
print("Loading dataset...")
reader = PdfReader(pdf_path)
documents = [page.extract_text() for page in reader.pages if page.extract_text()]
full_text = "\n".join(documents)

# Training and evaluation
split_point = int(0.8 * len(full_text))
train_text, eval_text = full_text[:split_point], full_text[split_point:]

# Mod
model_name = "meta-llama/Llama-3.1-8B"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()  # Add labels explicitly
    return tokens

# Datasets
print("Tokenizing dataset...")
train_dataset = Dataset.from_dict({"text": [train_text]})
eval_dataset = Dataset.from_dict({"text": [eval_text]})

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Remove 'text' key after tokenization and ensure tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Convert to PyTorch DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=1)

# Initialize Accelerator
print("Initializing Accelerator...")
accelerator = Accelerator()

# DeepSpeed configuration definition. This is to also ensure full prec.
ds_config = {
    "fp32": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO optimization stage
        "offload_optimizer": {
            "device": "cpu",  # Offload optimizer states to CPU
            "pin_memory": True
        },
        "allgather_bucket_size": 2e8,
        "reduce_bucket_size": 2e8
    },
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "zero_allow_untested_optimizer": True
}

# Save the DeepSpeed config to a JSON file
import json
ds_config_path = "./ds_config.json"
with open(ds_config_path, "w") as f:
    json.dump(ds_config, f, indent=4)

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Full precision
    use_cache=False,  # Disable caching for gradient checkpointing
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Prepare model, dataloaders, and optimizer with Accelerator and DeepSpeed
model, train_dataloader, eval_dataloader, optimizer = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer
)

# Training
print("Training model...")
model.train()
for epoch in range(1):  # Single epoch
    print(f"Epoch {epoch + 1}")
    for step, batch in enumerate(train_dataloader):
        # Ensure all tensors are on the correct device
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)  # Use Accelerate's `backward`
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0:  # Log every 5 steps
            print(f"Step {step}, Loss: {loss.item()}")

# Save the fine-tuned model
print("Saving fine-tuned model...")
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./fine_tuned_model", save_function=accelerator.save)
tokenizer.save_pretrained("./fine_tuned_model")

print("Training complete!")
