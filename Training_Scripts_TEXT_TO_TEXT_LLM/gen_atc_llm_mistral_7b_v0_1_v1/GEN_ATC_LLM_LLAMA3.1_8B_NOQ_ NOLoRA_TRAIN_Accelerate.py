import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator, infer_auto_device_map
import os
from PyPDF2 import PdfReader

# Path to the PDF file
pdf_path = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Pilot’s Handbook of Aeronautical Knowledge.pdf"

# Load the dataset from the PDF
print("Loading dataset...")
reader = PdfReader(pdf_path)
documents = [page.extract_text() for page in reader.pages if page.extract_text()]
full_text = "\n".join(documents)

# Dataset split
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

# Prepare the datasets
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

# Define max memory allocation for each device
max_memory = {
    0: "47GiB",  # Use 47GiB of GPU 0's memory
    "cpu": "48GiB",  # Use up to 48GiB of CPU memory
}

# Infer device map with corrected max_memory
print("Inferring device map...")
device_map = infer_auto_device_map(
    model=AutoModelForCausalLM.from_pretrained(model_name),  # Temporary instance
    max_memory=max_memory,
)

# Load the model with device mapping
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Full precision
    device_map=device_map,  # Use inferred device map
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Prepare the model, dataloaders, and optimizer using Accelerator
model, train_dataloader, eval_dataloader, optimizer = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer
)

# Training loop
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
