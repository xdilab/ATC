from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from PyPDF2 import PdfReader
import torch
import psutil
import time
import os

# Environment Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load Dataset from PDF
print("Loading dataset...")
pdf_path = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Pilot’s Handbook of Aeronautical Knowledge.pdf"
reader = PdfReader(pdf_path)
documents = [page.extract_text() for page in reader.pages if page.extract_text()]
full_text = "\n".join(documents)

split_point = int(0.8 * len(full_text))
train_text, eval_text = full_text[:split_point], full_text[split_point:]

# Tokenization
print("Loading tokenizer...")
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(text):
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
train_dataset = [tokenize_function(train_text)]

# Load Model with LoRA Integration
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Full precision
    device_map="auto",          # Automatically map to all GPUs
    use_cache=False             # Disable caching for gradient checkpointing
)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Setup
print("Setting up training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory Monitoring
gpu_memory_initial = torch.cuda.memory_allocated() / (1024 ** 3)
cpu_memory_initial = psutil.virtual_memory().used / (1024 ** 3)

start_time = time.time()

# Training Loop
print("Training model...")
model.train()
for epoch in range(1):
    for batch in train_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast(dtype=torch.float32):
            outputs = model(**inputs)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Loss: {loss.item()}")

# Free GPU Memory Before Inference
torch.cuda.empty_cache()

# Save Model
print("Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_model_lora")
tokenizer.save_pretrained("./fine_tuned_model_lora")

# Memory Metrics
end_time = time.time()
gpu_memory_final = torch.cuda.memory_allocated() / (1024 ** 3)
peak_gpu_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
cpu_memory_final = psutil.virtual_memory().used / (1024 ** 3)

print(f"\nExecution Time: {end_time - start_time:.2f} seconds")
print(f"GPU Memory Initial: {gpu_memory_initial:.2f} GB")
print(f"GPU Memory Final: {gpu_memory_final:.2f} GB")
print(f"Peak GPU Memory: {peak_gpu_memory:.2f} GB")
print(f"CPU Memory Initial: {cpu_memory_initial:.2f} GB")
print(f"CPU Memory Final: {cpu_memory_final:.2f} GB")

# Inference
print("\nLoading fine-tuned model for inference...")
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model_lora", torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model_lora")

instruction = (
    "This is a transcription from one of Greensboro airport frequency towers. "
    "Generate and return only the corrected transcription. "
    "Do not explain the corrections or provide steps for correcting."
)
input_text = (
    "taxi way delta closed between taxi way kilo and taxi way delto one taxi way kilo klosed "
    "between runway one for and taxi way kilo one taxi way te one betwen taxi way delta and "
    "ils runway five right out of service runway five righ approach fht istm out of service "
    "runway five right precision approach passing t kaber out of servie runway two three righlt "
    "rea tak runway hold short and"
)

prompt = f"{instruction}\n\n{input_text}"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=300, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nModel Response:")
print(response)
