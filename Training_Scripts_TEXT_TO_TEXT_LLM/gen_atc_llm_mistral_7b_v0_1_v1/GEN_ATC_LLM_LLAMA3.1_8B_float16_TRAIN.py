from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import GradScaler, autocast
import torch
from PyPDF2 import PdfReader
import psutil
import time
import os

# Environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Specify the local path to the PDF dataset
pdf_path = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Pilot’s Handbook of Aeronautical Knowledge.pdf"

# Load Dataset
print("Loading dataset...")
try:
    reader = PdfReader(pdf_path)
    documents = [page.extract_text() for page in reader.pages]
except Exception as e:
    print(f"Error reading PDF file: {e}")
    exit()

print("Preparing dataset...")
full_text = "\n".join([doc for doc in documents if doc])  # Avoid NoneType results

# Split dataset into training (80%) and evaluation (20%)
split_point = int(0.8 * len(full_text))
train_text = full_text[:split_point]
eval_text = full_text[split_point:]

# Tokenize Dataset
print("Loading tokenizer...")
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(text):
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
train_dataset = [tokenize_function(train_text)]
eval_dataset = [tokenize_function(eval_text)]

# Load Model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Mixed precision
    device_map=None,  # Explicitly map to a single device
    use_cache=False  # Required for gradient checkpointing
)
model = model.to(device)  # Move model explicitly to CUDA
model.gradient_checkpointing_enable()

# Training Setup
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
max_grad_norm = 1.0  # Clip gradients to prevent exploding

# Track memory usage and execution time
gpu_memory_initial = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)
cpu_memory_initial = psutil.virtual_memory().used / (1024 ** 3)
start_time = time.time()

print("Training model...")
model.train()
for epoch in range(1):  # Single epoch
    print(f"Epoch {epoch + 1}")
    for batch in train_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}  # Ensure inputs are on the same device
        optimizer.zero_grad()

        with autocast(dtype=torch.float16):  # Correct usage of autocast
            outputs = model(**inputs)
            loss = outputs.loss

        scaler.scale(loss).backward()  # Scale the loss for FP16
        try:
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Clip gradients
            scaler.step(optimizer)
            scaler.update()
        except ValueError as e:
            print(f"Gradient scaling error: {e}. Skipping this step.")
            continue  # Skip this step if gradients are invalid

        print(f"Loss: {loss.item()}")

end_time = time.time()

# Save fine-tuned model
print("Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Track memory usage after training
gpu_memory_final = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)
peak_gpu_memory = sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)
cpu_memory_final = psutil.virtual_memory().used / (1024 ** 3)
exec_time = end_time - start_time

# Log memory and execution time metrics
print(f"\nExecution Time: {exec_time:.2f} seconds")
print(f"GPU Memory Initial: {gpu_memory_initial:.2f} GB")
print(f"GPU Memory Final: {gpu_memory_final:.2f} GB")
print(f"Peak GPU Memory: {peak_gpu_memory:.2f} GB")
print(f"CPU Memory Initial: {cpu_memory_initial:.2f} GB")
print(f"CPU Memory Final: {cpu_memory_final:.2f} GB")

# Inference
print("\nLoading fine-tuned model for inference...")
model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_model",
    torch_dtype=torch.float16,
    device_map=None
)
model = model.to(device)  # Ensure model is on the same device
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

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

print("Generating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=300,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nModel Response:")
print(response)
