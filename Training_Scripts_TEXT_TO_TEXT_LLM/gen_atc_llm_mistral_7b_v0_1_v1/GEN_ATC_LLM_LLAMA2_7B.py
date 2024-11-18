# Disable progress bars in transformers and tqdm
import os
os.environ["DISABLE_PROGRESS_BAR"] = "true"

# Imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

# Model and Tokenizer Setup
model_name = "meta-llama/Llama-2-7b-chat-hf"
token = "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"  # Your Hugging Face token

# Configure quantization with BitsAndBytesConfig for 4-bit quantization
quant_config = BitsAndBytesConfig(load_in_4bit=True)

# Load the tokenizer with authentication token
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token compatibility

# Load the model with quantization and token configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,  # Use quantization config
    device_map="auto",                 # Automatically place on available GPUs
    use_auth_token=token               # Pass the token for access
)

# Use a pipeline as a high-level helper for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Function to ask the model a question and measure execution time
def ask_question(pipe, question, max_new_tokens=200):
    start_time = time.time()
    response = pipe(
        question,
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # Lower randomness for deterministic outputs
        top_k=10,         # Restrict to the top 10 tokens at each step
        top_p=0.85,       # Use cumulative probability for token inclusion
        repetition_penalty=1.2  # Penalize repetitive outputs
    )
    end_time = time.time()
    execution_time = end_time - start_time
    return response[0]['generated_text'], execution_time

# Sample Question
question = (
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

full_question = f"{question}\n\nInput:\n{input_text}"

response, exec_time = ask_question(pipe, full_question)
print(f"Question: {question}")
print(f"Input: {input_text}")
print(f"Response: {response}")
print(f"Execution Time: {exec_time:.2f} seconds")

# Capture GPU memory usage after query
try:
    gpu_memory_initial = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Initial GPU memory in GB
    gpu_memory_final = torch.cuda.max_memory_reserved(0) / (1024 ** 3)  # Peak GPU memory in GB
except Exception:
    gpu_memory_initial = gpu_memory_final = 0  # Default to 0 if GPU not in use

# Output memory metrics
if gpu_memory_initial > 0 or gpu_memory_final > 0:
    print(f"GPU Memory Initial: {gpu_memory_initial:.2f} GB")
    print(f"GPU Memory Final: {gpu_memory_final:.2f} GB")
else:
    print("GPU not in use. Falling back to CPU metrics.")
