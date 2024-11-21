# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import psutil  # For CPU memory usage

# Model and Tokenizer Setup
model_name = "meta-llama/Llama-3.2-90B-Vision"
token = "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"  # Replace with your Hugging Face token

# Configure quantization with BitsAndBytesConfig for 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computation
    bnb_4bit_use_double_quant=True,   # Enable double quantization
    bnb_4bit_quant_type="nf4"         # Use NormalFloat4 quantization type
)

# Adjust max memory based on your system's GPU and CPU capacity
available_gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)  # In GB
max_memory = {
    "cpu": "200GiB",  # Allow large offload to CPU
    0: f"{available_gpu_memory - 4}GiB",  # Reserve 4GiB of GPU memory for other processes
}

# Load the tokenizer with authentication token
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token compatibility

# Load the model with GPU and CPU offloading
print("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,  # Use quantization config
        torch_dtype=torch.float16,         # Efficient memory usage
        device_map="auto",                 # Automatically place layers on available GPUs
        max_memory=max_memory,             # Allocate memory for GPU and CPU
        token=token                        # Pass authentication token
    )
    print("Model successfully loaded on GPU with CPU offloading.")
except RuntimeError as e:
    print(f"Error loading model on GPU: {e}")
    print("Falling back to full CPU precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,  # Use quantization config
        torch_dtype=torch.float32,         # Full precision for CPU
        device_map={"": "cpu"},            # Force CPU usage
        token=token
    )
    print("Model successfully loaded on CPU.")

# Function to track execution time and memory usage
def ask_question(model, tokenizer, instruction, input_text, max_new_tokens=200):
    """
    Generate a response, track execution time, and monitor memory usage.
    """
    # Format the input
    prompt = f"{instruction}\n\nInput:\n{input_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Record memory usage before processing
    cpu_memory_initial = psutil.virtual_memory().used / (1024 ** 3)  # In GB

    # Generate response and track execution time
    start_time = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # Lower randomness for deterministic outputs
        top_k=10,         # Consider only top 10 tokens
        top_p=0.85,       # Use cumulative probability for token inclusion
        repetition_penalty=1.2  # Penalize repetitive outputs
    )
    end_time = time.time()

    # Record memory usage after processing
    cpu_memory_final = psutil.virtual_memory().used / (1024 ** 3)  # In GB

    # Decode the response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Capture GPU memory usage
    try:
        gpu_memory_initial = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Initial GPU memory in GB
        gpu_memory_final = torch.cuda.max_memory_reserved(0) / (1024 ** 3)  # Peak GPU memory in GB
    except Exception:
        gpu_memory_initial = gpu_memory_final = 0  # Default to 0 if GPU not in use

    execution_time = end_time - start_time

    return response_text, execution_time, cpu_memory_initial, cpu_memory_final, gpu_memory_initial, gpu_memory_final


# Example Usage
instruction = "This is a transcription from one of Greensboro airport frequency towers. "\
              "Generate and return only the corrected transcription. "\
              "Do not explain the corrections or provide steps for correcting."

input_text = "taxi way delta closed between taxi way kilo and taxi way delto one taxi way kilo klosed "\
             "between runway one for and taxi way kilo one taxi way te one betwen taxi way delta and "\
             "ils runway five right out of service runway five righ approach fht istm out of service "\
             "runway five right precision approach passing t kaber out of servie runway two three righlt "\
             "rea tak runway hold short and"

# Generate and display the response
print("Generating response...")
response, exec_time, cpu_memory_initial, cpu_memory_final, gpu_memory_initial, gpu_memory_final = ask_question(
    model, tokenizer, instruction, input_text, max_new_tokens=200
)

# Print results
print("\nResults:")
print(f"Instruction: {instruction}")
print(f"Input: {input_text}")
print(f"Response: {response}")
print(f"Execution Time: {exec_time:.2f} seconds")

# Output memory metrics
if gpu_memory_initial > 0 or gpu_memory_final > 0:
    print(f"GPU Memory Initial: {gpu_memory_initial:.2f} GB")
    print(f"GPU Memory Final: {gpu_memory_final:.2f} GB")
else:
    print("GPU not in use. Returning CPU memory metrics:")
    print(f"CPU Memory Initial: {cpu_memory_initial:.2f} GB")
    print(f"CPU Memory Final: {cpu_memory_final:.2f} GB")
