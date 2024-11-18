# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import psutil  # For CPU memory usage

# Model and Tokenizer Setup
model_name = "meta-llama/Llama-2-70b-chat-hf"
token = "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"  # Replace with your Hugging Face token

# Load the tokenizer with authentication token
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token compatibility

# Load the model, attempting GPU usage first
print("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},  # Explicitly use GPU 0
        use_auth_token=token
    )
    print("Model successfully loaded on GPU.")
except RuntimeError as e:
    print(f"Error loading model on GPU: {e}")
    print("Falling back to CPU with full precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use full precision for CPU
        device_map={"": "cpu"},     # Place model on CPU
        use_auth_token=token
    )
    print("Model successfully loaded on CPU.")

# Define the prompt format
mod_prompt = """ Below is an instruction that describes a task, paired with an input that provides further context. Write a response that directly completes the task.

### Instruction:

{}


### Input:

{}


### Corrected Transcription:

{}"""

# Format the prompt
def format_prompt(instruction, input_text, output_text=""):
    """
    Format the prompt with the provided instruction, input, and optional output.

    :param instruction: The task-specific instruction for the model.
    :param input_text: Input text providing the necessary context.
    :param output_text: (Optional) Pre-defined output or empty for generation.
    :return: A formatted prompt string.
    """
    return mod_prompt.format(instruction, input_text, output_text)

# Function to track execution time, memory usage, and generate a response
def ask_question(model, tokenizer, instruction, input_text, max_new_tokens=200):
    """
    Generate a response, track execution time, and monitor memory usage.

    :param model: The loaded Hugging Face model.
    :param tokenizer: The Hugging Face tokenizer.
    :param instruction: The instruction for the model.
    :param input_text: Input text providing the necessary context.
    :param max_new_tokens: Maximum number of tokens to generate.
    :return: Response, execution time, and memory metrics.
    """
    formatted_prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Record CPU memory usage before processing
    cpu_memory_initial = psutil.virtual_memory().used / (1024 ** 3)  # In GB

    # Track execution time and generate output
    start_time = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # Lower randomness for deterministic outputs
        top_k=10,         # Restrict to the top 10 tokens at each step
        top_p=0.85,       # Use cumulative probability for token inclusion
        repetition_penalty=1.2  # Penalize repetitive outputs
    )
    end_time = time.time()

    # Record CPU memory usage after processing
    cpu_memory_final = psutil.virtual_memory().used / (1024 ** 3)  # In GB

    # Decode the response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Capture GPU memory usage if available
    try:
        gpu_memory_initial = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Initial GPU memory in GB
        gpu_memory_final = torch.cuda.max_memory_reserved(0) / (1024 ** 3)  # Peak GPU memory in GB
    except Exception:
        gpu_memory_initial = gpu_memory_final = 0  # Default to 0 if GPU not in use

    execution_time = end_time - start_time

    return response_text, execution_time, cpu_memory_initial, cpu_memory_final, gpu_memory_initial, gpu_memory_final

# Example instruction and input for transcription correction
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

# Get response and execution time
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
    print("GPU not in use. Falling back to CPU metrics.")
    print(f"CPU Memory Initial: {cpu_memory_initial:.2f} GB")
    print(f"CPU Memory Final: {cpu_memory_final:.2f} GB")
