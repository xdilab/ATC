# Disable progress bars in transformers and tqdm
import os
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import torch
import time

os.environ["DISABLE_PROGRESS_BAR"] = "true"

# Model and Tokenizer Setup
model_name = "ai-aerospace/Mistral-7B-Instruct-v0.1_asm_60e4dc58"
token = "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"  # Replace with your Hugging Face token

# Set up quantization configuration using BitsAndBytesConfig for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Specify 4-bit quantization
)

# Use a compatible tokenizer for LLaMA models
tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set to eos_token

# Load the model with the quantization config and authentication token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,  # Pass BitsAndBytesConfig for quantization
    device_map="auto",  # Automatically place on available GPUs
    use_auth_token=token  # Use token for authentication
)

# Define the prompt format with a specific transcription marker
mod_prompt = """ Below is an instruction that describes a task, paired with an input that provides further context. Write a response that directly completes the task.

### Instruction:

{}


### Input:

{}


### Corrected Transcription:

{}"""

EOS_TOKEN = tokenizer.eos_token


# Format the prompt
def format_prompt(instruction, input_text, output_text=""):
    """
    Format a prompt with the provided instruction, input, and output.

    :param instruction: Instruction for the model
    :param input_text: Contextual input
    :param output_text: Expected output (optional)
    :return: Formatted prompt
    """
    return mod_prompt.format(instruction, input_text, output_text) + EOS_TOKEN


# Function to ask the model a question and measure execution time
def ask_question(model, tokenizer, instruction, input_text, max_new_tokens=50):
    formatted_prompt = format_prompt(instruction, input_text)

    # Encode input text
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate output with adjusted decoding parameters
    start_time = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # Low randomness for deterministic outputs
        top_k=10,  # Considers only the top 10 tokens for each generation step
        top_p=0.85,  # Uses cumulative probability to include most likely tokens
        repetition_penalty=1.2,  # Penalizes repeated phrases
        use_cache=True
    )
    end_time = time.time()

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    execution_time = end_time - start_time

    return response, execution_time


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
response, exec_time = ask_question(model, tokenizer, instruction, input_text, max_new_tokens=100)

# Print results
print(f"Instruction: {instruction}")
print(f"Input: {input_text}")
print(f"Response: {response}")
print(f"Execution Time: {exec_time} seconds")

# Capture GPU memory usage after query
gpu_memory_initial = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Initial GPU memory in GB
gpu_memory_final = torch.cuda.max_memory_reserved(0) / (1024 ** 3)  # Peak GPU memory in GB

# Output memory metrics
print(f"GPU Memory Initial: {gpu_memory_initial} GB")
print(f"GPU Memory Final: {gpu_memory_final} GB")
