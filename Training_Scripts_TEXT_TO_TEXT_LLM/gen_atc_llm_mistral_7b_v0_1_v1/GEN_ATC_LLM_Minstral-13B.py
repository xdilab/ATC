import time
from llama_cpp import Llama
import psutil  # For capturing CPU memory usage

# Load the GGUF model using llama_cpp
print("Loading model...")
model_path = "./amethyst-13b-mistral.Q2_K.gguf"  # Ensure this path is correct
llm = Llama(model_path=model_path)
print("Model successfully loaded.")

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

    :param instruction: Task-specific instruction for the model.
    :param input_text: Input text providing necessary context.
    :param output_text: (Optional) Pre-defined output for validation or empty for generation.
    :return: A formatted prompt string.
    """
    return mod_prompt.format(instruction, input_text, output_text)

# Function to generate a response, track execution time, and monitor memory usage
def ask_question(llm, instruction, input_text, max_tokens=200):
    """
    Generate a response and track execution time and memory usage.

    :param llm: The loaded Llama model.
    :param instruction: Instruction for the model.
    :param input_text: Input text providing the necessary context.
    :param max_tokens: Maximum number of tokens to generate.
    :return: Response, execution time, and memory metrics.
    """
    formatted_prompt = format_prompt(instruction, input_text)
    start_time = time.time()

    # Record initial CPU memory usage
    cpu_memory_initial = psutil.virtual_memory().used / (1024 ** 3)  # In GB

    # Generate the response
    response = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=0.3,  # Lower randomness for deterministic outputs
        top_k=10,  # Restrict to the top 10 tokens at each step
        top_p=0.85  # Use cumulative probability for token inclusion
    )
    end_time = time.time()

    # Record final CPU memory usage
    cpu_memory_final = psutil.virtual_memory().used / (1024 ** 3)  # In GB

    # Try capturing GPU memory usage
    gpu_memory_initial = gpu_memory_final = 0  # GGUF models generally don't use GPU

    execution_time = end_time - start_time

    return response["choices"][0]["text"], execution_time, cpu_memory_initial, cpu_memory_final, gpu_memory_initial, gpu_memory_final

# Example instruction and input
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
    llm, instruction, input_text, max_tokens=200
)

# Print results
print("\nResults:")
print(f"Instruction: {instruction}")
print(f"Input: {input_text}")
print(f"Response: {response}")
print(f"Execution Time: {exec_time:.2f} seconds")

# Output memory metrics
print(f"CPU Memory Initial: {cpu_memory_initial:.2f} GB")
print(f"CPU Memory Final: {cpu_memory_final:.2f} GB")
if gpu_memory_initial > 0 or gpu_memory_final > 0:
    print(f"GPU Memory Initial: {gpu_memory_initial:.2f} GB")
    print(f"GPU Memory Final: {gpu_memory_final:.2f} GB")
