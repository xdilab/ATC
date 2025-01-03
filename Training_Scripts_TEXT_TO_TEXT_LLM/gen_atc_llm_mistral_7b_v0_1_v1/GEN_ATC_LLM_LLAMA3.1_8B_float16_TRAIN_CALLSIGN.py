import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import Dataset
from PyPDF2 import PdfReader
import os

# Optimize memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# List of PDF paths
pdf_paths = [
    "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Pilot’s Handbook of Aeronautical Knowledge.pdf",
    "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Communication Transcripts-Redacted.PDF",
    "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/Greensboro SOP_1544056009.pdf",
    "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/g_atc_datasets/GreensboroATC_Standard_Operating_Procedures.pdf"
]

# Load datasets from all PDFs
print("Loading datasets...")
documents = []
for path in pdf_paths:
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                documents.append(text)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue

full_text = "\n".join(documents)

# Split dataset into training (80%) and evaluation (20%)
split_point = int(0.8 * len(full_text))
train_text = full_text[:split_point]
eval_text = full_text[split_point:]

# Tokenizer setup
print("Loading tokenizer...")
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Prepare datasets
print("Tokenizing dataset...")
train_dataset = Dataset.from_dict({"text": [train_text]})
eval_dataset = Dataset.from_dict({"text": [eval_text]})

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=1)

# Load model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training
print("Training model...")
model.train()
gradient_accumulation_steps = 8
for epoch in range(1):  # Single epoch for testing
    print(f"Epoch {epoch + 1}")
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if step % 5 == 0:  # Log every 5 steps
            print(f"Step {step}, Loss: {loss.item()}")

# Save the fine-tuned model
print("Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# NER Inference Example
print("\nGenerating NER output for test input...")
model.eval()
test_input = "runway three four left cleared to land china southern three two five"
airlines = open("airlines.txt").read().split("\n")  # Load IATA airline list

# Format prompt
prompt = f"This is a transcription from one of Greensboro airport frequency towers. Generate and return only the corrected transcription along with the NER for this format:\n\nInput: {test_input}\n\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=300, temperature=0.7, top_k=50, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Post-processing for NER
def format_ner_output(text, airlines):
    words = text.split()
    result = []
    callsign = ""
    command = ""
    value = ""
    for word in words:
        if word.lower() in airlines:
            callsign = word
        elif word.startswith("runway") or word.startswith("taxiway"):
            value = word
        elif word in ["cleared", "taxi", "hold", "land"]:
            command = word
    if callsign and command and value:
        result.append(f"<atc> <CallSign> {callsign} </CallSign> <Command> {command} </Command> <Value> {value} </Value> </atc>")
    return "\n".join(result)

formatted_response = format_ner_output(test_input, airlines)
print("\nGenerated NER Output:")
print(formatted_response)
