import os
import gc
import torch
import pandas as pd
import numpy as np
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.preprocessing import LabelEncoder

# ========= CONFIG ==========
PHI2_MODEL_PATH = "./phi2_anomaly_classifier_v1"
PHI4_MODEL_PATH = "./phi4_atc_response_model_v1"
WAV2VEC2_MODEL_PATH = "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim"
DATASET_PATH = "LLM_ATC_200_Scenarios_Finalized.csv"
ELEVENLABS_API_KEY = "sk_bedea6f1cd436687d9b44231dc297f0bf80b5297962ad91e"
ELEVENLABS_VOICE_ID = "1SM7GgM6IMuvQlz2BwM3"

# ========= SETUP ==========
os.environ["TORCH_ENABLE_FALLBACK_MULTI_TENSOR_REDUCE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()

# Load Dataset
df = pd.read_csv(DATASET_PATH)
df["Is_Anomaly"] = df["Is_Anomaly"].astype(bool)
anomaly_df = df[df["Is_Anomaly"] == True].dropna(subset=["Anomaly_Type"]).reset_index(drop=True)
normal_df = df[df["Is_Anomaly"] == False].reset_index(drop=True)

# Label encoding
label_encoder = LabelEncoder()
anomaly_df["label"] = label_encoder.fit_transform(anomaly_df["Anomaly_Type"])
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Load Models
phi2_tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_PATH)
phi2_model = AutoModelForSequenceClassification.from_pretrained(PHI2_MODEL_PATH).to(device).eval()
phi4_tokenizer = AutoTokenizer.from_pretrained(PHI4_MODEL_PATH)
phi4_model = AutoModelForCausalLM.from_pretrained(
    PHI4_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
).eval()
processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_PATH)
asr_model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_PATH).to(device).eval()

# ========= AUDIO HANDLING ==========
def record_audio(filename="pilot_input.wav", duration=8, fs=16000):
    print("\n Start speaking your Pilot Request...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(" Recording complete.")
    return filename

def transcribe_audio(filename):
    speech, sr = torchaudio.load(filename)
    speech = torchaudio.functional.resample(speech, sr, 16000)
    input_values = processor(speech.squeeze(), return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = asr_model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.decode(pred_ids[0]).strip()

# ========= TEXT-BASED PIPELINE ==========
def select_scenario(anomaly_mode=True):
    if anomaly_mode:
        options = sorted(anomaly_df["Anomaly_Type"].unique())
        print("\nAvailable Scenario Types:")
        for i, s in enumerate(options, 1):
            print(f"{i}. {s}")
        while True:
            try:
                type_choice = int(input("\nSelect Scenario Type (enter number): "))
                if 1 <= type_choice <= len(options):
                    break
                else:
                    print("Invalid selection, try again.")
            except ValueError:
                print("Please enter a number.")
        selected_type = options[type_choice - 1]
        subset = anomaly_df[anomaly_df["Anomaly_Type"] == selected_type]
    else:
        options = sorted(normal_df["Comm_Type"].dropna().unique())
        print("\nAvailable Scenario Types:")
        for i, s in enumerate(options, 1):
            print(f"{i}. {s}")
        while True:
            try:
                type_choice = int(input("\nSelect Scenario Type (enter number): "))
                if 1 <= type_choice <= len(options):
                    break
                else:
                    print("Invalid selection, try again.")
            except ValueError:
                print("Please enter a number.")
        selected_type = options[type_choice - 1]
        subset = normal_df[normal_df["Comm_Type"] == selected_type]

    subset = subset.sample(min(5, len(subset)), random_state=42).reset_index(drop=True)

    print("\nAvailable Scenarios:")
    for i, row in subset.iterrows():
        print(f"{i + 1}. {row['Prompt_Pilot']}")

    while True:
        try:
            scenario_choice = int(input("\nSelect Scenario (enter number): "))
            if 1 <= scenario_choice <= len(subset):
                break
            else:
                print("Invalid scenario selection.")
        except ValueError:
            print("Please enter a number.")

    return subset.iloc[scenario_choice - 1]

def build_input_text(row, custom_pilot=None):
    pilot_text = custom_pilot if custom_pilot else row["Prompt_Pilot"]
    return (
        f"Phase: {row['Flight_Phase']}. "
        f"Description: {row['Flight_Phase_Description']}. "
        f"Pilot: {pilot_text}. "
        f"Weather: {row['Weather_Condition']}. "
        f"Environment: {row['Environment_Light']}."
    )

def detect_anomaly(text):
    inputs = phi2_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = phi2_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return label_encoder.inverse_transform([pred])[0]

def generate_phi4_response(row, anomaly):
    prompt = (
        f"Flight Phase: {row['Flight_Phase']}\n"
        f"Description: {row['Flight_Phase_Description']}\n"
        f"Pilot Request: {row['Prompt_Pilot']}\n"
        f"Weather: {row['Weather_Condition']}\n"
        f"Environment: {row['Environment_Light']}\n"
        f"Anomaly Type: {anomaly}\n"
        f"ATC Response:"
    )
    inputs = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    output = phi4_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,
        num_beams=5,
        length_penalty=0.8,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = phi4_tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("ATC Response:")[-1].strip()

def play_response_with_elevenlabs(text, output_file="atc_response.wav"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.6}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"\n Audio saved to: {output_file}")
    else:
        print("\n TTS Failed:", response.status_code, response.text)

# ========= MAIN ENTRY POINT ==========
if __name__ == "__main__":
    print("\n XAION Digital Twin - Unified Simulation")
    print("\nChoose an input mode:")
    print("1. Select Normal Scenario")
    print("2. Select Anomaly Scenario")
    print("3. Vocal Pilot Prompt ")

    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "1":
        row = select_scenario(anomaly_mode=False)
        full_input = build_input_text(row)
    elif mode == "2":
        row = select_scenario(anomaly_mode=True)
        full_input = build_input_text(row)
    elif mode == "3":
        audio_file = record_audio()
        pilot_input = transcribe_audio(audio_file)
        print(f"\n Transcribed Pilot Input:\n\"{pilot_input}\"")
        row = df.sample(1).iloc[0]
        row["Prompt_Pilot"] = pilot_input
        full_input = build_input_text(row, custom_pilot=pilot_input)
    else:
        print(" Invalid selection.")
        exit(1)

    detected = detect_anomaly(full_input)
    print(f"\n Anomaly Detected: {detected}")
    response = generate_phi4_response(row, detected)
    print("\n Final Response:\n" + response)
    play_response_with_elevenlabs(response)
    print("\n Complete.\n")
