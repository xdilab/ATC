import os
import time
import csv
import numpy as np

import torch
import librosa
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, AutoModelForPreTraining, AutoModelForCTC
import noisereduce as nr
from timeit import default_timer as timer


start_of_execution = 0.0
present_time = 0.0
csv_filename = ""

MODEL_NAME = "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim"
THRESHOLD_AMOUNT = 0.01 # cutoff point for when audio is considered too quiet relative to the highest peak
SILENCE_MINIMUM = 1 #duration of silence to check
CATEGORIES = ["Filename", "Start", "End", "Transcription"]

# loading tokenizer and model
tokenizer = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

"""denoises & cleans up audio the entire audio clip"""
def denoise(sample_rate, data):
    print("...DENOISING...")
    # sets all audio to mono
    if len(data.shape) == 2: data = np.mean(data, axis=1)
    
    reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)
    return reduced_noise

"""given an audio clip, return the translation"""
def transcription(path_name, sample_rate, start, timespan):
    print("...TRANSCRIBING...")
    # tokenize and transcribe the audio segment
    librosa_audio, _ = librosa.load(path_name,
                                    sr=sample_rate,
                                    offset=start/sample_rate,
                                    duration=timespan)
    
    if len(librosa_audio) == 0:
        print("Empty audio segment, skipping transcription.")
        return ""
    
    # Ensure the sampling rate is set correctly for the tokenizer
    tokenizer.sampling_rate = sample_rate
    
    # tokenization & transcription
    librosa_values = tokenizer(librosa_audio, return_tensors="pt").input_values
    logits = model(librosa_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # retrieving, autocorrecting and storing predicted transcription
    text = tokenizer.batch_decode(predicted_ids)[0]
    print("---> ", text, "\n")
    return text

"""given the file, its times, and transcription--add that line to the csv"""
def add_info(info):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(info)

"""checks for silence and extracts the next section with significant audio and where they end"""
def silence_check(data, sample_rate, threshold, start):
    silence_duration = SILENCE_MINIMUM * sample_rate
    index = start
    data_length = len(data)

    # checks if the current section has audio or not based on the section
    def is_silent(segment):
        return np.all(np.abs(segment) < threshold)

    while index < data_length:
        end_index = min(index + silence_duration, data_length)
        if is_silent(data[index:end_index]):
            index += silence_duration
        else:
            start_index = index

            while index < data_length:
                end_index = min(index + silence_duration, data_length)
                if is_silent(data[index:end_index]):
                    break
                index += silence_duration

            print("NEXT SECTION:", time.strftime('%H:%M:%S', time.gmtime(start_index/sample_rate)))
            return data[start_index:index], index
    # if at end of file return nothing and the last index
    return np.array([]), index

"""process of finding audio portions, gathering information, and adding it to the csv"""
def splice_analysis(sample_rate, data, path_name, filename):
    threshold = np.max(data) * THRESHOLD_AMOUNT
    used_positions = []
    
    current_position = 0
    
    while True:
        # get the next portion of audio and the index where that audio ends
        next_section, current_position = silence_check(data, sample_rate, threshold, current_position)
        
        # makes sure there aren't any repeats
        if current_position in used_positions: break
        used_positions.append(current_position)
        
        starting_position = current_position - len(next_section)
        timespan = len(next_section) / sample_rate
        
        # transcribe a the audio
        text = transcription(path_name, sample_rate, starting_position, timespan)
        if text.replace(' ', '') == "": 
            print("\t---! blank text !\n")
            continue
        
        start_formatted = time.strftime('%H:%M:%S', time.gmtime(starting_position/sample_rate))
        end_formatted = time.strftime('%H:%M:%S', time.gmtime(current_position/sample_rate))
        print(start_formatted, "-", end_formatted)
        
        # adds info to csv if it's valid and not empty
        csv_info = [filename, start_formatted, end_formatted, text]
        add_info(csv_info)
        
        present_time = timer()
        print("Elapsed Time:", time.strftime('%H:%M:%S', time.gmtime(present_time-start_of_execution)))
        

"""given a directory, go through each file to be processed if wav file"""
def load_audio(directory: str):
    # iterate over each audio file in the folder
    for filename in os.listdir(directory):
            path_name = os.path.join(directory, filename)
            
            # check if it's a WAV audio file
            if filename.endswith(".wav"):
                print("\n\n\tProcessing file:", filename, '\n')
            try:
                sample_rate, data = wavfile.read(path_name)
                data = denoise(sample_rate, data)
                splice_analysis(sample_rate, data, path_name, filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
            


""" Main Execution """
try:
    # enter the directory of the folder with all the audio samples in the console
    folder = input("Enter directory with audio samples: ").replace('\"','')
    start_of_execution = timer()

    # setting up folder and heading
    folder_name = os.path.basename(folder)
    directory = os.path.dirname(os.path.abspath(__file__))
    csv_filename = f"WAV2TXT - {folder_name}.csv"
    csv_filename = os.path.join(directory, csv_filename)

    with open(csv_filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CATEGORIES)
        writer.writeheader()
        
    print("The CSV file has been created:", csv_filename,"\n")

    # translates all wav files in directory and writes info to csv
    load_audio(folder)
    print("\n\t\tProcessing completed.")
    present_time = timer()
    print("\t\tTOTAL ELAPSED TIME:", time.strftime('%H:%M:%S', time.gmtime(present_time-start_of_execution)))
    
except Exception as e:
    print(f"An error occurred: {e}")
