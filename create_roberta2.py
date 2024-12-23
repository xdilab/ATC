import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch
from itertools import product
import os

class ProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print(f"Step {state.global_step}/{state.max_steps} completed.")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed.")

def add_labels(example):
    true_text = example.get("True Transcription", "").strip() if example.get("True Transcription") else "MISSING"
    predicted_text = example.get("Predicted Transcription", "").strip() if example.get("Predicted Transcription") else "MISSING"
    example["labels"] = int(true_text == predicted_text)
    return example

def tokenize_function(examples, max_length):
    combined_text = [
        f"Predicted: {pred} True: {true}"
        for pred, true in zip(examples["Predicted Transcription"], examples["True Transcription"])
    ]
    return tokenizer(combined_text, truncation=True, padding='max_length', max_length=max_length)

if __name__ == '__main__':
    print(f"GPUs available: {torch.cuda.device_count()}")

    file_path = 'llm_evaluation_summary.csv'
    print("Loading data...")
    data = pd.read_csv(file_path)

    required_columns = ['True Transcription', 'Predicted Transcription']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Drop rows with missing required fields
    data = data.dropna(subset=required_columns)

    print("Splitting data...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    hf_train_data = Dataset.from_pandas(train_data)
    hf_test_data = Dataset.from_pandas(test_data)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    hf_train_data = hf_train_data.map(add_labels)
    hf_test_data = hf_test_data.map(add_labels)

    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    learning_rates = [0.01, 0.00001]
    batch_sizes = [16]
    epochs = [3, 5, 10, 50, 100]
    epochs = [3, 50]
    max_lengths = [512]

    # Directory to save comparison spreadsheets
    os.makedirs("comparison_spreadsheets", exist_ok=True)

    for lr, bs, ep, ml in product(learning_rates, batch_sizes, epochs, max_lengths):
        print(f"Training with lr={lr}, bs={bs}, epochs={ep}, max_length={ml}")

        tokenized_train_data = hf_train_data.map(lambda x: tokenize_function(x, max_length=ml), batched=True, num_proc=1)
        tokenized_test_data = hf_test_data.map(lambda x: tokenize_function(x, max_length=ml), batched=True, num_proc=1)

        tokenized_train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_test_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        training_args = TrainingArguments(
            output_dir=f"./results_lr{lr}_bs{bs}_ep{ep}_ml{ml}",
            eval_strategy="steps",
            eval_steps=5000,
            save_strategy="steps",
            save_steps=5000,
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            num_train_epochs=ep,
            weight_decay=0.01,
            bf16=True,
            logging_dir=f'./logs_lr{lr}_bs{bs}_ep{ep}_ml{ml}',
            dataloader_num_workers=16,
            save_total_limit=2,
            logging_steps=100,
            report_to=None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_test_data,
            callbacks=[ProgressCallback()],
        )

        trainer.train()
        print(f"Training completed for lr={lr}, bs={bs}, epochs={ep}, max_length={ml}")

        # Save the model
        model_dir = f'./corrected-model_lr{lr}_bs{bs}_ep{ep}_ml{ml}'
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model saved to '{model_dir}'")

        # Generate predictions on the test set
        print("Generating predictions on the test set...")
        predictions = trainer.predict(tokenized_test_data)
        pred_labels = predictions.predictions.argmax(axis=1)

        # Convert test_data (pandas) and add predictions
        test_df = test_data.copy().reset_index(drop=True)
        test_df['Model Predicted Label'] = pred_labels
        test_df['True Label'] = test_df.apply(lambda row: 1 if row['True Transcription'].strip() == row['Predicted Transcription'].strip() else 0, axis=1)
        
        # Save comparison to a CSV file
        comparison_csv_path = f"comparison_spreadsheets/comparison_lr{lr}_bs{bs}_ep{ep}_ml{ml}.csv"
        test_df.to_csv(comparison_csv_path, index=False)
        print(f"Comparison spreadsheet saved to '{comparison_csv_path}'")
