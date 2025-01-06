""" 
    This script loads a CSV file of transcription data, removes empty rows,
    and optionally samples the data to limit memory usage.
    It then tokenizes the “Predicted Transcription” as input and the “True Transcription” as the target,
    initializing a BART model for sequence-to-sequence training. After tokenization, 
    it trains the model with minimal logging and no evaluation (to reduce overhead),
    and finally saves the trained model and tokenizer.
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

"""
Loads CSV, drops NaNs, and samples up to 'n_samples' rows
to keep memory usage as low as possible.
"""
def load_data(file_path: str, num_samples=17000):
    df = pd.read_csv(file_path).dropna(subset=["Predicted Transcription", "True Transcription"])
    if len(df) > num_samples:
        df = df.sample(n=num_samples).reset_index(drop=True)
    return df

"""
Tokenizes source (Predicted Transcription) and target (True Transcription) 
using text_target to avoid the deprecated as_target_tokenizer().
"""
def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["Predicted Transcription"],
        text_target=examples["True Transcription"],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )


if __name__ == "__main__":
    csv_path = "llm_evaluation_summary.csv"
    print("Loading data...")
    df = load_data(csv_path, num_samples=17000)
    print(f"Total samples loaded: {len(df)}")

    # Convert to Dataset
    dataset = Dataset.from_pandas(df)
    # train on about 70% of the dataset

    print("Initializing model & tokenizer...")
    model_name = "facebook/bart-large"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Use_cache can be memory-heavy
    model.config.use_cache = False
    # No gradient checkpointing to keep it extremely simple
    model.gradient_checkpointing_disable()

    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        num_proc=1,                # single-process to avoid overhead
        load_from_cache_file=False # no caching to disk
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Minimal Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="my_bart_minimal",
        num_train_epochs=10,                       # low epoch count
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        weight_decay=0.01,

        eval_strategy="no",
        save_strategy="no",
        logging_steps=9999999,                    # effectively no frequent logs
        report_to="none",                         # no W&B or HF logs
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,  # no evaluation, may be where the memory issue was
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting minimal training...")
    trainer.train()
    print("Training complete.")

    # Save final model & tokenizer
    save_dir = "bart_minimal"
    print(f"Saving final model to: {save_dir} ...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Done.")
