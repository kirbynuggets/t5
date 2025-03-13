import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data locally (one-time operation)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration - CHANGE THESE VARIABLES
TSV_FILE_PATH = "samanantar_4950k_filtered.tsv"  # REPLACE WITH YOUR FILE PATH
OUTPUT_DIR = "t5_khasi_english_model"
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
# Determine if the dataset is Khasi->English or English->Khasi
IS_KHASI_TO_ENGLISH = True  # CHANGE THIS BASED ON YOUR DATASET DIRECTION

# Define markers for language indication
SRC_PREFIX = "translate Khasi to English: " if IS_KHASI_TO_ENGLISH else "translate English to Khasi: "
TGT_PREFIX = "translate English to Khasi: " if IS_KHASI_TO_ENGLISH else "translate Khasi to English: "

# Load the dataset
def load_dataset(tsv_file_path):
    # The file has \t \t \t \t \t as the end marker
    # Assuming the format is source \t target \t \t \t \t \t
    with open(tsv_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sources = []
    targets = []

    for line in lines:
        # Replace the end marker
        line = line.replace("\t \t \t \t \t", "")
        # Split on tab
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            sources.append(parts[0].strip())
            targets.append(parts[1].strip())

    return pd.DataFrame({'source': sources, 'target': targets})

# Calculate BLEU score using NLTK (no external API)
def calculate_bleu(reference, hypothesis):
    smoother = SmoothingFunction().method1

    # Tokenize sentences (split into words)
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())

    # Calculate BLEU score (with smoothing for short sentences)
    try:
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoother) * 100
    except Exception:
        return 0.0  # Return 0 if calculation fails

# Translation Dataset
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length, source_prefix, target_prefix=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix  # For cycle consistency

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data.iloc[idx]['source']
        target = self.data.iloc[idx]['target']

        source_text = self.source_prefix + source

        # Tokenize inputs
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create input_ids and attention_mask
        input_ids = source_encoding.input_ids.squeeze()
        attention_mask = source_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Replace pad tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_source": source,  # Store original for cycle consistency
            "original_target": target   # Store original for cycle consistency
        }

# Compute BLEU Score (no external API)
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU scores for each prediction-reference pair
    bleu_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        bleu_score = calculate_bleu(ref, pred)
        bleu_scores.append(bleu_score)

    # Average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    return {"bleu": avg_bleu}

# Cycle Consistency Loss Calculation
def compute_cycle_loss(model, tokenizer, batch, device):
    # Forward pass: source -> target
    src_texts = [SRC_PREFIX + text for text in batch["original_source"]]
    tgt_texts = batch["original_target"]

    # Source -> Target
    src_encodings = tokenizer(src_texts, padding=True, truncation=True,
                             max_length=MAX_SOURCE_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=src_encodings.input_ids,
            attention_mask=src_encodings.attention_mask,
            max_length=MAX_TARGET_LENGTH
        )

    # Decode the generated targets
    generated_tgts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Target -> Source (cycle back)
    inverse_texts = [TGT_PREFIX + text for text in generated_tgts]
    inverse_encodings = tokenizer(inverse_texts, padding=True, truncation=True,
                                 max_length=MAX_SOURCE_LENGTH, return_tensors="pt").to(device)

    with torch.no_grad():
        cycle_outputs = model.generate(
            input_ids=inverse_encodings.input_ids,
            attention_mask=inverse_encodings.attention_mask,
            max_length=MAX_SOURCE_LENGTH
        )

    # Decode the reconstructed sources
    reconstructed_srcs = tokenizer.batch_decode(cycle_outputs, skip_special_tokens=True)

    # Calculate BLEU between original sources and reconstructed sources
    cycle_scores = []
    for orig_src, recon_src in zip(batch["original_source"], reconstructed_srcs):
        bleu = calculate_bleu(orig_src, recon_src)
        cycle_scores.append(bleu)

    return sum(cycle_scores) / len(cycle_scores) if cycle_scores else 0

# Custom Trainer with Cycle Loss
class CycleConsistencyTrainer(Seq2SeqTrainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None,
                 eval_dataset=None, tokenizer=None, compute_metrics=None, **kwargs):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        self.cycle_losses = []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # Calculate cycle loss on evaluation dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model.to(self.args.device)
        model.eval()

        # Process batches for cycle loss
        cycle_scores = []
        for batch in eval_dataloader:
            # Move batch to device
            batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Extract original text data
            orig_sources = []
            orig_targets = []

            # Try to get original source/target if available in batch
            if "original_source" in batch and "original_target" in batch:
                orig_sources = batch["original_source"]
                orig_targets = batch["original_target"]
            else:
                # Fallback to decoding from input_ids/labels
                orig_sources = [
                    self.tokenizer.decode(ids, skip_special_tokens=True).replace(SRC_PREFIX, "")
                    for ids in batch["input_ids"]
                ]
                orig_targets = [
                    self.tokenizer.decode(ids[ids != -100], skip_special_tokens=True)
                    for ids in batch["labels"]
                ]

            # Prepare batch for cycle loss
            batch_for_cycle = {
                "original_source": orig_sources,
                "original_target": orig_targets
            }

            # Compute cycle loss
            cycle_score = compute_cycle_loss(model, self.tokenizer, batch_for_cycle, self.args.device)
            cycle_scores.append(cycle_score)

        # Compute average cycle score
        avg_cycle_score = sum(cycle_scores) / len(cycle_scores) if cycle_scores else 0
        metrics["cycle_consistency"] = avg_cycle_score
        self.cycle_losses.append(avg_cycle_score)

        return metrics

# Main execution
def main():
    global tokenizer  # Make tokenizer available to compute_metrics function

    # Load the data
    print(f"Loading data from {TSV_FILE_PATH}...")
    df = load_dataset(TSV_FILE_PATH)
    print(f"Loaded {len(df)} translation pairs.")

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")

    # Load tokenizer and model
    print("Loading T5 tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Create datasets
    train_dataset = TranslationDataset(
        train_df, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, SRC_PREFIX
    )
    val_dataset = TranslationDataset(
        val_df, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, SRC_PREFIX
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # Generate a unique run name
    run_name = f"t5-khasi-english-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=run_name,
        eval_strategy="epoch",  # Use the updated parameter name
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        predict_with_generate=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",  # Disable all integrations (wandb, tensorboard, etc.)
        disable_tqdm=False,  # Show progress bars
    )

    # Trainer with Cycle Consistency
    trainer = CycleConsistencyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {metrics}")

    # Save the model and tokenizer
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Plot cycle consistency loss over time
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(trainer.cycle_losses) + 1), trainer.cycle_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Cycle Consistency Score (BLEU)')
        plt.title('Cycle Consistency Over Training')
        plt.savefig(f"{OUTPUT_DIR}/cycle_consistency.png")
        print(f"Cycle consistency plot saved to {OUTPUT_DIR}/cycle_consistency.png")
    except ImportError:
        print("Matplotlib not available, skipping cycle consistency plot.")

    print("Training completed!")

if __name__ == "__main__":
    main()