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
import shutil
import logging
import sys

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"translation_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
BASE_OUTPUT_DIR = "t5_khasi_english_model"
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3  # Epochs per iteration
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
NUM_ITERATIONS = 5  # Number of iterations
TRAIN_SIZE = 10000  # Size of training set
TEST_SIZE = 2000    # Size of test set

# Determine if the dataset is Khasi->English or English->Khasi
IS_KHASI_TO_ENGLISH = True  # CHANGE THIS BASED ON YOUR DATASET DIRECTION

# Define markers for language indication
SRC_PREFIX = "translate Khasi to English: " if IS_KHASI_TO_ENGLISH else "translate English to Khasi: "
TGT_PREFIX = "translate English to Khasi: " if IS_KHASI_TO_ENGLISH else "translate Khasi to English: "

# Load the dataset
def load_dataset(tsv_file_path):
    logger.info(f"Loading dataset from {tsv_file_path}")
    try:
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
        
        logger.info(f"Successfully loaded {len(sources)} translation pairs")
        return pd.DataFrame({'source': sources, 'target': targets})
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

# IMPROVED: Calculate BLEU score using NLTK with better handling for low-resource languages
def calculate_bleu(reference, hypothesis):
    if not hypothesis or not reference:
        return 0.0
    
    # Use character-level tokenization for Khasi if word tokenization is unreliable
    # For English, use standard word tokenization
    if IS_KHASI_TO_ENGLISH:
        # Source is Khasi (possibly use character-level for Khasi)
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    else:
        # Source is English
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Use a smoothing method more suitable for short translations
    smoother = SmoothingFunction().method4
    
    # Custom weights to balance precision across n-grams (can be tuned)
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weight to 1, 2, 3, 4-grams
    
    try:
        return sentence_bleu([reference_tokens], hypothesis_tokens, 
                             weights=weights,
                             smoothing_function=smoother) * 100
    except Exception as e:
        logger.error(f"BLEU calculation error: {e}")
        return 0.0

# IMPROVED: Translation Dataset with better handling of original text
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

# IMPROVED: Compute metrics with debugging output
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate BLEU scores for each prediction-reference pair
    bleu_scores = []
    
    # Print some examples for debugging
    logger.info("\n===== EVALUATION EXAMPLES =====")
    for i in range(min(5, len(decoded_preds))):
        logger.info(f"Reference: {decoded_labels[i]}")
        logger.info(f"Prediction: {decoded_preds[i]}")
        bleu = calculate_bleu(decoded_labels[i], decoded_preds[i])
        logger.info(f"BLEU score: {bleu:.2f}")
        logger.info("---")
        bleu_scores.append(bleu)
    
    # Calculate remaining scores
    for i in range(5, len(decoded_preds)):
        bleu = calculate_bleu(decoded_labels[i], decoded_preds[i])
        bleu_scores.append(bleu)
    
    # Average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    logger.info(f"Average BLEU score: {avg_bleu:.2f}")
    return {"bleu": avg_bleu}

# Cycle Consistency Loss Calculation - Khasi → English → Khasi
def compute_cycle_loss_k2e2k(model, tokenizer, batch, device):
    # Forward pass: source -> target
    src_texts = [SRC_PREFIX + text for text in batch["original_source"]]
    tgt_texts = batch["original_target"]
    
    # Debug: Print sample inputs
    if len(src_texts) > 0:
        logger.info(f"\nCycle K2E2K - Sample source: {src_texts[0]}")
    
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
    
    # Debug: Print sample outputs
    if len(generated_tgts) > 0:
        logger.info(f"Generated target: {generated_tgts[0]}")
    
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
    
    # Debug: Print sample reconstructed sources
    if len(reconstructed_srcs) > 0 and len(batch["original_source"]) > 0:
        logger.info(f"Original source: {batch['original_source'][0]}")
        logger.info(f"Reconstructed source: {reconstructed_srcs[0]}")
    
    # Calculate BLEU between original sources and reconstructed sources
    cycle_scores = []
    for orig_src, recon_src in zip(batch["original_source"], reconstructed_srcs):
        bleu = calculate_bleu(orig_src, recon_src)
        cycle_scores.append(bleu)
    
    avg_score = sum(cycle_scores) / len(cycle_scores) if cycle_scores else 0
    logger.info(f"Average K2E2K cycle score: {avg_score:.2f}")
    return avg_score

# Cycle Consistency Loss Calculation - English → Khasi → English
def compute_cycle_loss_e2k2e(model, tokenizer, batch, device):
    # Forward pass: target -> source (English -> Khasi)
    tgt_texts = [TGT_PREFIX + text for text in batch["original_target"]]
    
    # Debug: Print sample inputs
    if len(tgt_texts) > 0:
        logger.info(f"\nCycle E2K2E - Sample target: {tgt_texts[0]}")
    
    # Target -> Source
    tgt_encodings = tokenizer(tgt_texts, padding=True, truncation=True,
                             max_length=MAX_SOURCE_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tgt_encodings.input_ids,
            attention_mask=tgt_encodings.attention_mask,
            max_length=MAX_TARGET_LENGTH
        )
    
    # Decode the generated sources (Khasi)
    generated_srcs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Debug: Print sample outputs
    if len(generated_srcs) > 0:
        logger.info(f"Generated source: {generated_srcs[0]}")
    
    # Source -> Target (cycle back) (Khasi -> English)
    inverse_texts = [SRC_PREFIX + text for text in generated_srcs]
    inverse_encodings = tokenizer(inverse_texts, padding=True, truncation=True,
                                 max_length=MAX_SOURCE_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        cycle_outputs = model.generate(
            input_ids=inverse_encodings.input_ids,
            attention_mask=inverse_encodings.attention_mask,
            max_length=MAX_TARGET_LENGTH
        )
    
    # Decode the reconstructed targets (English)
    reconstructed_tgts = tokenizer.batch_decode(cycle_outputs, skip_special_tokens=True)
    
    # Debug: Print sample reconstructed targets
    if len(reconstructed_tgts) > 0 and len(batch["original_target"]) > 0:
        logger.info(f"Original target: {batch['original_target'][0]}")
        logger.info(f"Reconstructed target: {reconstructed_tgts[0]}")
    
    # Calculate BLEU between original targets and reconstructed targets
    cycle_scores = []
    for orig_tgt, recon_tgt in zip(batch["original_target"], reconstructed_tgts):
        bleu = calculate_bleu(orig_tgt, recon_tgt)
        cycle_scores.append(bleu)
    
    avg_score = sum(cycle_scores) / len(cycle_scores) if cycle_scores else 0
    logger.info(f"Average E2K2E cycle score: {avg_score:.2f}")
    return avg_score

# Custom Trainer with Bidirectional Cycle Loss and Improved Debugging
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
        self.cycle_losses_k2e2k = []
        self.cycle_losses_e2k2e = []
    
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
    
        # Custom Trainer with Bidirectional Cycle Loss and Improved Debugging
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
        self.cycle_losses_k2e2k = []
        self.cycle_losses_e2k2e = []
    
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
        cycle_scores_k2e2k = []  # Khasi -> English -> Khasi
        cycle_scores_e2k2e = []  # English -> Khasi -> English
        
        # Only process a subset of batches for cycle consistency to save time
        max_batches = 5  # Limit to 5 batches for cycle consistency evaluation
        batch_count = 0
        
        logger.info("\n===== CYCLE CONSISTENCY EVALUATION =====")
        for batch in eval_dataloader:
            # Limit number of batches processed
            batch_count += 1
            if batch_count > max_batches:
                break
                
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
            
            # Compute cycle loss in both directions
            cycle_score_k2e2k = compute_cycle_loss_k2e2k(model, self.tokenizer, batch_for_cycle, self.args.device)
            cycle_scores_k2e2k.append(cycle_score_k2e2k)
            
            # Compute English -> Khasi -> English cycle consistency
            cycle_score_e2k2e = compute_cycle_loss_e2k2e(model, self.tokenizer, batch_for_cycle, self.args.device)
            cycle_scores_e2k2e.append(cycle_score_e2k2e)
        
        # Compute average cycle scores
        avg_cycle_score_k2e2k = sum(cycle_scores_k2e2k) / len(cycle_scores_k2e2k) if cycle_scores_k2e2k else 0
        avg_cycle_score_e2k2e = sum(cycle_scores_e2k2e) / len(cycle_scores_e2k2e) if cycle_scores_e2k2e else 0
        
        # Average of both directions
        avg_cycle_score = (avg_cycle_score_k2e2k + avg_cycle_score_e2k2e) / 2
        
        # Store metrics
        metrics["cycle_k2e2k"] = avg_cycle_score_k2e2k
        metrics["cycle_e2k2e"] = avg_cycle_score_e2k2e
        metrics["cycle_consistency"] = avg_cycle_score
        
        self.cycle_losses_k2e2k.append(avg_cycle_score_k2e2k)
        self.cycle_losses_e2k2e.append(avg_cycle_score_e2k2e)
        
        logger.info(f"\nCycle consistency metrics:")
        logger.info(f"K2E2K: {avg_cycle_score_k2e2k:.2f}")
        logger.info(f"E2K2E: {avg_cycle_score_e2k2e:.2f}")
        logger.info(f"Overall: {avg_cycle_score:.2f}")
        
        return metrics

# Function to train for one iteration
def train_iteration(iteration, tokenizer, model, train_df, test_df):
    # Create output directory for this iteration
    output_dir = f"{BASE_OUTPUT_DIR}/iteration_{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_df, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, SRC_PREFIX
    )
    test_dataset = TranslationDataset(
        test_df, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, SRC_PREFIX
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Generate a unique run name
    run_name = f"t5-khasi-english-iter-{iteration}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
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
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    logger.info(f"Starting training for iteration {iteration}...")
    trainer.train()
    
    # Evaluate the model
    logger.info(f"Evaluating model for iteration {iteration}...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics for iteration {iteration}: {metrics}")
    
    # Save the model and tokenizer
    logger.info(f"Saving model for iteration {iteration} to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Plot cycle consistency loss over time
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        # Khasi → English → Khasi cycle consistency
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(trainer.cycle_losses_k2e2k) + 1), trainer.cycle_losses_k2e2k)
        plt.xlabel('Epoch')
        plt.ylabel('K→E→K Cycle Consistency Score (BLEU)')
        plt.title(f'K→E→K Cycle Consistency - Iteration {iteration}')
        plt.grid(True)
        
        # English → Khasi → English cycle consistency
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(trainer.cycle_losses_e2k2e) + 1), trainer.cycle_losses_e2k2e)
        plt.xlabel('Epoch')
        plt.ylabel('E→K→E Cycle Consistency Score (BLEU)')
        plt.title(f'E→K→E Cycle Consistency - Iteration {iteration}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cycle_consistency.png")
        logger.info(f"Cycle consistency plots saved to {output_dir}/cycle_consistency.png")
    except ImportError:
        logger.warning("Matplotlib not available, skipping cycle consistency plot.")
    
    # Save iteration metrics to a file
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(f"Iteration {iteration} Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Iteration {iteration} completed!")
    return model, tokenizer, metrics

# NEW: Test function for final model evaluation
def test_final_model(model, tokenizer, test_file_path=None, test_df=None):
    logger.info("\n===== FINAL MODEL TESTING =====")
    
    # Use provided test data or create a new test split
    if test_file_path:
        test_df = load_dataset(test_file_path)
        logger.info(f"Loaded test data from {test_file_path}: {len(test_df)} examples")
    elif test_df is None:
        # Sample from the full dataset with a different seed
        test_df = full_df.sample(n=min(TEST_SIZE, len(full_df)), random_state=999)
        logger.info(f"Created new test split with {len(test_df)} examples")
    else:
        logger.info(f"Using provided test split with {len(test_df)} examples")
    
    # Create test dataset
    test_dataset = TranslationDataset(
        test_df, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, SRC_PREFIX
    )
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    )
    
    # Collect predictions and references
    all_preds = []
    all_refs = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info("Generating translations for test set...")
    for batch in test_dataloader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Generate translations
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_TARGET_LENGTH
            )
        
        # Decode predictions
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Get references from labels
        refs = []
        for label_ids in batch["labels"]:
            # Replace -100 with pad token id for decoding
            label_ids = label_ids.cpu().numpy()
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            ref = tokenizer.decode(label_ids, skip_special_tokens=True)
            refs.append(ref)
        
        # Print some examples for debugging
        for i in range(min(3, len(preds))):
            if "original_source" in batch:
                logger.info(f"Source: {batch['original_source'][i]}")
            logger.info(f"Reference: {refs[i]}")
            logger.info(f"Prediction: {preds[i]}")
            bleu = calculate_bleu(refs[i], preds[i])
            logger.info(f"BLEU: {bleu:.2f}")
            logger.info("---")
        
        all_preds.extend(preds)
        all_refs.extend(refs)
    
    # Calculate BLEU and other metrics
    bleu_scores = [calculate_bleu(ref, pred) for ref, pred in zip(all_refs, all_preds)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # Calculate percentage of non-zero BLEU scores
    non_zero_bleu = sum(1 for score in bleu_scores if score > 0)
    non_zero_percent = (non_zero_bleu / len(bleu_scores)) * 100 if bleu_scores else 0
    
    logger.info(f"\nTest Results:")
    logger.info(f"Average BLEU score: {avg_bleu:.2f}")
    logger.info(f"Non-zero BLEU scores: {non_zero_percent:.2f}% ({non_zero_bleu}/{len(bleu_scores)})")
    
    # Save test results
    with open(f"{BASE_OUTPUT_DIR}/test_results.txt", "w") as f:
        f.write(f"Test Results:\n")
        f.write(f"Average BLEU score: {avg_bleu:.2f}\n")
        f.write(f"Non-zero BLEU scores: {non_zero_percent:.2f}% ({non_zero_bleu}/{len(bleu_scores)})\n")
        
        # Save some example translations
        f.write("\nExample Translations:\n")
        for i in range(min(20, len(all_preds))):
            f.write(f"Reference: {all_refs[i]}\n")
            f.write(f"Prediction: {all_preds[i]}\n")
            f.write(f"BLEU: {bleu_scores[i]:.2f}\n")
            f.write("---\n")
    
    return {
        "bleu": avg_bleu,
        "non_zero_percent": non_zero_percent,
        "bleu_scores": bleu_scores
    }

# Main execution
# Main execution
def main():
    global tokenizer, full_df  # Make tokenizer and full_df available to other functions
    
    try:
        # Create base output directory
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
        # Load the data
        logger.info(f"Loading data from {TSV_FILE_PATH}...")
        full_df = load_dataset(TSV_FILE_PATH)
        logger.info(f"Loaded {len(full_df)} translation pairs.")
        
        # Check data quality
        logger.info("\nData Sample:")
        for i in range(min(3, len(full_df))):
            logger.info(f"Source: {full_df.iloc[i]['source']}")
            logger.info(f"Target: {full_df.iloc[i]['target']}")
            logger.info("---")
        
        # Make sure we have enough data
        if len(full_df) < TRAIN_SIZE + TEST_SIZE:
            logger.warning(f"Warning: Not enough data in the file. Need at least {TRAIN_SIZE + TEST_SIZE} examples, but found {len(full_df)}.")
            TRAIN_SIZE = int(len(full_df) * 0.8)  # Use 80% for training
            TEST_SIZE = len(full_df) - TRAIN_SIZE  # Use 20% for testing
            logger.info(f"Adjusted TRAIN_SIZE to {TRAIN_SIZE} and TEST_SIZE to {TEST_SIZE}")
        
        # Load tokenizer and model
        logger.info("Loading T5 tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        
        # Track metrics across iterations
        all_metrics = []
        
        # Create a fixed test set for final evaluation
        final_test_df = full_df.sample(n=TEST_SIZE, random_state=999)
        logger.info(f"Created final test set with {len(final_test_df)} examples")
        
        for iteration in range(1, NUM_ITERATIONS + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting Iteration {iteration} of {NUM_ITERATIONS}")
            logger.info(f"{'='*50}\n")
            
            # Sample new train and test sets for this iteration
            sampled_df = full_df.sample(n=TRAIN_SIZE + TEST_SIZE, random_state=42 + iteration)
            train_df = sampled_df.iloc[:TRAIN_SIZE]
            test_df = sampled_df.iloc[TRAIN_SIZE:]
            logger.info(f"Training set: {len(train_df)} examples")
            logger.info(f"Test set: {len(test_df)} examples")
            
            # Train for this iteration
            model, tokenizer, metrics = train_iteration(iteration, tokenizer, model, train_df, test_df)
            all_metrics.append(metrics)
        
        # Save final model and tokenizer to the base directory
        logger.info(f"Saving final model to {BASE_OUTPUT_DIR}...")
        model.save_pretrained(BASE_OUTPUT_DIR)
        tokenizer.save_pretrained(BASE_OUTPUT_DIR)
        
        # Summarize metrics across all iterations
        logger.info("\nSummary of metrics across all iterations:")
        summary_df = pd.DataFrame(all_metrics)
        logger.info(summary_df.to_string())
        summary_df.to_csv(f"{BASE_OUTPUT_DIR}/all_iterations_metrics.csv", index=False)
        
        # Plot metrics across iterations
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(range(1, NUM_ITERATIONS + 1), summary_df['bleu'], 'b-o', label='BLEU')
            plt.xlabel('Iteration')
            plt.ylabel('BLEU Score')
            plt.title('BLEU Score Across Iterations')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, NUM_ITERATIONS + 1), summary_df['cycle_consistency'], 'r-o', label='Cycle Consistency')
            plt.xlabel('Iteration')
            plt.ylabel('Cycle Consistency Score')
            plt.title('Cycle Consistency Score Across Iterations')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{BASE_OUTPUT_DIR}/metrics_across_iterations.png")
            logger.info(f"Metrics plot saved to {BASE_OUTPUT_DIR}/metrics_across_iterations.png")
        except ImportError:
            logger.warning("Matplotlib not available, skipping metrics plot.")
        
        # Final evaluation on the fixed test set
        logger.info("\nEvaluating final model on the fixed test set...")
        final_metrics = test_final_model(model, tokenizer, test_df=final_test_df)
        
        logger.info("\nTraining and evaluation complete!")
        logger.info(f"Final model saved to {BASE_OUTPUT_DIR}")
        logger.info(f"Final BLEU score: {final_metrics['bleu']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())