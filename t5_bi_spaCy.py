import os
import re
import pandas as pd
import numpy as np
import torch
import spacy
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
        logging.FileHandler(f"t5_bi_spaCy.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy models - one for each language
try:
    # For English
    en_nlp = spacy.load("en_core_web_trf")
    # For Khasi - if there's no specific model, use a language-agnostic approach
    # This is a fallback as spaCy might not have a dedicated Khasi model
    kha_nlp = spacy.blank("xx")  # Using multi-language blank model for Khasi
    logger.info("Successfully loaded spaCy models")
except Exception as e:
    logger.warning(f"Error loading spaCy models: {e}")
    logger.warning("Falling back to simpler tokenization approach")
    en_nlp = None
    kha_nlp = None

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration - CHANGE THESE VARIABLES
TSV_FILE_PATH = "samanantar_4950k_filtered.tsv"  
BASE_OUTPUT_DIR = "t5_khasi_english_spaCy"
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

def load_dataset(tsv_file_path):
    logger.info(f"Loading dataset from {tsv_file_path}")
    
    # Check if file exists
    if not os.path.exists(tsv_file_path):
        logger.error(f"File not found: {tsv_file_path}")
        return pd.DataFrame(columns=['source', 'target'])
    
    # Get file size
    file_size = os.path.getsize(tsv_file_path)
    logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
    
    # Try multiple methods to load the data
    sources = []
    targets = []
    
    try:
        # Try to open and read the file
        logger.info("Reading file line by line...")
        with open(tsv_file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read the first few lines to identify the format
            sample_lines = [f.readline() for _ in range(5)]
            f.seek(0)  # Reset file pointer
            
            # Debug sample lines
            logger.info("Sample lines from file:")
            for i, line in enumerate(sample_lines):
                logger.info(f"Line {i+1}: {line.strip()}")
            
            # Skip header line if present
            first_line = f.readline().strip()
            if first_line.lower() in ["en\tkha", "en					kha"]:
                logger.info("Skipping header line")
            else:
                f.seek(0)  # Reset if no header
            
            # Process the whole file
            for line_num, line in enumerate(f, 1):
                try:
                    # Try to parse the line
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split by any number of tabs (handles multiple tabs)
                    parts = re.split(r'\t+', line)
                    
                    # Check if we have at least two parts
                    if len(parts) >= 2:
                        src = parts[0].strip()
                        tgt = parts[1].strip()
                        
                        # Check if both are non-empty
                        if src and tgt:
                            sources.append(src)
                            targets.append(tgt)
                    
                    # Progress logging
                    if line_num % 500000 == 0:
                        logger.info(f"Processed {line_num} lines, found {len(sources)} valid pairs")
                
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {str(e)}")
                    continue
    
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame({'source': sources, 'target': targets})
    
    # Log statistics
    logger.info(f"Total pairs found: {len(df)}")
    
    # Sample data
    if len(df) > 0:
        logger.info("\nSample data:")
        for i in range(min(5, len(df))):
            logger.info(f"Source: {df.iloc[i]['source']}")
            logger.info(f"Target: {df.iloc[i]['target']}")
    
    return df


# IMPROVED: Calculate BLEU score using spaCy instead of NLTK punkt
def calculate_bleu(reference, hypothesis):
    if not hypothesis or not reference:
        return 0.0
    
    # Use spaCy for tokenization
    if IS_KHASI_TO_ENGLISH:
        # Source is Khasi, Target is English
        if en_nlp is not None:
            reference_tokens = [token.text for token in en_nlp(reference.lower())]
            hypothesis_tokens = [token.text for token in en_nlp(hypothesis.lower())]
        else:
            # Fallback for English tokenization if spaCy model isn't loaded
            reference_tokens = reference.lower().split()
            hypothesis_tokens = hypothesis.lower().split()
    else:
        # Source is English, Target is Khasi
        if en_nlp is not None and kha_nlp is not None:
            # Use English spaCy model for English
            reference_tokens = [token.text for token in kha_nlp(reference.lower())]
            hypothesis_tokens = [token.text for token in kha_nlp(hypothesis.lower())]
        else:
            # Fallback for Khasi tokenization - simple whitespace tokenization
            reference_tokens = reference.lower().split()
            hypothesis_tokens = hypothesis.lower().split()
    
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
    
    # Define a custom collate function that handles 'original_source' and 'original_target'
    def custom_collate_fn(batch):
        # Extract 'original_source' and 'original_target' before collation
        original_sources = [item.pop('original_source') for item in batch]
        original_targets = [item.pop('original_target') for item in batch]
        
        # Use DataCollator for the rest of the items
        collated_batch = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=model, 
            padding=True
        )(batch)
        
        # Add back the original texts as lists (not as tensors)
        collated_batch['original_source'] = original_sources
        collated_batch['original_target'] = original_targets
        
        return collated_batch
    
    # Create dataloader with custom collate function
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=custom_collate_fn
    )
    
    # Collect predictions and references
    all_preds = []
    all_refs = []
    all_sources = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info("Generating translations for test set...")
    for batch in test_dataloader:
        # Move batch to device (only tensor elements)
        batch_for_model = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
        
        # Generate translations
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_for_model["input_ids"],
                attention_mask=batch_for_model["attention_mask"],
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
            if batch['original_source'] is not None:
                logger.info(f"Source: {batch['original_source'][i]}")
            logger.info(f"Reference: {refs[i]}")
            logger.info(f"Prediction: {preds[i]}")
            bleu = calculate_bleu(refs[i], preds[i])
            logger.info(f"BLEU: {bleu:.2f}")
            logger.info("---")
        
        all_preds.extend(preds)
        all_refs.extend(refs)
        if batch['original_source'] is not None:
            all_sources.extend(batch['original_source'])
    
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
            if i < len(all_sources):
                f.write(f"Source: {all_sources[i]}\n")
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
def main():
    global tokenizer, full_df, TRAIN_SIZE, TEST_SIZE
    
    try:
        # Create base output directory
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        
        # Load the data
        logger.info(f"Loading data from {TSV_FILE_PATH}...")
        full_df = load_dataset(TSV_FILE_PATH)
        logger.info(f"Loaded {len(full_df)} translation pairs.")
        
        # Check if we have any data
        if len(full_df) == 0:
            logger.error("No valid translation pairs found in the dataset. Please check the file format.")
            
            # Attempt to diagnose the issue
            logger.info("\nAttempting to diagnose the dataset issue...")
            try:
                with open(TSV_FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
                    header = f.readline().strip()
                    logger.info(f"First line of file: {header}")
                    
                    # Check if it's a CSV instead of TSV
                    if ',' in header and '\t' not in header:
                        logger.info("File appears to be comma-separated rather than tab-separated.")
                        logger.info("Please try changing the file format or modifying the separator in the code.")
                    
                    # Check if it might be a binary file
                    if any(ord(c) > 127 for c in header[:20]):
                        logger.info("File appears to contain binary data or non-UTF-8 characters.")
                        logger.info("Please ensure the file is properly encoded in UTF-8.")
                    
                    # Check if it might be empty
                    if not header:
                        logger.info("File appears to be empty.")
            except Exception as e:
                logger.error(f"Error diagnosing file: {str(e)}")
            
            return
        
        # Make sure we have enough data
        if len(full_df) < TRAIN_SIZE + TEST_SIZE:
            logger.warning(f"Warning: Not enough data in the file. Need at least {TRAIN_SIZE + TEST_SIZE} examples, but found {len(full_df)}.")
            
            # Set reasonable defaults based on available data
            total_examples = len(full_df)
            test_ratio = 0.2  # 20% for testing
            TEST_SIZE = min(TEST_SIZE, max(1, int(total_examples * test_ratio)))
            TRAIN_SIZE = max(1, total_examples - TEST_SIZE)  # Rest for training
            
            logger.info(f"Adjusted dataset sizes: TRAIN_SIZE={TRAIN_SIZE}, TEST_SIZE={TEST_SIZE}")
        
        # Split the data
        logger.info(f"Splitting data into training ({TRAIN_SIZE} examples) and testing ({TEST_SIZE} examples) sets...")
        train_df, test_df = train_test_split(full_df, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=42)
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        # Initialize tokenizer and model
        logger.info("Initializing T5 tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        
        # Run iterations
        all_metrics = []
        for iteration in range(1, NUM_ITERATIONS + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting iteration {iteration}/{NUM_ITERATIONS}")
            logger.info(f"{'='*50}")
            
            # Train for this iteration
            model, tokenizer, metrics = train_iteration(iteration, tokenizer, model, train_df, test_df)
            all_metrics.append(metrics)
            
            # Save metrics history
            with open(f"{BASE_OUTPUT_DIR}/all_metrics.txt", "w") as f:
                for i, m in enumerate(all_metrics, 1):
                    f.write(f"Iteration {i} Metrics:\n")
                    for key, value in m.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
        
        # Final testing on test data
        logger.info("\nPerforming final evaluation on test set...")
        test_results = test_final_model(model, tokenizer, test_df=test_df)
        
        # Create a best model symlink to the best iteration
        best_iteration = np.argmax([m.get('bleu', 0) for m in all_metrics]) + 1
        logger.info(f"Best model was from iteration {best_iteration} with BLEU score: {all_metrics[best_iteration-1].get('bleu', 0):.2f}")
        
        best_model_path = f"{BASE_OUTPUT_DIR}/iteration_{best_iteration}"
        best_symlink = f"{BASE_OUTPUT_DIR}/best_model"
        
        # Create symlink on Unix/Linux or copy on Windows
        if os.path.exists(best_symlink):
            if os.path.isdir(best_symlink):
                shutil.rmtree(best_symlink)
            else:
                os.remove(best_symlink)
                
        if hasattr(os, 'symlink'):
            os.symlink(f"iteration_{best_iteration}", best_symlink, target_is_directory=True)
            logger.info(f"Created symlink to best model at {best_symlink}")
        else:
            # For Windows, copy the directory instead
            shutil.copytree(best_model_path, best_symlink)
            logger.info(f"Copied best model to {best_symlink}")
        
        # Final message
        logger.info("\nTraining complete!")
        logger.info(f"Best model saved at: {best_symlink}")
        logger.info(f"Final BLEU score: {test_results['bleu']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
