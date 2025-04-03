import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset # Alias to avoid conflict
# Updated imports: remove load_metric, add evaluate
from datasets import Dataset as HFDataset
from datasets import DatasetDict
import evaluate # Import the evaluate library
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer, # Use fast tokenizer
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
import numpy as np
import logging
import os
import random
from tqdm.auto import tqdm # Progress bars
import sys
import re

# Optional: Try importing matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- Configuration ---
# >>>>>>>> ADJUST THESE PARAMETERS AS NEEDED <<<<<<<<<<

# --- File/Model Paths ---
DATASET_PATH = "samanantar_4950k_filtered.tsv" # <<< !!! CHANGE THIS TO YOUR ACTUAL FILE PATH !!!
MODEL_NAME = "t5-base"                        # Using t5-base model
OUTPUT_DIR = "t5_spaCy_full" # Directory for models and logs (change for new runs)
LOG_FILE_NAME = "t5_spaCy_full.log"           # Name for the main log file
# Path for saved tokenized data (will be inside OUTPUT_DIR)
TOKENIZED_DATA_PATH = os.path.join(OUTPUT_DIR, "tokenized_data")
# Set to True to ignore saved data and re-tokenize anyway
FORCE_RETOKENIZE = False

# --- Data Size Configuration ---
# Using 1 Million pairs as a strong starting point. Reduce if needed for resources/time.
TOTAL_PAIRS_TO_USE = 1000000 # Total Eng-Kh pairs to load (e.g., 1M out of ~4.95M)
TRAIN_SIZE_FRACTION = 0.95   # Use 95% of the loaded pairs for training (~950k pairs)
VAL_SIZE_FRACTION = 0.025    # Use 2.5% for validation (~25k pairs)
# TEST_SIZE_FRACTION is implicitly 2.5% (~25k pairs)

# --- Training Hyperparameters ---
# These are common starting points, may require tuning based on validation results
LEARNING_RATE = 1e-4           # 0.0001 - AdamW optimizer learning rate
NUM_EPOCHS = 3                 # Number of full passes over the training data
# max_steps = -1               # Set this > 0 instead of NUM_EPOCHS for step-based training

# Batch size / Gradient Accumulation (Adjust based on GPU VRAM)
# Effective Batch Size = N_GPU * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# Aim for effective batch size of 64, 128, or 256 if possible.
PER_DEVICE_TRAIN_BATCH_SIZE = 8   # Batch size per GPU (reduce to 4 or lower if OOM error)
GRADIENT_ACCUMULATION_STEPS = 8   # Accumulate gradients over N steps (increase if reducing batch size)
# (Effective batch size here = 1 * 8 * 8 = 64, assuming 1 GPU)

PER_DEVICE_EVAL_BATCH_SIZE = 16  # Batch size for evaluation (usually can be larger)
WEIGHT_DECAY = 0.01              # Weight decay for regularization
WARMUP_STEPS = 500               # Linear warmup steps for learning rate
FP16 = False # Enable mixed precision if CUDA GPU is available (recommended: speeds up training, saves memory)

# --- Evaluation / Saving / Logging Frequency ---
# Adjust based on total training steps (calculated from data size, batch size, epochs)
# With ~950k pairs * 2 directions / effective batch 64 * 3 epochs = ~89k steps total
EVAL_STEPS = 2000                # Evaluate on validation set every N steps
SAVE_STEPS = 2000                # Save a checkpoint every N steps
LOGGING_STEPS = 200              # Log training loss every N steps

# --- Generation Parameters ---
MAX_SOURCE_LENGTH = 128          # Max tokens for input sequence
MAX_TARGET_LENGTH = 128          # Max tokens for generated output sequence
NUM_BEAMS = 4                    # Number of beams for beam search generation

# --- Cycle Consistency Evaluation ---
CYCLE_EVAL_SAMPLES = 100         # Number of validation samples for cycle consistency check during evaluation

# >>>>>>>> END OF ADJUSTABLE PARAMETERS <<<<<<<<<<

# --- Derived Configuration ---
LOG_FILE = os.path.join(OUTPUT_DIR, LOG_FILE_NAME)
CYCLE_LOG_FILE = os.path.join(OUTPUT_DIR, "cycle_consistency_log.log")

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Setup Logging ---
# (Initialize logging setup as before)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Log to console as well
    ]
)
logger = logging.getLogger(__name__)

# --- Setup Separate Logger for Cycle Consistency Samples ---
# (Initialize cycle_logger as before)
cycle_logger = logging.getLogger("CycleLogger")
cycle_logger.setLevel(logging.INFO)
if not cycle_logger.hasHandlers():
    cycle_file_handler = logging.FileHandler(CYCLE_LOG_FILE)
    cycle_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    cycle_logger.addHandler(cycle_file_handler)


logger.info("=================================================")
logger.info("Starting English-Khasi T5 Training Script")
logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Dataset path: {DATASET_PATH}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Tokenized data path: {TOKENIZED_DATA_PATH}")
logger.info(f"Total pairs to use: {TOTAL_PAIRS_TO_USE}")
logger.info(f"Train fraction: {TRAIN_SIZE_FRACTION}, Val fraction: {VAL_SIZE_FRACTION}")
logger.info(f"FP16 enabled: {FP16}")
logger.info("=================================================")


# === Part 2: Data Loading and Preprocessing ===

def check_file_separator(filepath, sample_lines=5):
    """Check the actual separator used in the file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = []
            for _ in range(sample_lines):
                try:
                    lines.append(next(f))
                except StopIteration:
                    break
        
        if not lines:
            logger.error(f"Could not read sample lines from {filepath}")
            return '\t'  # Default to tab
            
        # Count tabs per line
        tab_counts = [line.count('\t') for line in lines]
        avg_tabs = sum(tab_counts)/len(tab_counts) if tab_counts else 0
        
        logger.info(f"File sample has average of {avg_tabs:.1f} tabs per line")
        
        # Try to determine if it's multiple tabs
        if avg_tabs >= 4.5:  # Likely 5 tabs
            logger.info("Detected 5-tab separator pattern in file")
            return '\t\t\t\t\t'  
        elif avg_tabs >= 2.5:  # Likely 3 tabs
            logger.info("Detected 3-tab separator pattern in file")
            return '\t\t\t'
        elif avg_tabs >= 1.5:  # Likely 2 tabs
            logger.info("Detected 2-tab separator pattern in file")
            return '\t\t'
        else:
            # Default to single tab
            logger.info("Using standard single tab separator")
            return '\t'
    except Exception as e:
        logger.error(f"Error checking file separator: {e}")
        return '\t'  # Default to tab on error

def load_and_preprocess_data(filepath, num_rows=None):
    """Loads data, adds prefixes, handles NAs, and combines directions."""
    logger.info(f"Loading data from {filepath} (max rows: {num_rows})...")
    try:
        # First check what separator the file is actually using
        sep = check_file_separator(filepath)
        
        # Try to read the file using the detected separator
        try:
            df = pd.read_csv(filepath, sep=sep, header=None, names=['english', 'khasi'], 
                             nrows=num_rows, on_bad_lines='skip', quoting=3, engine='python')
        except Exception as e:
            logger.warning(f"Error with detected separator '{sep}': {e}")
            logger.warning("Falling back to standard tab separator")
            df = pd.read_csv(filepath, sep='\t', header=None, names=['english', 'khasi'], 
                             nrows=num_rows, on_bad_lines='skip', quoting=3)
        
        # Check if dataframe loaded successfully
        if df is None or len(df) == 0:
            logger.error("Failed to load any data from file. Check file format.")
            return pd.DataFrame(columns=['input_text', 'target_text', 'original_english', 'original_khasi'])
            
        logger.info(f"Loaded initial dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Check that we have the right columns
        if len(df.columns) != 2:
            logger.warning(f"Expected 2 columns, but got {len(df.columns)}. Attempting to fix...")
            if len(df.columns) == 1 and isinstance(df.iloc[0, 0], str):
                # Try splitting the single column
                sample = df.iloc[0, 0]
                potential_sep = '\t' if '\t' in sample else None
                if potential_sep:
                    logger.info(f"Trying to split single column with separator: '{potential_sep}'")
                    # Create new DF with the split columns
                    new_df = pd.DataFrame()
                    new_df['english'] = df.iloc[:, 0].str.split(potential_sep).str[0]
                    new_df['khasi'] = df.iloc[:, 0].str.split(potential_sep).str[1]
                    df = new_df
                    logger.info(f"Split single column into 2 columns, new shape: {df.shape}")
        
        # Clean up the data
        df.dropna(subset=['english', 'khasi'], inplace=True) # Remove rows with missing data
        df['english'] = df['english'].astype(str).str.strip() # Ensure string type and remove whitespace
        df['khasi'] = df['khasi'].astype(str).str.strip()

        # Filter out potentially empty strings after stripping
        df = df[df['english'].str.len() > 0]
        df = df[df['khasi'].str.len() > 0]

        logger.info(f"Loaded and cleaned {len(df)} pairs.")

        if len(df) == 0:
             logger.error("No valid data loaded. Check file format and content.")
             # Return an empty DF with expected columns for consistency downstream
             return pd.DataFrame(columns=['input_text', 'target_text', 'original_english', 'original_khasi'])

        # Sample a few rows to verify content
        logger.info("Sample data:")
        for i in range(min(3, len(df))):
            logger.info(f"Row {i} - English: '{df.iloc[i, 0][:50]}...', Khasi: '{df.iloc[i, 1][:50]}...'")

        # Add task prefixes required by T5
        df['en_input'] = "translate English to Khasi: " + df['english']
        df['en_target'] = df['khasi']
        df['kh_input'] = "translate Khasi to English: " + df['khasi']
        df['kh_target'] = df['english']

        # Combine both directions for training, keep originals for cycle consistency later
        en_data = df[['en_input', 'en_target', 'english', 'khasi']].rename(
            columns={'en_input': 'input_text', 'en_target': 'target_text', 'english': 'original_english', 'khasi': 'original_khasi'}
        )
        kh_data = df[['kh_input', 'kh_target', 'english', 'khasi']].rename(
            columns={'kh_input': 'input_text', 'kh_target': 'target_text', 'english': 'original_english', 'khasi': 'original_khasi'}
        )

        combined_df = pd.concat([en_data, kh_data], ignore_index=True)
        logger.info(f"Combined data for both translation directions: {len(combined_df)} total samples.")
        return combined_df

    except FileNotFoundError:
        logger.error(f"Error: Dataset file not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}", exc_info=True)
        raise

# Load the subset of data
try:
    full_data_df = load_and_preprocess_data(DATASET_PATH, num_rows=TOTAL_PAIRS_TO_USE)
except Exception as e:
    logger.error(f"Fatal error during data loading: {e}")
    sys.exit("Exiting due to data loading failure.")

if len(full_data_df) == 0:
    logger.error("Exiting script because no data could be loaded.")
    sys.exit("Exiting due to data loading failure.")

# Verify data integrity
required_cols = ['input_text', 'target_text', 'original_english', 'original_khasi']
missing_cols = [col for col in required_cols if col not in full_data_df.columns]
if missing_cols:
    logger.error(f"Missing required columns in preprocessed data: {missing_cols}")
    sys.exit("Data preprocessing failed - required columns missing")

# Convert pandas DataFrame to Hugging Face Dataset
# Keep original text columns initially for the callback and tokenization function
try:
    hf_dataset = HFDataset.from_pandas(full_data_df)
    logger.info(f"Successfully created HuggingFace Dataset with {len(hf_dataset)} examples")
except Exception as e:
    logger.error(f"Error converting DataFrame to HF Dataset: {e}")
    sys.exit("Failed to create HuggingFace Dataset")

# Shuffle and split the dataset (do this *before* tokenization)
hf_dataset = hf_dataset.shuffle(seed=42)

# Calculate split sizes based on fractions
n_total = len(hf_dataset)
n_val = int(n_total * VAL_SIZE_FRACTION)
n_test = int(n_total * (1.0 - TRAIN_SIZE_FRACTION - VAL_SIZE_FRACTION))
n_train = n_total - n_val - n_test

if n_train <= 0 or n_val <= 0 or n_test <= 0:
     raise ValueError(f"Calculated split sizes are invalid (Train: {n_train}, Val: {n_val}, Test: {n_test}). Check fractions or TOTAL_PAIRS_TO_USE.")

logger.info(f"Calculated split sizes: Train={n_train}, Validation={n_val}, Test={n_test}")

# Perform the split
try:
    train_val_split = hf_dataset.train_test_split(test_size=(n_val + n_test) / n_total, seed=42)
    # Adjust test_size calculation for the second split
    val_test_split = train_val_split['test'].train_test_split(test_size=n_test / (n_val + n_test) if (n_val + n_test) > 0 else 0, seed=42)

    # This 'raw_datasets' contains the text data before tokenization
    raw_datasets = DatasetDict({
        'train': train_val_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })
except Exception as e:
    logger.error(f"Error splitting dataset: {e}")
    sys.exit("Failed to split dataset")

logger.info(f"Raw dataset splits created (before tokenization):")
logger.info(f"  Train: {len(raw_datasets['train'])} samples")
logger.info(f"  Validation: {len(raw_datasets['validation'])} samples")
logger.info(f"  Test: {len(raw_datasets['test'])} samples")

# Verify each split has required columns
for split_name, dataset in raw_datasets.items():
    missing_cols = [col for col in required_cols if col not in dataset.column_names]
    if missing_cols:
        logger.error(f"Missing columns in {split_name} split: {missing_cols}")
        sys.exit(f"Data integrity error in {split_name} split")

# Display a sample from raw data
logger.info("\nSample data point from raw validation set:")
if len(raw_datasets['validation']) > 0:
    logger.info(raw_datasets['validation'][0])
else:
    logger.warning("Raw validation set is empty.")


# === Part 3: Tokenization (with Save/Load Logic) ===

# Initialize tokenizer (needed for both tokenizing and loading)
logger.info(f"Loading tokenizer for {MODEL_NAME}...")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    logger.info(f"Tokenizer loaded successfully with vocab size: {tokenizer.vocab_size}")
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    sys.exit("Failed to load tokenizer")

# Check if tokenized data already exists
if os.path.exists(TOKENIZED_DATA_PATH) and not FORCE_RETOKENIZE:
    logger.info(f"Attempting to load tokenized dataset from disk: {TOKENIZED_DATA_PATH}")
    try:
        tokenized_datasets = DatasetDict.load_from_disk(TOKENIZED_DATA_PATH)
        logger.info("Successfully loaded tokenized data from disk.")
        # Optional: Verify columns if needed
        logger.info(f"Columns in loaded tokenized dataset: {tokenized_datasets['train'].column_names}")
        if 'input_ids' not in tokenized_datasets['train'].column_names or \
           'labels' not in tokenized_datasets['train'].column_names:
            logger.error("Loaded tokenized data seems incomplete (missing 'input_ids' or 'labels'). Forcing re-tokenization.")
            tokenized_datasets = None # Force re-tokenization
        elif len(tokenized_datasets['train']) != n_train:
             logger.warning(f"Loaded tokenized train set size ({len(tokenized_datasets['train'])}) differs from expected ({n_train}). Might be from a previous run with different settings. Consider setting FORCE_RETOKENIZE=True if issues arise.")
             # Continue with loaded data for now

    except Exception as e:
        logger.warning(f"Failed to load tokenized data from {TOKENIZED_DATA_PATH}: {e}")
        logger.warning("Will re-tokenize the data.")
        tokenized_datasets = None # Ensure we proceed to tokenization
else:
    if FORCE_RETOKENIZE:
        logger.info(f"FORCE_RETOKENIZE is True. Re-tokenizing data.")
    else:
        logger.info(f"Tokenized dataset not found at {TOKENIZED_DATA_PATH}. Starting tokenization.")
    tokenized_datasets = None # Signal that tokenization needs to be done

# If dataset wasn't loaded, perform tokenization
if tokenized_datasets is None:
    logger.info("Starting dataset tokenization...")

    # Log tokenizer details only when actually tokenizing
    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    logger.info(f"Tokenizer PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    logger.info(f"Tokenizer UNK token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")

    def tokenize_function(examples):
        """Tokenize the input and target texts."""
        try:
            # Verify inputs exist
            if "input_text" not in examples or "target_text" not in examples:
                missing_cols = []
                if "input_text" not in examples: missing_cols.append("input_text")
                if "target_text" not in examples: missing_cols.append("target_text")
                logger.error(f"Missing columns in tokenize_function: {missing_cols}")
                return {}
            
            # Tokenize inputs (already prefixed)
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                # Padding handled by DataCollator
            )
            # Tokenize targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["target_text"],
                    max_length=MAX_TARGET_LENGTH,
                    truncation=True,
                    # Padding handled by DataCollator
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        except Exception as e:
            logger.error(f"Error in tokenize_function: {e}")
            return {}  # Return empty dict on error

    # Apply tokenization using .map() on the raw_datasets
    # num_proc > 1 can speed this up significantly but uses more memory
    num_cpus = os.cpu_count()
    num_proc_to_use = max(1, min(num_cpus // 2, 8)) # Use half CPUs up to 8, ensure at least 1
    logger.info(f"Using {num_proc_to_use} processes for tokenization map function.")

    try:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc_to_use,
            # Remove original text columns after tokenization to save space
            remove_columns=['input_text', 'target_text', 'original_english', 'original_khasi']
        )
        logger.info("Tokenization complete.")
    except Exception as e:
        logger.error(f"Error during tokenization mapping: {e}")
        sys.exit("Tokenization failed")

    # Save the tokenized dataset
    logger.info(f"Saving tokenized dataset to disk: {TOKENIZED_DATA_PATH}")
    try:
        tokenized_datasets.save_to_disk(TOKENIZED_DATA_PATH)
        logger.info("Tokenized dataset saved successfully.")
    except Exception as e:
        logger.error(f"Error saving tokenized dataset to {TOKENIZED_DATA_PATH}: {e}", exc_info=True)
        logger.warning("Proceeding without saved tokenized data for this run.")


# --- Post-Tokenization Sanity Check ---
# Ensure datasets were either loaded or created
if tokenized_datasets is None:
    logger.error("Tokenized datasets are still None after tokenization attempt. Exiting.")
    sys.exit("Failed to create or load tokenized data.")

# Check if all required splits exist and have data
for split_name in ['train', 'validation', 'test']:
    if split_name not in tokenized_datasets:
        logger.error(f"Missing '{split_name}' split in tokenized datasets")
        sys.exit(f"Missing required '{split_name}' split")
    if len(tokenized_datasets[split_name]) == 0:
        logger.error(f"Tokenized '{split_name}' split is empty")
        sys.exit(f"Empty tokenized '{split_name}' split")

logger.info(f"Columns in final tokenized dataset for training: {tokenized_datasets['train'].column_names}")
logger.info("\nSample tokenized data point (from loaded or newly tokenized):")
if len(tokenized_datasets['train']) > 0:
    sample = tokenized_datasets['train'][0]
    logger.info(f"Input IDs: {sample['input_ids'][:20]}...")
    logger.info(f"Decoded Input: {tokenizer.decode(sample['input_ids'], skip_special_tokens=False)}")
    logger.info(f"Label IDs: {sample['labels'][:20]}...")
    label_ids_for_decode = [l if l != -100 else tokenizer.pad_token_id for l in sample['labels']]
    logger.info(f"Decoded Labels: {tokenizer.decode(label_ids_for_decode, skip_special_tokens=False)}")
else:
    logger.warning("Training dataset is empty.")


# === Part 4: Model Initialization and Metrics ===

# Load the pre-trained T5 model
logger.info(f"Loading model: {MODEL_NAME}")
try:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    logger.info(f"Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sys.exit("Failed to load model")

# Load BLEU metric using evaluate (preferred)
logger.info("Loading SacreBLEU metric using 'evaluate' library...")
try:
    bleu_metric = evaluate.load("sacrebleu")
    logger.info("Successfully loaded SacreBLEU via 'evaluate'.")
except Exception as e:
    logger.error(f"Failed to load SacreBLEU using 'evaluate': {e}", exc_info=True)
    logger.error("Please ensure 'evaluate' and 'sacrebleu' libraries are installed: pip install evaluate sacrebleu")
    raise # Re-raise the exception as it's critical


def compute_metrics(eval_pred):
    """Computes BLEU score for evaluation using SacreBLEU."""
    try:
        predictions, labels = eval_pred

        # Process predictions (usually IDs from trainer)
        if isinstance(predictions, tuple):
            preds = predictions[0] # Often the case with beam search
        else:
            preds = predictions

        # Ensure preds is usable (convert if needed, handle potential errors)
        if not isinstance(preds, np.ndarray):
            try:
                # If it's a tensor, move to CPU and convert
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy()
                else:
                    preds = np.array(preds) # Try direct conversion
            except Exception as e_np:
                logger.error(f"Could not convert predictions to numpy array: {e_np}")
                preds = None # Flag as unusable

        if preds is None:
            logger.error("Predictions unusable in compute_metrics. Returning 0 BLEU.")
            return {"bleu": 0.0, "non_zero_bleu_percent": 0.0, "gen_len": 0.0}

        # Replace -100 in labels with pad_token_id for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple post-processing: remove leading/trailing whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        # SacreBLEU expects labels to be lists of strings (for multiple references)
        decoded_labels_for_bleu = [[label.strip()] for label in decoded_labels]

        # Compute BLEU score
        try:
            result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_for_bleu)
        except Exception as e_bleu:
            logger.error(f"Error computing SacreBLEU: {e_bleu}")
            result = {"score": 0.0} # Default empty result on error

        # Extract BLEU score and calculate non-zero percentage
        bleu_score = result.get("score", 0.0) # Use .get for safety
        try:
            non_zero_bleu_count = sum(1 for p, l in zip(decoded_preds, decoded_labels_for_bleu)
                                    if bleu_metric.compute(predictions=[p], references=[l]).get("score", 0.0) > 0.0)
            non_zero_bleu_percent = (non_zero_bleu_count / len(decoded_preds)) * 100 if len(decoded_preds) > 0 else 0
        except Exception as e_nonzero:
            logger.error(f"Error calculating non-zero BLEU%: {e_nonzero}")
            non_zero_bleu_percent = 0.0

        result["non_zero_bleu_percent"] = non_zero_bleu_percent

        # Add generation length metric
        prediction_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        result["gen_len"] = np.mean(prediction_lens) if prediction_lens else 0.0

        # Log a few examples during evaluation
        logger.info("\n--- Sample Predictions (Eval) ---")
        for i in range(min(3, len(decoded_preds))):
            logger.info(f"Label:      {decoded_labels[i]}")
            logger.info(f"Prediction: {decoded_preds[i]}")
            logger.info("-" * 10)

        # Return metrics, rounded
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}
    except Exception as e:
        logger.error(f"Unhandled error in compute_metrics: {e}")
        return {"bleu": 0.0, "non_zero_bleu_percent": 0.0, "gen_len": 0.0}


# === Part 5: Training Arguments and Data Collator ===

# Data collator for padding sequences dynamically
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100, # Crucial: Use -100 for labels to ignore padding in loss
    pad_to_multiple_of=8 if FP16 else None # Optimize for tensor cores if using fp16
)

# Define Training Arguments
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    # Training Strategy
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    # Evaluation Strategy
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=NUM_BEAMS,
    # Logging & Saving Strategy
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,  # Only keep the 3 most recent checkpoints
    # Other Settings
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,  # Higher BLEU is better
    # Report results
    report_to="tensorboard",
    remove_unused_columns=True,
    push_to_hub=False,  # Set to True if you want to upload to HF Hub
)

logger.info(f"Training arguments configured: {args}")


# === Part 6: Custom Callbacks for Cycle Consistency Evaluation ===

class CycleConsistencyCallback(TrainerCallback):
    """Custom callback to evaluate back-translation cycle consistency during training."""
    
    def __init__(self, tokenizer, raw_validation_data, num_samples=CYCLE_EVAL_SAMPLES):
        self.tokenizer = tokenizer
        # Get validation data with original English and Khasi texts
        self.validation_data = raw_validation_data
        self.num_samples = min(num_samples, len(self.validation_data))
        logger.info(f"CycleConsistencyCallback initialized with {self.num_samples} samples")
        
    def get_cycle_consistency_samples(self):
        """Get random samples from validation data for cycle consistency eval."""
        if len(self.validation_data) == 0:
            logger.warning("No validation data available for cycle consistency evaluation")
            return []
            
        indices = random.sample(range(len(self.validation_data)), self.num_samples)
        return [self.validation_data[i] for i in indices]
        
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run cycle consistency evaluation on checkpoint."""
        logger.info("Running cycle consistency evaluation...")
        
        if model is None:
            logger.error("No model provided to CycleConsistencyCallback")
            return
            
        # Get samples for evaluation
        samples = self.get_cycle_consistency_samples()
        if not samples:
            logger.warning("No samples available for cycle consistency evaluation")
            return
            
        # Track metrics
        exact_match_en2kh2en = 0
        exact_match_kh2en2kh = 0
        bleu_scores_en2kh2en = []
        bleu_scores_kh2en2kh = []
        
        cycle_logger.info(f"\n=== Cycle Consistency at Step {state.global_step} ===")
        
        # Process each sample through the cycle
        for sample_idx, sample in enumerate(tqdm(samples[:10], desc="Cycle consistency")):
            original_en = sample["original_english"]
            original_kh = sample["original_khasi"]
            
            # English -> Khasi -> English cycle
            try:
                # Step 1: En -> Kh
                en_input = f"translate English to Khasi: {original_en}"
                en_input_ids = self.tokenizer(en_input, return_tensors="pt", 
                                           max_length=MAX_SOURCE_LENGTH, 
                                           truncation=True).input_ids.to(model.device)
                                           
                with torch.no_grad():
                    en2kh_output = model.generate(
                        en_input_ids,
                        max_length=MAX_TARGET_LENGTH,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
                
                predicted_kh = self.tokenizer.decode(en2kh_output[0], skip_special_tokens=True)
                
                # Step 2: Predicted Kh -> En
                kh_input = f"translate Khasi to English: {predicted_kh}"
                kh_input_ids = self.tokenizer(kh_input, return_tensors="pt", 
                                           max_length=MAX_SOURCE_LENGTH, 
                                           truncation=True).input_ids.to(model.device)
                                           
                with torch.no_grad():
                    kh2en_output = model.generate(
                        kh_input_ids,
                        max_length=MAX_TARGET_LENGTH,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
                    
                predicted_en_cycle = self.tokenizer.decode(kh2en_output[0], skip_special_tokens=True)
                
                # Calculate metrics
                if original_en.strip().lower() == predicted_en_cycle.strip().lower():
                    exact_match_en2kh2en += 1
                
                # Calculate BLEU for En->Kh->En
                en2kh2en_bleu = bleu_metric.compute(
                    predictions=[predicted_en_cycle], 
                    references=[[original_en]]
                )["score"]
                bleu_scores_en2kh2en.append(en2kh2en_bleu)
                
                # Khasi -> English -> Khasi cycle
                # Step 1: Kh -> En
                kh_input = f"translate Khasi to English: {original_kh}"
                kh_input_ids = self.tokenizer(kh_input, return_tensors="pt", 
                                           max_length=MAX_SOURCE_LENGTH, 
                                           truncation=True).input_ids.to(model.device)
                                           
                with torch.no_grad():
                    kh2en_output = model.generate(
                        kh_input_ids,
                        max_length=MAX_TARGET_LENGTH,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
                    
                predicted_en = self.tokenizer.decode(kh2en_output[0], skip_special_tokens=True)
                
                # Step 2: Predicted En -> Kh
                en_input = f"translate English to Khasi: {predicted_en}"
                en_input_ids = self.tokenizer(en_input, return_tensors="pt", 
                                           max_length=MAX_SOURCE_LENGTH, 
                                           truncation=True).input_ids.to(model.device)
                                           
                with torch.no_grad():
                    en2kh_output = model.generate(
                        en_input_ids,
                        max_length=MAX_TARGET_LENGTH,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
                    
                predicted_kh_cycle = self.tokenizer.decode(en2kh_output[0], skip_special_tokens=True)
                
                # Calculate metrics
                if original_kh.strip().lower() == predicted_kh_cycle.strip().lower():
                    exact_match_kh2en2kh += 1
                    
                # Calculate BLEU for Kh->En->Kh
                kh2en2kh_bleu = bleu_metric.compute(
                    predictions=[predicted_kh_cycle], 
                    references=[[original_kh]]
                )["score"]
                bleu_scores_kh2en2kh.append(kh2en2kh_bleu)
                
                # Log first 10 samples 
                if sample_idx < 10:
                    cycle_logger.info(f"\nSample {sample_idx + 1}:")
                    cycle_logger.info(f"Original English: {original_en}")
                    cycle_logger.info(f"Predicted Khasi: {predicted_kh}")
                    cycle_logger.info(f"Cycle English: {predicted_en_cycle}")
                    cycle_logger.info(f"En->Kh->En BLEU: {en2kh2en_bleu}")
                    cycle_logger.info("-" * 50)
                    cycle_logger.info(f"Original Khasi: {original_kh}")
                    cycle_logger.info(f"Predicted English: {predicted_en}")
                    cycle_logger.info(f"Cycle Khasi: {predicted_kh_cycle}")
                    cycle_logger.info(f"Kh->En->Kh BLEU: {kh2en2kh_bleu}")
                    cycle_logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Error in cycle consistency evaluation for sample {sample_idx}: {e}")
                # Continue with next sample
        
        # Calculate and log overall metrics
        en2kh2en_exact_pct = (exact_match_en2kh2en / len(samples)) * 100 if samples else 0
        kh2en2kh_exact_pct = (exact_match_kh2en2kh / len(samples)) * 100 if samples else 0
        
        avg_en2kh2en_bleu = sum(bleu_scores_en2kh2en) / len(bleu_scores_en2kh2en) if bleu_scores_en2kh2en else 0
        avg_kh2en2kh_bleu = sum(bleu_scores_kh2en2kh) / len(bleu_scores_kh2en2kh) if bleu_scores_kh2en2kh else 0
        
        # Log results
        metrics_msg = (
            f"\n====== Cycle Consistency Results (Step {state.global_step}) =======\n"
            f"Samples evaluated: {len(samples)}\n"
            f"En->Kh->En Exact match: {exact_match_en2kh2en}/{len(samples)} ({en2kh2en_exact_pct:.2f}%)\n"
            f"En->Kh->En Average BLEU: {avg_en2kh2en_bleu:.2f}\n"
            f"Kh->En->Kh Exact match: {exact_match_kh2en2kh}/{len(samples)} ({kh2en2kh_exact_pct:.2f}%)\n"
            f"Kh->En->Kh Average BLEU: {avg_kh2en2kh_bleu:.2f}\n"
            f"=================================================="
        )
        
        logger.info(metrics_msg)
        cycle_logger.info(metrics_msg)
        
        # Add to metrics
        state.log_history[-1]["cycle_en2kh2en_exact_pct"] = en2kh2en_exact_pct
        state.log_history[-1]["cycle_kh2en2kh_exact_pct"] = kh2en2kh_exact_pct
        state.log_history[-1]["cycle_en2kh2en_bleu"] = avg_en2kh2en_bleu
        state.log_history[-1]["cycle_kh2en2kh_bleu"] = avg_kh2en2kh_bleu


# === Part 7: Initialize and Run Training ===

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[CycleConsistencyCallback(tokenizer, raw_datasets["validation"])]
)

# Display training info
logger.info("=================================================")
logger.info("Starting training with configuration:")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Training set size: {len(tokenized_datasets['train'])} examples")
logger.info(f"Validation set size: {len(tokenized_datasets['validation'])} examples")
logger.info(f"Test set size: {len(tokenized_datasets['test'])} examples")
logger.info(f"Batch size per device: {PER_DEVICE_TRAIN_BATCH_SIZE}")
logger.info(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info(f"Number of epochs: {NUM_EPOCHS}")
logger.info(f"Max source length: {MAX_SOURCE_LENGTH}")
logger.info(f"Max target length: {MAX_TARGET_LENGTH}")
logger.info("=================================================")

# Initialize TensorBoard if available
try:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "runs"))
    logger.info(f"TensorBoard logs will be saved to {os.path.join(OUTPUT_DIR, 'runs')}")
except ImportError:
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")
    tb_writer = None

# Optional: Plot a sample before training
if MATPLOTLIB_AVAILABLE:
    try:
        plt.figure(figsize=(10, 6))
        plt.hist([len(x) for x in tokenized_datasets["train"]["input_ids"]], bins=50)
        plt.title("Input sequence lengths")
        plt.xlabel("Length")
        plt.ylabel("Count")
        plt.savefig(os.path.join(OUTPUT_DIR, "input_length_hist.png"))
        logger.info(f"Saved input length histogram to {os.path.join(OUTPUT_DIR, 'input_length_hist.png')}")
    except Exception as e:
        logger.warning(f"Failed to generate input length histogram: {e}")

# Start training
logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training completed successfully!")
except Exception as e:
    logger.error(f"Error during training: {e}", exc_info=True)
    sys.exit("Training failed")

# Save the final model
logger.info("Saving final model...")
try:
    model_save_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Final model saved to {model_save_path}")
except Exception as e:
    logger.error(f"Error saving final model: {e}")

# === Part 8: Evaluation on Test Set ===

logger.info("\n=== Final Evaluation on Test Set ===")
try:
    test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    logger.info(f"Test Results: {test_results}")
    
    # Save test results to file
    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key} = {value}\n")
    logger.info(f"Test results saved to {os.path.join(OUTPUT_DIR, 'test_results.txt')}")
except Exception as e:
    logger.error(f"Error during test evaluation: {e}")

# === Part 9: Generate Translation Examples ===

def translate_text(text, src_lang, tgt_lang, model, tokenizer, device="cpu"):
    """Translate text using trained model."""
    # Create prefix based on direction
    if src_lang == "english" and tgt_lang == "khasi":
        prefix = "translate English to Khasi: "
    elif src_lang == "khasi" and tgt_lang == "english":
        prefix = "translate Khasi to English: "
    else:
        logger.error(f"Unsupported language direction: {src_lang} to {tgt_lang}")
        return ""
    
    # Prepare input
    input_text = prefix + text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, 
                         truncation=True).input_ids.to(device)
    
    # Generate translation
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=MAX_TARGET_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True
        )
    
    # Decode output
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translation

# Function to generate and log example translations
def generate_example_translations(num_examples=10):
    """Generate and log example translations from test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    logger.info("\n=== Example Translations from Test Set ===")
    
    # Get test examples with original texts
    test_examples = raw_datasets["test"]
    
    # Select a sample of examples
    indices = random.sample(range(len(test_examples)), min(num_examples, len(test_examples)))
    
    for idx in indices:
        example = test_examples[idx]
        english_text = example["original_english"]
        khasi_text = example["original_khasi"]
        
        # English to Khasi
        try:
            predicted_khasi = translate_text(english_text, "english", "khasi", model, tokenizer, device)
            logger.info(f"\nEnglish to Khasi:")
            logger.info(f"Source (English): {english_text}")
            logger.info(f"Reference (Khasi): {khasi_text}")
            logger.info(f"Prediction (Khasi): {predicted_khasi}")
        except Exception as e:
            logger.error(f"Error generating English to Khasi translation: {e}")
        
        # Khasi to English
        try:
            predicted_english = translate_text(khasi_text, "khasi", "english", model, tokenizer, device)
            logger.info(f"\nKhasi to English:")
            logger.info(f"Source (Khasi): {khasi_text}")
            logger.info(f"Reference (English): {english_text}")
            logger.info(f"Prediction (English): {predicted_english}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error generating Khasi to English translation: {e}")

# Generate examples
try:
    generate_example_translations(15)
except Exception as e:
    logger.error(f"Error generating example translations: {e}")

# Function for interactive translation testing
def interactive_translation():
    """Interactive console for testing translations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print("\n=== Interactive Translation Mode ===")
    print("Enter text to translate. Type 'en2kh' or 'kh2en' to switch direction.")
    print("Type 'exit' to quit.")
    
    # Default translation direction
    src_lang, tgt_lang = "english", "khasi"
    direction = "en2kh"
    
    while True:
        try:
            # Show current direction
            print(f"\nCurrent direction: {src_lang.capitalize()} → {tgt_lang.capitalize()}")
            
            # Get input
            user_input = input("Enter text: ").strip()
            
            # Check for commands
            if user_input.lower() == "exit":
                print("Exiting interactive mode.")
                break
            elif user_input.lower() == "en2kh":
                src_lang, tgt_lang = "english", "khasi"
                direction = "en2kh"
                print("Switched to English → Khasi")
                continue
            elif user_input.lower() == "kh2en":
                src_lang, tgt_lang = "khasi", "english"
                direction = "kh2en"
                print("Switched to Khasi → English")
                continue
            elif not user_input:
                continue
                
            # Translate text
            translation = translate_text(user_input, src_lang, tgt_lang, model, tokenizer, device)
            print(f"\nTranslation: {translation}")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")

# Optional: Run interactive testing mode if script is run directly
if __name__ == "__main__":
    print("\nTraining and evaluation completed.")
    try:
        if input("Would you like to enter interactive translation mode? (y/n): ").lower().startswith('y'):
            interactive_translation()
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")

logger.info("Script execution completed.")
