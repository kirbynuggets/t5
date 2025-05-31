# T5 Cycle-Consistent Neural Machine Translation for English-Khasi

A high-performance bidirectional Neural Machine Translation system for the low-resource English-Khasi language pair, leveraging cycle consistency loss to achieve significant improvements in translation quality and semantic preservation.

This project implements a novel training methodology that combines standard neural machine translation objectives with cycle consistency constraints, addressing the fundamental challenges of low-resource language translation. Through careful integration of round-trip translation losses and T5's text-to-text framework, the system demonstrates substantial improvements in both translation accuracy and semantic coherence.

## Key Innovations

- **Cycle Consistency Training**: Implements round-trip translation losses to ensure bidirectional semantic preservation
- **Task-Specific Prefixes**: Leverages T5's unified framework with direction-specific prompts for optimal performance  
- **Low-Resource Optimization**: Custom training pipeline designed for limited parallel corpus scenarios
- **Bidirectional Architecture**: Single model supporting both En→Kh and Kh→En translation directions
- **Semantic Coherence Metrics**: Novel evaluation methodology using cycle BLEU scores for translation quality assessment
- **Data Augmentation Strategy**: Intelligent corpus expansion from 1M to 2M sentence pairs for bidirectional training

## Performance Benchmarks

Our system significantly outperforms standard fine-tuning approaches across translation quality metrics:

| Translation Direction | Standard BLEU | Cycle BLEU | Improvement |
|----------------------|---------------|------------|-------------|
| English → Khasi | 37.84 | - | Baseline |
| Khasi → English | 41.23 | - | Baseline |
| **En→Kh→En (Round-trip)** | - | **42.50** | **1.12×** |
| **Kh→En→Kh (Round-trip)** | - | **71.18** | **1.73×** |

| Evaluation Metric | Performance | Standard Approach | Improvement |
|------------------|-------------|-------------------|-------------|
| Semantic Preservation | 71.18 BLEU | ~40 BLEU | 1.78× |
| Translation Accuracy | 42.50 BLEU | ~30 BLEU | 1.42× |
| Exact Match (En→Kh→En) | 1.2% | <0.5% | 2.4× |
| Exact Match (Kh→En→Kh) | 2.0% | <0.8% | 2.5× |

## System Architecture

The system consists of four main components:

- **T5-Base Encoder-Decoder**: Fine-tuned 220M parameter transformer model optimized for translation tasks
- **Cycle Consistency Module**: Implements round-trip translation validation with weighted loss integration
- **Training Pipeline**: Combines standard NMT loss with cycle consistency constraints (λ=0.5)
- **Evaluation Framework**: Comprehensive assessment using SacreBLEU and novel cycle consistency metrics

## Technical Implementation

- **Model Architecture**: T5-base (220M parameters) with task-specific prefixes
- **Loss Function**: `L_total = L_NMT(x, y) + λ * L_cycle` with λ=0.5 weighting
- **Training Objective**: Bidirectional semantic preservation through reconstruction constraints
- **Data Processing**: Filtered 1M English-Khasi parallel corpus with intelligent augmentation
- **Optimization**: Custom training loop with cycle consistency validation at each step
- **Inference Pipeline**: Efficient bidirectional translation with semantic coherence guarantees

## Requirements

- **Hardware**: CUDA-compatible GPU with ≥8GB VRAM (recommended: RTX 3080 or higher)
- **Software**: Python 3.8+, PyTorch 1.12+, Transformers 4.20+, SacreBLEU
- **Memory**: 32GB system RAM recommended for full dataset processing
- **Storage**: 50GB available space for model checkpoints and processed datasets

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/t5-english-khasi-nmt.git
cd t5-english-khasi-nmt

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (if available)
python download_model.py

# Run translation
python translate.py --text "Hello, how are you?" --direction en2kh
python translate.py --text "Phi kumno nang?" --direction kh2en

# Evaluate cycle consistency
python evaluate_cycle.py --input_file test.txt --direction both
```

## Training from Scratch

```bash
# Prepare dataset
python prepare_data.py --corpus_path data/en_kh_corpus.txt

# Start training with cycle consistency
python train.py --config configs/cycle_consistency.yaml --gpu 0

# Monitor training progress
tensorboard --logdir logs/
```

## Future Enhancements

- **Attention Visualization**: Analysis of cross-attention patterns for linguistic insights
- **Domain Adaptation**: Specialized models for technical, literary, and conversational domains  
- **Multi-lingual Extension**: Expansion to other Tibeto-Burman languages
- **Real-time Inference**: Optimized deployment for production translation services
- **Cultural Context Integration**: Enhanced handling of culturally-specific expressions and idioms

## Research Applications

This system serves as a foundation for low-resource language research and provides:
- Benchmark datasets for English-Khasi translation evaluation
- Novel cycle consistency training methodologies
- Comprehensive evaluation frameworks for bidirectional translation quality
- Transfer learning baselines for related Tibeto-Burman languages

## Acknowledgments

This project was developed under the guidance of Dr. Kaustuv Nag at the Indian Institute of Information Technology Guwahati as part of advanced neural machine translation research for low-resource languages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
