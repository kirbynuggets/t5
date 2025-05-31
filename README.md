# Cycle Loss-Driven Fine-Tuning of T5 for Bidirectional English-Khasi Translation

A bidirectional Neural Machine Translation (NMT) system for low-resource English-Khasi language pairs using fine-tuned T5 model with cycle consistency loss.

## 📋 Overview

This project develops an effective translation system between English and Khasi languages by addressing the challenges of low-resource language translation. The system uses cycle consistency loss during training to enhance model coherence and semantic accuracy, making it particularly suitable for the limited data available for Khasi language.

## 🎯 Objectives

- Create an effective bidirectional English-Khasi translation system
- Address low-resource challenges including limited corpora and complex grammar
- Ensure semantic accuracy through comprehensive evaluation metrics
- Establish a scalable NMT benchmark for Khasi language processing
- Achieve cycle consistency BLEU scores above 40 for robust translation quality

## 🚀 Key Features

- **Bidirectional Translation**: Supports both English→Khasi and Khasi→English translation
- **Cycle Consistency**: Implements novel cycle consistency loss for improved translation quality
- **Low-Resource Optimization**: Specifically designed for languages with limited training data
- **T5-Based Architecture**: Leverages the powerful T5-base model (220M parameters)
- **Comprehensive Evaluation**: Uses both standard BLEU metrics and innovative cycle consistency checks

## 📊 Performance Results

| Metric | Value |
|--------|-------|
| BLEU (En→Kh) | 37.84 |
| BLEU (Kh→En) | 41.23 |
| **En→Kh→En Cycle BLEU** | **42.50** |
| **Kh→En→Kh Cycle BLEU** | **71.18** |
| Exact Match (En→Kh→En) | 1.2% |
| Exact Match (Kh→En→Kh) | 2.0% |

*Results based on first 40,000 training steps*

## 🏗️ Architecture & Approach

### Model Architecture
- **Base Model**: T5-base (220M parameters)
- **Task Prefixes**: 
  - `"translate English to Khasi: "` for En→Kh
  - `"translate Khasi to English: "` for Kh→En

### Training Methodology
- **Dataset**: 1M filtered English-Khasi parallel sentence pairs, augmented to 2M for bidirectional training
- **Loss Function**: Combined training objective with cycle consistency

```
L_total = L_NMT(x, y) + λ * L_cycle
```

Where:
- `L_NMT`: Standard Neural Machine Translation loss
- `L_cycle`: Cycle consistency loss encouraging reconstruction
- `λ = 0.5`: Weighting parameter

### Cycle Consistency Logic
The cycle consistency loss encourages:
- English sentence `x ≈ x''` after En→Kh→En translation
- Khasi sentence `y ≈ y''` after Kh→En→Kh translation

## 📈 Training Progress

- **Convergence**: Good convergence rate with significant improvements during the first epoch
- **Direction Preference**: Higher consistency observed in Kh→En→Kh direction (71.18 vs 42.50 BLEU)
- **Stability**: Consistent performance improvements throughout training

## ✅ Strengths

- **Simple Sentences**: Excellent performance on straightforward sentence structures
- **Proper Names**: Accurate handling of proper nouns and names
- **Descriptive Phrases**: Maintains semantic meaning in descriptive content
- **Novel Evaluation**: Pioneering use of cycle consistency for low-resource NMT evaluation

## 🔄 Current Challenges & Future Work

### Identified Challenges
- **Complex Grammar**: Difficulty with intricate grammatical structures
- **Domain-Specific Terms**: Limited performance on specialized vocabulary
- **Cultural References**: Challenges with culture-specific content

### Planned Improvements
- Training on larger, more diverse datasets
- Enhanced handling of complex grammatical structures
- Improved coverage of domain-specific terminology
- Better cultural context understanding

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run translation
python translate.py --source "Your English text here" --direction en2kh
python translate.py --source "Your Khasi text here" --direction kh2en
```

## 📊 Evaluation Metrics

- **SacreBLEU**: Standard BLEU score calculation
- **Cycle Consistency**: Novel bidirectional translation quality assessment
- **Exact Match**: Percentage of perfect round-trip translations

## 🤝 Contributing

Contributions are welcome! This project aims to advance low-resource language translation, particularly for Khasi and similar languages.

## 📚 References

- [Exploring Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Arya Sahu** (2201033)  
Indian Institute of Information Technology Guwahati

## 🏷️ Tags

`neural-machine-translation` `t5` `low-resource-languages` `khasi` `cycle-consistency` `bidirectional-translation` `nlp` `transformer`

---

*This project contributes valuable tools and benchmarks for Khasi language processing and serves as a foundation for future low-resource language translation research.*
