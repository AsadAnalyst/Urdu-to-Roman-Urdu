# Urdu to Roman Urdu Transliteration - Usage Guide

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### 2. Data Preprocessing
```bash
# Run preprocessing (this downloads data and creates train/val/test splits)
python src/preprocess.py
```

This will:
- Download and process the Urdu Ghazals dataset
- Clean and normalize Urdu and Roman text
- Create character-level tokenization
- Split into train (50%), val (25%), test (25%)
- Save processed data to `data/processed/`

### 3. Train a Single Model
```bash
# Train with default configuration
python src/train.py --config config.json

# Resume from checkpoint
python src/train.py --config config.json --resume models/checkpoint_epoch_5.pt
```

### 4. Run All Experiments
```bash
# Run the three predefined experiments
python src/experiments.py --run_predefined --config config.json
```

This will run 9 experiments total:
- **Embedding variations**: 128, 256, 512 dimensions
- **Dropout variations**: 0.1, 0.3, 0.5 rates  
- **Hidden size variations**: 256, 512, 768 units

### 5. Evaluate a Trained Model
```bash
# Evaluate best model
python src/evaluate.py --model_path models/best_model.pt --config config.json

# Evaluate specific experiment
python src/evaluate.py --model_path experiments/exp1_embedding_256/models/best_model.pt
```

### 6. Launch Demo App
```bash
# Start Streamlit app
streamlit run app.py
```

## Project Structure

```
urdu-to-roman-urdu/
├── src/
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── model.py           # BiLSTM seq2seq model
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation metrics
│   ├── experiments.py     # Experiment runner
│   └── utils.py           # Utility functions
├── data/
│   ├── urdu_ghazals_rekhta/  # Raw dataset
│   └── processed/            # Processed data
├── models/                   # Saved model checkpoints
├── experiments/              # Experiment results
├── results/                  # Evaluation results
├── logs/                    # Training logs
├── app.py                   # Streamlit demo app
├── config.json             # Configuration file
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Configuration

Edit `config.json` to modify model hyperparameters:

```json
{
  "model": {
    "embedding_dim": 256,      # Embedding dimension
    "hidden_size": 512,        # LSTM hidden size
    "encoder_layers": 2,       # BiLSTM encoder layers
    "decoder_layers": 4,       # LSTM decoder layers
    "dropout": 0.3,           # Dropout rate
    "attention_dim": 256,     # Attention dimension
    "max_sequence_length": 100 # Max sequence length
  },
  "training": {
    "learning_rate": 0.001,    # Learning rate
    "batch_size": 64,         # Batch size
    "epochs": 50,             # Number of epochs
    "patience": 10,           # Early stopping patience
    "gradient_clip": 1.0,     # Gradient clipping
    "teacher_forcing_ratio": 0.5  # Teacher forcing ratio
  }
}
```

## Model Architecture

The model implements a sequence-to-sequence architecture with:

1. **Encoder**: Bidirectional LSTM (2 layers)
   - Input: Urdu character sequences
   - Output: Contextualized representations

2. **Attention Mechanism**: Additive attention
   - Allows decoder to focus on relevant encoder states
   - Improves translation quality

3. **Decoder**: Unidirectional LSTM (4 layers)
   - Input: Roman character sequences + attention context
   - Output: Roman Urdu transliteration

4. **Training**: Teacher forcing with configurable ratio

## Experiments

### Experiment 1: Embedding Dimension
- **Variants**: 128, 256, 512 dimensions
- **Purpose**: Find optimal embedding size for character representation

### Experiment 2: Dropout Rate  
- **Variants**: 0.1, 0.3, 0.5 rates
- **Purpose**: Balance between learning capacity and generalization

### Experiment 3: Hidden Size
- **Variants**: 256, 512, 768 units
- **Purpose**: Determine optimal model capacity

## Evaluation Metrics

The model is evaluated using:

1. **BLEU Score**: Translation quality (1-gram to 4-gram)
2. **Perplexity**: Model confidence measure
3. **Character Error Rate (CER)**: Character-level accuracy
4. **Edit Distance**: Levenshtein distance
5. **Exact Match Accuracy**: Perfect translations

## Usage Examples

### Custom Experiment
```bash
# Run custom experiment with specific modifications
python src/experiments.py \\
  --experiment_name "custom_test" \\
  --modifications '{"model.embedding_dim": 384, "model.dropout": 0.2}'
```

### Evaluate on Validation Set
```bash
# Evaluate on validation instead of test set
python src/evaluate.py \\
  --model_path models/best_model.pt \\
  --dataset val \\
  --max_samples 500
```

### Generate Sample Translations
```python
from src.evaluate import Evaluator

# Initialize evaluator
evaluator = Evaluator("models/best_model.pt")

# Generate translations
sources, predictions, references = evaluator.generate_translations(test_loader, max_samples=10)

# Print results
for i, (src, pred, ref) in enumerate(zip(sources[:5], predictions[:5], references[:5])):
    print(f"Example {i+1}:")
    print(f"Source: {src}")
    print(f"Prediction: {pred}")
    print(f"Reference: {ref}")
    print()
```

## Tips for Better Results

1. **Data Quality**: Ensure clean, consistent Urdu text input
2. **Sequence Length**: Keep input sequences under 100 characters
3. **Training Time**: Allow sufficient epochs for convergence
4. **Hyperparameter Tuning**: Start with provided configs, then experiment
5. **Evaluation**: Use multiple metrics for comprehensive assessment

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in config.json
   "batch_size": 32  # or 16
   ```

2. **Slow Training**:
   ```bash
   # Use CPU if GPU is slow
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Poor Translation Quality**:
   - Check data preprocessing
   - Increase model capacity
   - Train for more epochs
   - Adjust attention mechanism

### Data Issues

1. **Dataset Not Found**: Ensure `data/urdu_ghazals_rekhta/` exists
2. **Preprocessing Fails**: Check text encoding and file permissions
3. **Vocabulary Too Large**: Adjust `vocab_size_limit` in config

### Model Issues

1. **NaN Loss**: Reduce learning rate or gradient clipping
2. **No Improvement**: Check teacher forcing ratio
3. **Overfitting**: Increase dropout or reduce model size

## Performance Expectations

With the provided dataset and configuration:
- **Training Time**: 2-4 hours per experiment (CPU)
- **Expected BLEU**: 15-25 (character-level transliteration)
- **Expected CER**: 25-40%
- **Model Size**: ~20M parameters

## Extension Ideas

1. **Advanced Models**: Replace BiLSTM with Transformer
2. **Data Augmentation**: Add noise, back-translation
3. **Subword Tokenization**: Use BPE or SentencePiece
4. **Multi-task Learning**: Add language identification
5. **Beam Search**: Implement proper beam search decoding

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{urdu2roman2024,
  title={Urdu to Roman Urdu Transliteration using BiLSTM Sequence-to-Sequence Models},
  author={},
  year={2024},
  note={Implementation based on Urdu Ghazals Rekhta dataset}
}
```