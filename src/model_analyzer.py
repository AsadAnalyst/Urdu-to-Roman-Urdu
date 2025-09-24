import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics import confusion_matrix
import Levenshtein
from sacrebleu import BLEU

from improved_model import ImprovedSeq2SeqModel
from improved_preprocess import ImprovedVocabularyBuilder


class ModelAnalyzer:
    """Advanced model analysis and debugging tools"""
    
    def __init__(self, model: ImprovedSeq2SeqModel, src_vocab: Dict, tgt_vocab: Dict, config: Dict):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = next(model.parameters()).device
        
        # Create reverse vocabularies
        self.src_idx_to_token = {idx: token for token, idx in src_vocab.items()}
        self.tgt_idx_to_token = {idx: token for token, idx in tgt_vocab.items()}
        
        self.vocab_builder = ImprovedVocabularyBuilder(config['data']['tokenization'])
        
    def analyze_attention_patterns(self, src_text: str, max_length: int = 100) -> Tuple[str, np.ndarray]:
        """Analyze attention patterns for a given input"""
        self.model.eval()
        
        # Tokenize input
        src_tokens = self.vocab_builder.tokenize(src_text, self.src_vocab)
        src_tensor = torch.tensor([src_tokens], device=self.device)
        
        # Generate with attention
        with torch.no_grad():
            generated_seq, attention_weights = self.model.generate(
                src_tensor, max_length=max_length
            )
        
        # Convert to text
        pred_text = self.vocab_builder.detokenize(
            generated_seq[0].cpu().tolist(), self.tgt_vocab
        )
        
        # Get attention weights as numpy array
        attention_matrix = attention_weights[0].cpu().numpy()  # [output_len, src_len]
        
        return pred_text, attention_matrix
    
    def plot_attention_heatmap(self, src_text: str, pred_text: str, attention_matrix: np.ndarray):
        """Plot attention heatmap"""
        # Prepare tokens for display
        src_tokens = list(src_text)
        pred_tokens = list(pred_text)
        
        # Trim attention matrix to match actual lengths
        attention_matrix = attention_matrix[:len(pred_tokens), :len(src_tokens)]
        
        # Create heatmap
        plt.figure(figsize=(max(10, len(src_tokens) * 0.5), max(8, len(pred_tokens) * 0.3)))
        sns.heatmap(
            attention_matrix,
            xticklabels=src_tokens,
            yticklabels=pred_tokens,
            cmap='Blues',
            cbar=True,
            square=False,
            linewidths=0.1
        )
        
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Source Characters')
        plt.ylabel('Generated Characters')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def analyze_gradient_flow(self, src_batch: torch.Tensor, tgt_batch: torch.Tensor):
        """Analyze gradient flow through the model"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(src_batch, tgt_batch)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['<PAD>'])
        loss = criterion(
            outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
            tgt_batch[:, 1:].contiguous().view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.abs().mean().item()
        
        # Plot gradient flow
        self.plot_gradient_flow(gradients)
        
        return gradients
    
    def plot_gradient_flow(self, gradients: Dict[str, float]):
        """Plot gradient flow across layers"""
        layer_names = list(gradients.keys())
        gradient_values = list(gradients.values())
        
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(layer_names)), gradient_values)
        plt.xticks(range(len(layer_names)), layer_names, rotation=90)
        plt.ylabel('Average Gradient Magnitude')
        plt.title('Gradient Flow Across Model Layers')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_character_confusions(self, predictions: List[str], targets: List[str]) -> Dict:
        """Analyze character-level confusion patterns"""
        pred_chars = ''.join(predictions)
        tgt_chars = ''.join(targets)
        
        # Character frequency analysis
        pred_char_freq = Counter(pred_chars)
        tgt_char_freq = Counter(tgt_chars)
        
        # Character error analysis
        char_errors = defaultdict(int)
        for pred, tgt in zip(predictions, targets):
            for i, (p_char, t_char) in enumerate(zip(pred, tgt)):
                if p_char != t_char:
                    char_errors[f"{t_char}->{p_char}"] += 1
        
        return {
            'pred_char_freq': pred_char_freq,
            'tgt_char_freq': tgt_char_freq,
            'char_errors': dict(char_errors)
        }
    
    def analyze_sequence_lengths(self, predictions: List[str], targets: List[str]) -> Dict:
        """Analyze sequence length patterns"""
        pred_lengths = [len(p) for p in predictions]
        tgt_lengths = [len(t) for t in targets]
        
        length_analysis = {
            'pred_avg_length': np.mean(pred_lengths),
            'tgt_avg_length': np.mean(tgt_lengths),
            'pred_std_length': np.std(pred_lengths),
            'tgt_std_length': np.std(tgt_lengths),
            'length_correlation': np.corrcoef(pred_lengths, tgt_lengths)[0, 1]
        }
        
        # Plot length distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(pred_lengths, alpha=0.7, label='Predictions', bins=20)
        plt.hist(tgt_lengths, alpha=0.7, label='Targets', bins=20)
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.title('Sequence Length Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(tgt_lengths, pred_lengths, alpha=0.6)
        plt.plot([min(tgt_lengths), max(tgt_lengths)], [min(tgt_lengths), max(tgt_lengths)], 'r--')
        plt.xlabel('Target Length')
        plt.ylabel('Prediction Length')
        plt.title('Length Correlation')
        
        plt.tight_layout()
        plt.show()
        
        return length_analysis
    
    def analyze_repetition_patterns(self, predictions: List[str]) -> Dict:
        """Analyze character repetition patterns in predictions"""
        repetition_stats = {
            'total_repetitions': 0,
            'max_repetition_length': 0,
            'repetition_by_char': defaultdict(list)
        }
        
        for pred in predictions:
            # Find consecutive character repetitions
            i = 0
            while i < len(pred):
                char = pred[i]
                rep_count = 1
                
                while i + rep_count < len(pred) and pred[i + rep_count] == char:
                    rep_count += 1
                
                if rep_count > 2:  # Consider 3+ as repetition
                    repetition_stats['total_repetitions'] += 1
                    repetition_stats['max_repetition_length'] = max(
                        repetition_stats['max_repetition_length'], rep_count
                    )
                    repetition_stats['repetition_by_char'][char].append(rep_count)
                
                i += rep_count
        
        return dict(repetition_stats)
    
    def calculate_detailed_metrics(self, predictions: List[str], targets: List[str]) -> Dict:
        """Calculate detailed evaluation metrics"""
        metrics = {}
        
        # BLEU score
        bleu = BLEU()
        bleu_score = bleu.corpus_score(predictions, [targets]).score
        
        # Character Error Rate (CER)
        total_chars = sum(len(tgt) for tgt in targets)
        total_char_errors = sum(
            Levenshtein.distance(pred, tgt) 
            for pred, tgt in zip(predictions, targets)
        )
        cer = total_char_errors / total_chars if total_chars > 0 else 0
        
        # Word Error Rate (WER)
        total_words = sum(len(tgt.split()) for tgt in targets)
        total_word_errors = sum(
            Levenshtein.distance(pred.split(), tgt.split())
            for pred, tgt in zip(predictions, targets)
        )
        wer = total_word_errors / total_words if total_words > 0 else 0
        
        # Exact match accuracy
        exact_matches = sum(1 for pred, tgt in zip(predictions, targets) if pred == tgt)
        exact_match_accuracy = exact_matches / len(predictions) if predictions else 0
        
        # Average edit distance
        edit_distances = [
            Levenshtein.distance(pred, tgt) 
            for pred, tgt in zip(predictions, targets)
        ]
        avg_edit_distance = np.mean(edit_distances)
        
        metrics = {
            'bleu_score': bleu_score,
            'character_error_rate': cer,
            'word_error_rate': wer,
            'exact_match_accuracy': exact_match_accuracy,
            'average_edit_distance': avg_edit_distance,
            'total_predictions': len(predictions)
        }
        
        return metrics
    
    def comprehensive_analysis(self, test_data: List[Tuple[str, str]], output_dir: str = "analysis_results"):
        """Perform comprehensive model analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Performing comprehensive model analysis...")
        
        # Generate predictions
        predictions = []
        targets = []
        attention_matrices = []
        
        self.model.eval()
        with torch.no_grad():
            for src_text, tgt_text in test_data[:100]:  # Limit for analysis
                pred_text, attention_matrix = self.analyze_attention_patterns(src_text)
                predictions.append(pred_text)
                targets.append(tgt_text)
                attention_matrices.append(attention_matrix)
        
        # 1. Calculate detailed metrics
        print("Calculating detailed metrics...")
        metrics = self.calculate_detailed_metrics(predictions, targets)
        
        # 2. Analyze character confusions
        print("Analyzing character confusions...")
        confusion_analysis = self.analyze_character_confusions(predictions, targets)
        
        # 3. Analyze sequence lengths
        print("Analyzing sequence lengths...")
        length_analysis = self.analyze_sequence_lengths(predictions, targets)
        
        # 4. Analyze repetition patterns
        print("Analyzing repetition patterns...")
        repetition_analysis = self.analyze_repetition_patterns(predictions)
        
        # 5. Save attention examples
        print("Saving attention examples...")
        self.save_attention_examples(
            test_data[:10], predictions[:10], attention_matrices[:10], output_dir
        )
        
        # 6. Create analysis report
        analysis_report = {
            'metrics': metrics,
            'character_analysis': {
                'confusions': confusion_analysis,
                'repetitions': repetition_analysis
            },
            'length_analysis': length_analysis,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'src_vocab_size': len(self.src_vocab),
                'tgt_vocab_size': len(self.tgt_vocab)
            }
        }
        
        # Save report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"BLEU Score: {metrics['bleu_score']:.2f}")
        print(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
        print(f"Word Error Rate: {metrics['word_error_rate']:.4f}")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"Average Edit Distance: {metrics['average_edit_distance']:.2f}")
        print(f"Total Repetitions: {repetition_analysis['total_repetitions']}")
        print(f"Max Repetition Length: {repetition_analysis['max_repetition_length']}")
        print(f"Length Correlation: {length_analysis['length_correlation']:.3f}")
        
        return analysis_report
    
    def save_attention_examples(self, test_data: List[Tuple[str, str]], 
                              predictions: List[str], attention_matrices: List[np.ndarray], 
                              output_dir: str):
        """Save attention visualization examples"""
        attention_dir = os.path.join(output_dir, 'attention_examples')
        os.makedirs(attention_dir, exist_ok=True)
        
        for i, ((src_text, tgt_text), pred_text, attention_matrix) in enumerate(
            zip(test_data, predictions, attention_matrices)
        ):
            plt.figure(figsize=(12, 8))
            
            # Prepare tokens
            src_tokens = list(src_text)
            pred_tokens = list(pred_text)
            
            # Trim attention matrix
            attention_matrix = attention_matrix[:len(pred_tokens), :len(src_tokens)]
            
            # Create heatmap
            sns.heatmap(
                attention_matrix,
                xticklabels=src_tokens,
                yticklabels=pred_tokens,
                cmap='Blues',
                cbar=True
            )
            
            plt.title(f'Attention Example {i+1}\nSource: {src_text}\nTarget: {tgt_text}\nPrediction: {pred_text}')
            plt.xlabel('Source Characters')
            plt.ylabel('Generated Characters')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(attention_dir, f'attention_example_{i+1}.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()


def analyze_model_performance(model_path: str, config_path: str, test_data_path: str):
    """Main function to analyze model performance"""
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load vocabularies
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    # Create model
    from improved_model import create_improved_model
    model = create_improved_model(config, len(src_vocab), len(tgt_vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    test_data = list(zip(test_df['urdu'].tolist(), test_df['roman'].tolist()))
    
    # Create analyzer
    analyzer = ModelAnalyzer(model, src_vocab, tgt_vocab, config)
    
    # Perform comprehensive analysis
    analysis_report = analyzer.comprehensive_analysis(test_data)
    
    return analysis_report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze model performance')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--test_data', required=True, help='Path to test data CSV')
    parser.add_argument('--output_dir', default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Analyze model
    analyze_model_performance(args.model, args.config, args.test_data)