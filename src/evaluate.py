import os
import json
import argparse
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import Levenshtein

# Import sacrebleu for BLEU calculation
try:
    from sacrebleu import corpus_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. BLEU scores will not be calculated.")

from model import create_model
from preprocess import Seq2SeqDataset, collate_fn
from utils import load_vocabularies, load_checkpoint, get_device, sequence_to_text


class Evaluator:
    """Evaluation class for sequence-to-sequence model"""
    
    def __init__(self, model_path: str, config_path: str = "config.json"):
        self.device = get_device()
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Load vocabularies
        self.src_vocab, self.tgt_vocab = load_vocabularies("data/processed/vocabularies.pkl")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['word2idx']['<PAD>'])
        
        print(f"Evaluator initialized on device: {self.device}")
        print(f"Model loaded from: {model_path}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        src_vocab_size = len(self.src_vocab['vocab'])
        tgt_vocab_size = len(self.tgt_vocab['vocab'])
        
        # Create model
        model = create_model(self.config, src_vocab_size, tgt_vocab_size)
        model.to(self.device)
        
        # Load checkpoint
        model, _, epoch, loss = load_checkpoint(model_path, model, device=self.device)
        print(f"Loaded model from epoch {epoch} with loss {loss:.4f}")
        
        return model
    
    def calculate_perplexity(self, data_loader: DataLoader) -> float:
        """Calculate perplexity on a dataset"""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(data_loader, desc="Calculating perplexity"):
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(src_batch, tgt_batch, teacher_forcing_ratio=1.0)
                
                # Calculate loss
                loss = self.criterion(
                    outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
                    tgt_batch[:, 1:].contiguous().view(-1)
                )
                
                # Count non-padding tokens
                non_pad_tokens = (tgt_batch[:, 1:] != self.tgt_vocab['word2idx']['<PAD>']).sum().item()
                
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU score using sacrebleu"""
        if not SACREBLEU_AVAILABLE:
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        # Format references as list of lists (sacrebleu format)
        formatted_refs = [[ref] for ref in references]
        
        # Calculate BLEU scores
        bleu = corpus_bleu(predictions, formatted_refs)
        
        # Calculate individual n-gram BLEU scores
        bleu_1 = corpus_bleu(predictions, formatted_refs, max_ngram_order=1)
        bleu_2 = corpus_bleu(predictions, formatted_refs, max_ngram_order=2)
        bleu_3 = corpus_bleu(predictions, formatted_refs, max_ngram_order=3)
        bleu_4 = corpus_bleu(predictions, formatted_refs, max_ngram_order=4)
        
        return {
            'bleu': bleu.score,
            'bleu_1': bleu_1.score,
            'bleu_2': bleu_2.score,
            'bleu_3': bleu_3.score,
            'bleu_4': bleu_4.score
        }
    
    def calculate_edit_distance_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate character-level edit distance metrics"""
        total_edit_distance = 0
        total_ref_length = 0
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            # Calculate Levenshtein distance
            edit_dist = Levenshtein.distance(pred, ref)
            total_edit_distance += edit_dist
            total_ref_length += len(ref)
            
            # Check for exact match
            if pred == ref:
                exact_matches += 1
        
        # Character Error Rate (CER)
        cer = (total_edit_distance / total_ref_length) * 100 if total_ref_length > 0 else 100
        
        # Exact Match Accuracy
        exact_match_acc = (exact_matches / len(predictions)) * 100 if predictions else 0
        
        # Average Edit Distance
        avg_edit_distance = total_edit_distance / len(predictions) if predictions else 0
        
        return {
            'character_error_rate': cer,
            'exact_match_accuracy': exact_match_acc,
            'average_edit_distance': avg_edit_distance,
            'total_edit_distance': total_edit_distance
        }
    
    def generate_translations(self, data_loader: DataLoader, max_samples: int = None) -> Tuple[List[str], List[str], List[str]]:
        """Generate translations for evaluation"""
        sources = []
        predictions = []
        references = []
        
        sample_count = 0
        
        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(data_loader, desc="Generating translations"):
                if max_samples and sample_count >= max_samples:
                    break
                
                src_batch = src_batch.to(self.device)
                
                # Generate translations
                generated_seqs, _ = self.model.generate(src_batch, max_length=100)
                
                # Convert to text
                for i in range(src_batch.shape[0]):
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    src_text = sequence_to_text(
                        src_batch[i].cpu().tolist(), self.src_vocab['idx2word']
                    )
                    pred_text = sequence_to_text(
                        generated_seqs[i].cpu().tolist(), self.tgt_vocab['idx2word']
                    )
                    ref_text = sequence_to_text(
                        tgt_batch[i].cpu().tolist(), self.tgt_vocab['idx2word']
                    )
                    
                    sources.append(src_text)
                    predictions.append(pred_text)
                    references.append(ref_text)
                    
                    sample_count += 1
        
        return sources, predictions, references
    
    def evaluate_dataset(self, dataset_type: str = "test", max_samples: int = None) -> Dict:
        """Evaluate model on a specific dataset split"""
        print(f"\nEvaluating on {dataset_type} dataset...")
        
        # Load data
        data_path = f"data/processed/{dataset_type}_data.pt"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        data = torch.load(data_path)
        if max_samples:
            data = data[:max_samples]
        
        dataset = Seq2SeqDataset(data)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # Calculate perplexity
        print("Calculating perplexity...")
        perplexity = self.calculate_perplexity(data_loader)
        
        # Generate translations
        print("Generating translations...")
        sources, predictions, references = self.generate_translations(data_loader, max_samples)
        
        # Calculate BLEU scores
        print("Calculating BLEU scores...")
        bleu_scores = self.calculate_bleu_score(predictions, references)
        
        # Calculate edit distance metrics
        print("Calculating edit distance metrics...")
        edit_metrics = self.calculate_edit_distance_metrics(predictions, references)
        
        # Compile results
        results = {
            'dataset': dataset_type,
            'num_samples': len(predictions),
            'perplexity': perplexity,
            **bleu_scores,
            **edit_metrics
        }
        
        return results, sources, predictions, references
    
    def generate_qualitative_examples(self, sources: List[str], predictions: List[str], 
                                    references: List[str], num_examples: int = 10) -> List[Dict]:
        """Generate qualitative examples for analysis"""
        examples = []
        
        # Calculate edit distances for ranking
        edit_distances = []
        for pred, ref in zip(predictions, references):
            edit_dist = Levenshtein.distance(pred, ref)
            edit_distances.append(edit_dist)
        
        # Get indices sorted by edit distance (best to worst)
        sorted_indices = np.argsort(edit_distances)
        
        # Select examples: best, worst, and random samples
        best_indices = sorted_indices[:num_examples//3]
        worst_indices = sorted_indices[-num_examples//3:]
        random_indices = np.random.choice(
            sorted_indices[num_examples//3:-num_examples//3], 
            num_examples - len(best_indices) - len(worst_indices), 
            replace=False
        ) if len(sorted_indices) > num_examples else []
        
        selected_indices = np.concatenate([best_indices, random_indices, worst_indices])
        
        for i, idx in enumerate(selected_indices):
            edit_dist = edit_distances[idx]
            cer = (edit_dist / len(references[idx])) * 100 if references[idx] else 100
            
            example = {
                'example_id': i + 1,
                'category': 'best' if idx in best_indices else 'worst' if idx in worst_indices else 'random',
                'source': sources[idx],
                'reference': references[idx],
                'prediction': predictions[idx],
                'edit_distance': edit_dist,
                'character_error_rate': cer,
                'exact_match': predictions[idx] == references[idx]
            }
            examples.append(example)
        
        return examples
    
    def run_complete_evaluation(self, output_dir: str = "results/evaluation", 
                               max_samples: int = None) -> Dict:
        """Run complete evaluation pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting complete evaluation...")
        
        # Evaluate on test set
        results, sources, predictions, references = self.evaluate_dataset("test", max_samples)
        
        # Generate qualitative examples
        print("Generating qualitative examples...")
        examples = self.generate_qualitative_examples(sources, predictions, references)
        
        # Print results summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Dataset: {results['dataset']}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"BLEU Score: {results['bleu']:.2f}")
        print(f"BLEU-1: {results['bleu_1']:.2f}")
        print(f"BLEU-2: {results['bleu_2']:.2f}")
        print(f"BLEU-3: {results['bleu_3']:.2f}")
        print(f"BLEU-4: {results['bleu_4']:.2f}")
        print(f"Character Error Rate: {results['character_error_rate']:.2f}%")
        print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.2f}%")
        print(f"Average Edit Distance: {results['average_edit_distance']:.2f}")
        print("="*60)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save qualitative examples
        examples_path = os.path.join(output_dir, "qualitative_examples.json")
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        # Save all translations
        translations_path = os.path.join(output_dir, "all_translations.json")
        translations_data = {
            'sources': sources,
            'predictions': predictions,
            'references': references
        }
        with open(translations_path, 'w', encoding='utf-8') as f:
            json.dump(translations_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Evaluation results: {results_path}")
        print(f"- Qualitative examples: {examples_path}")
        print(f"- All translations: {translations_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Urdu to Roman Urdu Seq2Seq Model")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--output_dir", default="results/evaluation", help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--dataset", default="test", choices=["train", "val", "test"], help="Dataset to evaluate")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = Evaluator(args.model_path, args.config)
        
        # Run evaluation
        results = evaluator.run_complete_evaluation(args.output_dir, args.max_samples)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()