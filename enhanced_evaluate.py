"""
Enhanced Evaluation Script with Token Analysis and Visualization
Shows detailed breakdowns of transliteration quality with attention analysis
"""

import torch
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from token_analyzer import TokenAnalyzer
from improved_model import ImprovedSeq2SeqModel, create_improved_model
from improved_preprocess import ImprovedVocabularyBuilder

class EnhancedEvaluator:
    """Comprehensive evaluation with token analysis and visualization"""
    
    def __init__(self, model_path: str, config_path: str, vocab_path: str = None):
        """
        Initialize the enhanced evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            vocab_path: Path to vocabularies (optional)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.vocab_path = vocab_path
        
        # Load components
        self.config = self._load_config()
        self.model, self.src_vocab, self.tgt_vocab = self._load_model_and_vocabs()
        self.token_analyzer = TokenAnalyzer(self.model, self.src_vocab, self.tgt_vocab)
        
        print(f"✅ Enhanced evaluator initialized successfully!")
        print(f"   Model parameters: {self._count_parameters():,}")
        print(f"   Source vocabulary size: {len(self.src_vocab)}")
        print(f"   Target vocabulary size: {len(self.tgt_vocab)}")
    
    def _load_config(self) -> Dict:
        """Load model configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model_and_vocabs(self):
        """Load trained model and vocabularies"""
        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Get vocabularies from checkpoint
        if 'src_vocab' in checkpoint and 'tgt_vocab' in checkpoint:
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
        else:
            # Try to load from separate files
            if self.vocab_path and os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                    src_vocab = vocab_data['src_vocab']
                    tgt_vocab = vocab_data['tgt_vocab']
            else:
                raise FileNotFoundError("Vocabularies not found in checkpoint or separate file")
        
        # Create model
        model = create_improved_model(
            self.config,
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab)
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, src_vocab, tgt_vocab
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def evaluate_samples(self, samples: List[Dict[str, str]], 
                        save_results: bool = True, 
                        show_detailed: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of samples with token analysis
        
        Args:
            samples: List of {'source': urdu_text, 'target': roman_text}
            save_results: Whether to save detailed results to files
            show_detailed: Whether to print detailed analysis
            
        Returns:
            Complete evaluation results
        """
        print(f"\n🚀 Starting Enhanced Evaluation of {len(samples)} samples...")
        print("="*80)
        
        # Analyze all samples
        analyses = self.token_analyzer.batch_analyze(samples, verbose=True)
        
        # Calculate overall statistics
        stats = self._calculate_statistics(analyses)
        
        # Print results
        if show_detailed:
            self._print_detailed_results(analyses, stats)
        else:
            self._print_summary_results(stats)
        
        # Save results
        if save_results:
            self._save_results(analyses, stats)
        
        return {
            'analyses': analyses,
            'statistics': stats,
            'config': self.config
        }
    
    def _calculate_statistics(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        # Accuracy metrics
        exact_matches = sum(1 for a in analyses if a['accuracy'] and a['accuracy']['exact_match'])
        char_accuracies = [a['accuracy']['character_accuracy'] for a in analyses if a['accuracy']]
        similarities = [a['accuracy']['similarity'] for a in analyses if a['accuracy']]
        edit_distances = [a['accuracy']['edit_distance'] for a in analyses if a['accuracy']]
        
        # Token statistics
        source_lengths = [len(a['source']['tokens']) for a in analyses]
        prediction_lengths = [len(a['prediction']['tokens']) for a in analyses]
        confidence_scores = []
        for a in analyses:
            if a['prediction']['confidence_scores']:
                confidence_scores.extend(a['prediction']['confidence_scores'])
        
        # Attention statistics
        attention_entropies = []
        for a in analyses:
            if a['attention_matrix']:
                # Calculate entropy of attention weights
                att_matrix = np.array(a['attention_matrix'])
                for row in att_matrix:
                    if row.sum() > 0:
                        normalized_row = row / row.sum()
                        entropy = -np.sum(normalized_row * np.log(normalized_row + 1e-8))
                        attention_entropies.append(entropy)
        
        return {
            'accuracy': {
                'exact_match_rate': exact_matches / len(analyses),
                'exact_matches': exact_matches,
                'total_samples': len(analyses),
                'avg_character_accuracy': np.mean(char_accuracies) if char_accuracies else 0.0,
                'avg_similarity': np.mean(similarities) if similarities else 0.0,
                'avg_edit_distance': np.mean(edit_distances) if edit_distances else 0.0,
                'character_accuracy_std': np.std(char_accuracies) if char_accuracies else 0.0
            },
            'tokens': {
                'avg_source_length': np.mean(source_lengths),
                'avg_prediction_length': np.mean(prediction_lengths),
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0,
                'length_ratio': np.mean(prediction_lengths) / np.mean(source_lengths) if source_lengths else 1.0
            },
            'attention': {
                'avg_entropy': np.mean(attention_entropies) if attention_entropies else 0.0,
                'entropy_std': np.std(attention_entropies) if attention_entropies else 0.0
            }
        }
    
    def _print_detailed_results(self, analyses: List[Dict], stats: Dict):
        """Print detailed analysis results"""
        print(f"\n📊 OVERALL STATISTICS")
        print("="*80)
        acc = stats['accuracy']
        print(f"🎯 Accuracy Metrics:")
        print(f"   Exact Matches: {acc['exact_matches']}/{acc['total_samples']} ({acc['exact_match_rate']:.1%})")
        print(f"   Avg Character Accuracy: {acc['avg_character_accuracy']:.3f} ± {acc['character_accuracy_std']:.3f}")
        print(f"   Avg Similarity: {acc['avg_similarity']:.3f}")
        print(f"   Avg Edit Distance: {acc['avg_edit_distance']:.1f}")
        
        tokens = stats['tokens']
        print(f"\n🔤 Token Statistics:")
        print(f"   Avg Source Length: {tokens['avg_source_length']:.1f} characters")
        print(f"   Avg Prediction Length: {tokens['avg_prediction_length']:.1f} characters")
        print(f"   Length Ratio: {tokens['length_ratio']:.3f}")
        print(f"   Avg Confidence: {tokens['avg_confidence']:.3f} ± {tokens['confidence_std']:.3f}")
        
        att = stats['attention']
        print(f"\n🎯 Attention Analysis:")
        print(f"   Avg Attention Entropy: {att['avg_entropy']:.3f} ± {att['entropy_std']:.3f}")
        
        # Show individual sample analyses
        print(f"\n📝 DETAILED SAMPLE ANALYSES")
        print("="*80)
        for analysis in analyses:
            self.token_analyzer.print_analysis(analysis)
    
    def _print_summary_results(self, stats: Dict):
        """Print summary results only"""
        acc = stats['accuracy']
        tokens = stats['tokens']
        
        print(f"\n📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"🎯 Exact Matches: {acc['exact_matches']}/{acc['total_samples']} ({acc['exact_match_rate']:.1%})")
        print(f"📏 Avg Character Accuracy: {acc['avg_character_accuracy']:.3f}")
        print(f"🔄 Avg Similarity: {acc['avg_similarity']:.3f}")
        print(f"💡 Avg Confidence: {tokens['avg_confidence']:.3f}")
    
    def _save_results(self, analyses: List[Dict], stats: Dict):
        """Save results to files"""
        os.makedirs('logs/enhanced_analysis', exist_ok=True)
        
        # Save detailed analyses
        with open('logs/enhanced_analysis/detailed_token_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2, default=str)
        
        # Save statistics
        with open('logs/enhanced_analysis/evaluation_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(analyses, stats)
        
        print(f"\n💾 Results saved to:")
        print(f"   📄 logs/enhanced_analysis/detailed_token_analysis.json")
        print(f"   📊 logs/enhanced_analysis/evaluation_statistics.json")
        print(f"   📝 logs/enhanced_analysis/evaluation_report.md")
    
    def _create_summary_report(self, analyses: List[Dict], stats: Dict):
        """Create markdown summary report"""
        acc = stats['accuracy']
        tokens = stats['tokens']
        att = stats['attention']
        
        report = f"""# Enhanced Urdu to Roman Transliteration Evaluation Report

## 📊 Overall Performance

### Accuracy Metrics
- **Exact Matches**: {acc['exact_matches']}/{acc['total_samples']} ({acc['exact_match_rate']:.1%})
- **Average Character Accuracy**: {acc['avg_character_accuracy']:.3f} ± {acc['character_accuracy_std']:.3f}
- **Average Similarity**: {acc['avg_similarity']:.3f}
- **Average Edit Distance**: {acc['avg_edit_distance']:.1f}

### Token Statistics
- **Average Source Length**: {tokens['avg_source_length']:.1f} characters
- **Average Prediction Length**: {tokens['avg_prediction_length']:.1f} characters
- **Length Ratio**: {tokens['length_ratio']:.3f}
- **Average Confidence**: {tokens['avg_confidence']:.3f} ± {tokens['confidence_std']:.3f}

### Attention Analysis
- **Average Attention Entropy**: {att['avg_entropy']:.3f} ± {att['entropy_std']:.3f}

## 🎯 Sample Results

| Sample | Source | Target | Prediction | Accuracy | Confidence |
|--------|--------|--------|------------|----------|------------|
"""
        
        for i, analysis in enumerate(analyses[:10]):  # Show first 10 samples
            source = analysis['source']['text'][:30] + ('...' if len(analysis['source']['text']) > 30 else '')
            target = analysis['target']['text'][:30] + ('...' if analysis['target']['text'] and len(analysis['target']['text']) > 30 else '') if analysis['target']['text'] else 'N/A'
            prediction = analysis['prediction']['text'][:30] + ('...' if len(analysis['prediction']['text']) > 30 else '')
            
            if analysis['accuracy']:
                accuracy = f"{analysis['accuracy']['character_accuracy']:.3f}"
            else:
                accuracy = "N/A"
            
            if analysis['prediction']['confidence_scores']:
                confidence = f"{np.mean(analysis['prediction']['confidence_scores']):.3f}"
            else:
                confidence = "N/A"
            
            report += f"| {i+1} | {source} | {target} | {prediction} | {accuracy} | {confidence} |\n"
        
        report += f"""
## 🔧 Model Configuration

- **Model Type**: Enhanced BiLSTM Encoder-Decoder with Attention
- **Parameters**: {self._count_parameters():,}
- **Source Vocabulary**: {len(self.src_vocab)} characters
- **Target Vocabulary**: {len(self.tgt_vocab)} characters

## 📈 Performance Analysis

The model shows {'excellent' if acc['exact_match_rate'] > 0.8 else 'good' if acc['exact_match_rate'] > 0.6 else 'moderate'} performance with {acc['exact_match_rate']:.1%} exact matches.

{'🎉 Outstanding results! The model has achieved near-perfect transliteration quality.' if acc['exact_match_rate'] > 0.8 else '✅ Good performance with room for improvement in edge cases.' if acc['exact_match_rate'] > 0.6 else '🔧 Model shows promise but may benefit from additional training or data.'}
"""
        
        with open('logs/enhanced_analysis/evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)


def create_attention_heatmap(analysis: Dict, save_path: str = None):
    """Create attention heatmap visualization"""
    if not analysis['attention_matrix']:
        print("No attention weights available for visualization")
        return
    
    # Extract data
    attention_matrix = np.array(analysis['attention_matrix'])
    source_tokens = analysis['source']['tokens']
    prediction_tokens = analysis['prediction']['tokens']
    
    # Limit to reasonable size for visualization
    max_tokens = 30
    attention_matrix = attention_matrix[:max_tokens, :max_tokens]
    source_tokens = source_tokens[:max_tokens]
    prediction_tokens = prediction_tokens[:max_tokens]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Plot heatmap
    ax = sns.heatmap(
        attention_matrix,
        xticklabels=source_tokens,
        yticklabels=prediction_tokens,
        cmap=cmap,
        annot=True,
        fmt='.3f',
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(f'Character-Level Attention Weights\\nSource: {analysis["source"]["text"][:50]}...', 
              fontsize=14, pad=20)
    plt.xlabel('Source (Urdu) Characters', fontsize=12)
    plt.ylabel('Target (Roman) Characters', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to: {save_path}")
    
    plt.show()


def main():
    """Main evaluation function"""
    print("🚀 Enhanced Urdu to Roman Transliteration Evaluation")
    print("="*60)
    
    # Configuration
    model_path = 'models/best_model.pth'
    config_path = 'improved_config.json'
    
    # Your epoch 20 samples (perfect results!)
    test_samples = [
        {
            "source": "اس ایک بات پہ دنیا سے جنگ جاری ہے",
            "target": "is ek baat pe duniy se jag jaar hai"
        },
        {
            "source": "پرو کی اب کے نہی حوصلو کی باری ہے",
            "target": "paro k ab ke nah hauslo k baar hai"
        },
        {
            "source": "گردش مجنو بہ چشمک ہائے لیلیٰ آشنا",
            "target": "gardish-e-majn ba-chashmak-h-e-lail shn"
        },
        {
            "source": "ابھی سے تجھ کو بہت ناگوار ہی ہمدم",
            "target": "abh se tujh ko bahut ngavr hai hamdam"
        },
        {
            "source": "جس تشنہ لب کے ہاتھ می جام شراب ہے",
            "target": "jis tishna-lab ke haath me jm-e-sharb hai"
        },
        {
            "source": "کیا جی لگا کے سنتے افسانے آدمی ہی",
            "target": "ky j lag ke sunte afsne aadm hai"
        }
    ]
    
    try:
        # Initialize evaluator
        evaluator = EnhancedEvaluator(model_path, config_path)
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_samples(
            test_samples,
            save_results=True,
            show_detailed=True
        )
        
        # Create attention visualization for first sample
        if results['analyses']:
            print(f"\\n🎨 Creating attention visualization...")
            create_attention_heatmap(
                results['analyses'][0], 
                'logs/enhanced_analysis/attention_heatmap_sample1.png'
            )
        
        print(f"\\n🎉 Enhanced evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()