"""
Enhanced Token Analyzer for Urdu to Roman Transliteration
Shows detailed token breakdowns, attention weights, and character alignments
"""

import torch
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class TokenAnalyzer:
    """Comprehensive token analysis for sequence-to-sequence models"""
    
    def __init__(self, model, src_vocab: Dict, tgt_vocab: Dict):
        """
        Initialize the token analyzer
        
        Args:
            model: Trained sequence-to-sequence model
            src_vocab: Source vocabulary (Urdu characters to IDs)
            tgt_vocab: Target vocabulary (Roman characters to IDs)
        """
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Create reverse vocabularies (ID to character)
        self.src_itos = {v: k for k, v in src_vocab.items()}
        self.tgt_itos = {v: k for k, v in tgt_vocab.items()}
        
        # Special tokens
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        
    def tokenize_urdu(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize Urdu text into characters and convert to IDs
        
        Args:
            text: Input Urdu text
            
        Returns:
            Tuple of (tokens, token_ids)
        """
        # Add special tokens
        tokens = [self.sos_token] + list(text.strip()) + [self.eos_token]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.src_vocab:
                token_ids.append(self.src_vocab[token])
            else:
                token_ids.append(self.src_vocab.get(self.unk_token, 0))
                
        return tokens, token_ids
    
    def detokenize_roman(self, token_ids) -> Tuple[List[str], str]:
        """
        Convert Roman token IDs back to characters and text
        
        Args:
            token_ids: List of token IDs or tensor
            
        Returns:
            Tuple of (tokens, text)
        """
        # Handle different input types
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif isinstance(token_ids, (int, np.integer)):
            token_ids = [token_ids]
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        
        tokens = []
        text = ""
        
        for token_id in token_ids:
            token = self.tgt_itos.get(token_id, self.unk_token)
            
            # Stop at EOS token
            if token == self.eos_token:
                break
                
            # Skip special tokens in output
            if token not in [self.sos_token, self.pad_token]:
                tokens.append(token)
                text += token
                
        return tokens, text
    
    def predict_with_attention(self, urdu_text: str, max_length: int = 100) -> Dict[str, Any]:
        """
        Generate prediction with detailed attention analysis
        
        Args:
            urdu_text: Input Urdu text
            max_length: Maximum output length
            
        Returns:
            Dictionary with detailed prediction analysis
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Tokenize input
            src_tokens, src_ids = self.tokenize_urdu(urdu_text)
            src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
            
            # Generate prediction
            if hasattr(self.model, 'forward_with_attention'):
                # Use enhanced forward method with attention
                outputs, attention_weights = self.model.forward_with_attention(src_tensor)
            else:
                # Fallback to regular forward or generate method
                if hasattr(self.model, 'generate'):
                    outputs, attention_weights = self.model.generate(src_tensor, max_length=max_length)
                else:
                    outputs = self.model(src_tensor)
                    attention_weights = None
            
            # Get predicted IDs - handle different output formats
            if isinstance(outputs, tuple):
                predicted_ids = outputs[0]
            else:
                predicted_ids = outputs
                
            if predicted_ids.dim() > 2:
                predicted_ids = torch.argmax(predicted_ids, dim=-1)
            
            predicted_ids = predicted_ids.squeeze(0)
            pred_tokens, pred_text = self.detokenize_roman(predicted_ids)
            
            # Calculate confidence scores (only if we have logits)
            confidence_scores = []
            if predicted_ids.dtype == torch.float32 or predicted_ids.dtype == torch.float64:
                # We have logits, can calculate probabilities
                probabilities = torch.softmax(outputs, dim=-1).squeeze(0)
                for i, pred_id in enumerate(predicted_ids):
                    if i < probabilities.size(0):
                        confidence = probabilities[i, pred_id].item()
                        confidence_scores.append(confidence)
                    else:
                        break
            else:
                # We have token IDs already, set default confidence
                confidence_scores = [1.0] * len(pred_tokens)
            
            return {
                'input': {
                    'text': urdu_text,
                    'tokens': src_tokens,
                    'token_ids': src_ids,
                    'length': len(src_tokens)
                },
                'output': {
                    'text': pred_text,
                    'tokens': pred_tokens,
                    'token_ids': predicted_ids[:len(pred_tokens)].cpu().tolist(),
                    'length': len(pred_tokens),
                    'confidence_scores': confidence_scores[:len(pred_tokens)]
                },
                'attention': {
                    'weights': attention_weights.squeeze(0).cpu().tolist() if attention_weights is not None else None,
                    'shape': list(attention_weights.shape) if attention_weights is not None else None
                }
            }
    
    def create_alignment_analysis(self, prediction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create detailed character alignment analysis
        
        Args:
            prediction_data: Output from predict_with_attention
            
        Returns:
            List of alignment dictionaries
        """
        if prediction_data['attention']['weights'] is None:
            return []
        
        src_tokens = prediction_data['input']['tokens']
        tgt_tokens = prediction_data['output']['tokens']
        attention_weights = np.array(prediction_data['attention']['weights'])
        
        alignments = []
        
        for i, tgt_token in enumerate(tgt_tokens):
            if i < attention_weights.shape[0]:
                # Get attention weights for this target token
                weights = attention_weights[i]
                
                # Create alignment entries
                source_alignments = []
                for j, src_token in enumerate(src_tokens):
                    if j < len(weights):
                        source_alignments.append({
                            'source_token': src_token,
                            'weight': float(weights[j]),
                            'position': j
                        })
                
                # Sort by attention weight (highest first)
                source_alignments.sort(key=lambda x: x['weight'], reverse=True)
                
                alignments.append({
                    'target_token': tgt_token,
                    'target_position': i,
                    'confidence': prediction_data['output']['confidence_scores'][i] if i < len(prediction_data['output']['confidence_scores']) else 0.0,
                    'source_alignments': source_alignments[:5],  # Top 5 alignments
                    'top_alignment': source_alignments[0] if source_alignments else None
                })
        
        return alignments
    
    def analyze_sample(self, urdu_text: str, target_text: str = None) -> Dict[str, Any]:
        """
        Complete analysis of a single sample
        
        Args:
            urdu_text: Input Urdu text
            target_text: Target Roman text (optional, for comparison)
            
        Returns:
            Complete analysis dictionary
        """
        # Get prediction with attention
        prediction_data = self.predict_with_attention(urdu_text)
        
        # Create alignment analysis
        alignments = self.create_alignment_analysis(prediction_data)
        
        # Calculate accuracy if target is provided
        accuracy_metrics = None
        if target_text:
            accuracy_metrics = self.calculate_accuracy(
                prediction_data['output']['text'], 
                target_text
            )
        
        return {
            'source': {
                'text': urdu_text,
                'tokens': prediction_data['input']['tokens'],
                'token_ids': prediction_data['input']['token_ids']
            },
            'target': {
                'text': target_text,
                'tokens': list(target_text) if target_text else None
            },
            'prediction': {
                'text': prediction_data['output']['text'],
                'tokens': prediction_data['output']['tokens'],
                'token_ids': prediction_data['output']['token_ids'],
                'confidence_scores': prediction_data['output']['confidence_scores']
            },
            'alignments': alignments,
            'attention_matrix': prediction_data['attention']['weights'],
            'accuracy': accuracy_metrics
        }
    
    def calculate_accuracy(self, predicted: str, target: str) -> Dict[str, float]:
        """
        Calculate various accuracy metrics
        
        Args:
            predicted: Predicted text
            target: Target text
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Character-level accuracy
        pred_chars = list(predicted)
        tgt_chars = list(target)
        
        # Exact match
        exact_match = predicted == target
        
        # Character accuracy
        max_len = max(len(pred_chars), len(tgt_chars))
        correct_chars = sum(1 for i in range(min(len(pred_chars), len(tgt_chars))) 
                           if pred_chars[i] == tgt_chars[i])
        char_accuracy = correct_chars / max_len if max_len > 0 else 0.0
        
        # Edit distance (Levenshtein)
        edit_distance = self.levenshtein_distance(predicted, target)
        normalized_edit_distance = edit_distance / max_len if max_len > 0 else 0.0
        
        return {
            'exact_match': exact_match,
            'character_accuracy': char_accuracy,
            'edit_distance': edit_distance,
            'normalized_edit_distance': normalized_edit_distance,
            'similarity': 1.0 - normalized_edit_distance
        }
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein (edit) distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def batch_analyze(self, samples: List[Dict[str, str]], verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze multiple samples
        
        Args:
            samples: List of {'source': urdu_text, 'target': roman_text} dictionaries
            verbose: Whether to print progress
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, sample in enumerate(samples):
            if verbose:
                print(f"Analyzing sample {i+1}/{len(samples)}: {sample['source'][:50]}...")
            
            analysis = self.analyze_sample(
                sample['source'], 
                sample.get('target')
            )
            analysis['sample_id'] = i + 1
            results.append(analysis)
        
        return results
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """
        Print formatted analysis results
        
        Args:
            analysis: Analysis result from analyze_sample
        """
        print(f"\n{'='*80}")
        print(f"📝 SAMPLE {analysis.get('sample_id', '')} ANALYSIS")
        print(f"{'='*80}")
        
        # Basic info
        print(f"🔤 INPUT (Urdu):  {analysis['source']['text']}")
        if analysis['target']['text']:
            print(f"🎯 TARGET:        {analysis['target']['text']}")
        print(f"🤖 PREDICTED:     {analysis['prediction']['text']}")
        
        # Accuracy metrics
        if analysis['accuracy']:
            acc = analysis['accuracy']
            print(f"\n📊 ACCURACY METRICS:")
            print(f"   Exact Match: {'✅' if acc['exact_match'] else '❌'}")
            print(f"   Character Accuracy: {acc['character_accuracy']:.3f}")
            print(f"   Similarity: {acc['similarity']:.3f}")
            print(f"   Edit Distance: {acc['edit_distance']}")
        
        # Token breakdown
        print(f"\n🔤 TOKEN BREAKDOWN:")
        print(f"   Source Tokens:    {analysis['source']['tokens']}")
        print(f"   Predicted Tokens: {analysis['prediction']['tokens']}")
        print(f"   Source IDs:       {analysis['source']['token_ids']}")
        print(f"   Predicted IDs:    {analysis['prediction']['token_ids']}")
        
        # Confidence scores
        if analysis['prediction']['confidence_scores']:
            avg_confidence = np.mean(analysis['prediction']['confidence_scores'])
            print(f"   Avg Confidence:   {avg_confidence:.3f}")
        
        # Top alignments
        if analysis['alignments']:
            print(f"\n🎯 TOP CHARACTER ALIGNMENTS:")
            for alignment in analysis['alignments'][:10]:  # Show first 10
                if alignment['top_alignment']:
                    top = alignment['top_alignment']
                    conf = alignment['confidence']
                    print(f"   '{alignment['target_token']}' ← '{top['source_token']}' "
                          f"(att: {top['weight']:.3f}, conf: {conf:.3f})")
        
        print(f"{'='*80}")