import os
import json
import argparse
import time
import math
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from improved_model import create_improved_model, count_parameters
from improved_preprocess import create_improved_datasets, improved_collate_fn, ImprovedVocabularyBuilder
from utils import (
    set_seed, get_device, load_checkpoint, 
    EarlyStopping, ProgressTracker, calculate_perplexity, get_lr, log_experiment
)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing loss
        
        Args:
            predictions: [batch_size * seq_len, vocab_size]
            targets: [batch_size * seq_len]
        """
        # Create smoothed labels
        batch_size = predictions.size(0)
        true_dist = torch.zeros_like(predictions)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for ignore_index and true class
        
        # Set confidence for true class
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Mask padding tokens
        mask = (targets != self.ignore_index).float()
        true_dist = true_dist * mask.unsqueeze(1)
        
        # Calculate KL divergence
        log_probs = F.log_softmax(predictions, dim=1)
        loss = -(true_dist * log_probs).sum(dim=1)
        
        # Average over non-padding tokens
        return (loss * mask).sum() / mask.sum()


class ImprovedTrainer:
    """Improved trainer with advanced training techniques"""
    
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Set device and seed
        self.device = get_device()
        set_seed(42)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Vocabularies
        self.src_vocab = None
        self.tgt_vocab = None
        
        # Training utilities
        self.early_stopping = None
        self.progress_tracker = ProgressTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_perplexity': [], 
            'val_perplexity': [], 'learning_rates': [], 'gradient_norms': []
        }
        
        # Curriculum learning parameters
        self.initial_teacher_forcing = self.config['training'].get('teacher_forcing_ratio', 0.8)
        self.min_teacher_forcing = 0.3
        self.teacher_forcing_decay = 0.05
        
    def setup_data(self, data_path: str):
        """Setup datasets and data loaders"""
        print("Setting up datasets...")
        
        # Create improved datasets
        train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = create_improved_datasets(
            data_path, self.config
        )
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=improved_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=improved_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Data setup complete. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
    def setup_model(self):
        """Setup model with improved architecture"""
        print("Setting up improved model...")
        
        # Create improved model
        self.model = create_improved_model(
            self.config, 
            len(self.src_vocab), 
            len(self.tgt_vocab)
        )
        self.model.to(self.device)
        
        print(f"Model created with {count_parameters(self.model):,} parameters")
        
        # Setup improved loss function with label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=len(self.tgt_vocab),
            smoothing=self.config['training'].get('label_smoothing', 0.1),
            ignore_index=self.tgt_vocab['<PAD>']
        )
        
        # Setup improved optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 1e-4),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Setup learning rate scheduler
        scheduler_type = self.config['training'].get('scheduler', 'reduce_on_plateau')
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, 
                min_lr=1e-6
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'cosine_restart':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['patience'],
            min_delta=1e-4,
            restore_best_weights=True
        )
        
    def get_current_teacher_forcing_ratio(self) -> float:
        """Calculate current teacher forcing ratio with curriculum learning"""
        decay = self.teacher_forcing_decay * self.current_epoch
        current_ratio = max(
            self.min_teacher_forcing,
            self.initial_teacher_forcing - decay
        )
        return current_ratio
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch with improved techniques"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        gradient_norms = []
        
        # Get current teacher forcing ratio
        teacher_forcing_ratio = self.get_current_teacher_forcing_ratio()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch['src_lengths'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                src, tgt, 
                teacher_forcing_ratio=teacher_forcing_ratio,
                src_lengths=src_lengths
            )
            
            # Calculate loss (exclude first token which is SOS)
            loss = self.criterion(
                outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip']
            )
            gradient_norms.append(grad_norm.item())
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_tokens = (tgt[:, 1:] != self.tgt_vocab['<PAD>']).sum().item()
            total_tokens += num_tokens
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{math.exp(min(avg_loss, 10)):.2f}',
                'tf_ratio': f'{teacher_forcing_ratio:.3f}',
                'grad_norm': f'{grad_norm.item():.3f}'
            })
            
            # Step scheduler if using cosine restart
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(self.current_epoch + batch_idx / len(self.train_loader))
        
        avg_loss = total_loss / len(self.train_loader)
        avg_perplexity = math.exp(min(avg_loss, 10))
        avg_grad_norm = np.mean(gradient_norms)
        
        return avg_loss, avg_perplexity, avg_grad_norm
    
    def validate_epoch(self) -> Tuple[float, float, List[str]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        sample_translations = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lengths = batch['src_lengths'].to(self.device)
                
                # Forward pass (no teacher forcing during validation)
                outputs = self.model(src, tgt, teacher_forcing_ratio=0.0, src_lengths=src_lengths)
                
                # Calculate loss
                loss = self.criterion(
                    outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                
                total_loss += loss.item()
                num_tokens = (tgt[:, 1:] != self.tgt_vocab['<PAD>']).sum().item()
                total_tokens += num_tokens
                
                # Generate sample translations for the first batch
                if len(sample_translations) < 5:
                    generated_seqs, _ = self.model.generate(
                        src[:min(3, src.size(0))], 
                        max_length=100,
                        src_lengths=src_lengths[:min(3, src.size(0))]
                    )
                    
                    # Convert to text
                    vocab_builder = ImprovedVocabularyBuilder(self.config['data']['tokenization'])
                    for i in range(generated_seqs.size(0)):
                        src_text = vocab_builder.detokenize(
                            src[i].cpu().tolist(), self.src_vocab
                        )
                        tgt_text = vocab_builder.detokenize(
                            tgt[i].cpu().tolist(), self.tgt_vocab
                        )
                        pred_text = vocab_builder.detokenize(
                            generated_seqs[i].cpu().tolist(), self.tgt_vocab
                        )
                        
                        sample_translations.append({
                            'source': src_text,
                            'target': tgt_text,
                            'prediction': pred_text
                        })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_perplexity = math.exp(min(avg_loss, 10))
        
        return avg_loss, avg_perplexity, sample_translations
    
    def train(self, data_path: str):
        """Main training loop with improved techniques"""
        print("Starting improved training...")
        
        # Setup data and model
        self.setup_data(data_path)
        self.setup_model()
        
        # Training loop
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            print(f"{'='*50}")
            
            # Train epoch
            train_loss, train_ppl, avg_grad_norm = self.train_epoch()
            
            # Validate epoch
            val_loss, val_ppl, sample_translations = self.validate_epoch()
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            
            # Log metrics
            current_lr = get_lr(self.optimizer)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_perplexity'].append(train_ppl)
            self.training_history['val_perplexity'].append(val_ppl)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['gradient_norms'].append(avg_grad_norm)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Gradient Norm: {avg_grad_norm:.3f}")
            print(f"Teacher Forcing Ratio: {self.get_current_teacher_forcing_ratio():.3f}")
            
            # Show sample translations
            print(f"\nSample Translations:")
            for i, trans in enumerate(sample_translations[:3]):
                print(f"Sample {i+1}:")
                print(f"  Source: {trans['source']}")
                print(f"  Target: {trans['target']}")
                print(f"  Prediction: {trans['prediction']}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'src_vocab': self.src_vocab,
                'tgt_vocab': self.tgt_vocab,
                'config': self.config,
                'training_history': self.training_history
            }
            
            # Save checkpoint
            os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
            
            # Save latest checkpoint
            latest_path = os.path.join(self.config['paths']['models_dir'], 'latest_checkpoint.pth')
            torch.save(checkpoint, latest_path)
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(self.config['paths']['models_dir'], 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save sample translations
            self.save_sample_translations(sample_translations, epoch)
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Plot training progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_progress()
        
        print("\nTraining completed!")
        self.plot_training_progress()
        
    def save_sample_translations(self, sample_translations: List[Dict], epoch: int):
        """Save sample translations to file"""
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        
        output_file = os.path.join(
            self.config['paths']['logs_dir'], 
            f"sample_translations_epoch_{epoch+1}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_translations, f, ensure_ascii=False, indent=2)
    
    def plot_training_progress(self):
        """Plot training progress"""
        if len(self.training_history['train_loss']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Perplexity plot
        axes[0, 1].plot(self.training_history['train_perplexity'], label='Train PPL', color='blue')
        axes[0, 1].plot(self.training_history['val_perplexity'], label='Val PPL', color='red')
        axes[0, 1].set_title('Perplexity Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.training_history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Gradient norm plot
        axes[1, 1].plot(self.training_history['gradient_norms'], color='orange')
        axes[1, 1].set_title('Gradient Norm Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        plt.savefig(
            os.path.join(self.config['paths']['logs_dir'], 'training_progress.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train improved Urdu to Roman transliteration model')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--data', default='data/urdu_ghazals_rekhta/urdu_ghazals_rekhta.csv', 
                       help='Path to training data')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = ImprovedTrainer(args.config)
    trainer.train(args.data)


if __name__ == "__main__":
    main()