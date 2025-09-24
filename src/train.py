import os
import json
import argparse
import time
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np

from model import create_model, count_parameters
from preprocess import Seq2SeqDataset, collate_fn
from utils import (
    set_seed, get_device, load_vocabularies, save_checkpoint, 
    load_checkpoint, EarlyStopping, LearningRateScheduler,
    ProgressTracker, calculate_perplexity, get_lr, log_experiment
)


class Trainer:
    """Trainer class for sequence-to-sequence model"""
    
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
        
        # Paths
        self.models_dir = self.config['paths']['models_dir']
        self.logs_dir = self.config['paths']['logs_dir']
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        print(f"Training on device: {self.device}")
    
    def load_data(self):
        """Load preprocessed data and create data loaders"""
        print("Loading preprocessed data...")
        
        # Load vocabularies
        self.src_vocab, self.tgt_vocab = load_vocabularies("data/processed/vocabularies.pkl")
        
        print(f"Source vocabulary size: {len(self.src_vocab['vocab'])}")
        print(f"Target vocabulary size: {len(self.tgt_vocab['vocab'])}")
        
        # Load train and validation data
        train_data = torch.load("data/processed/train_data.pt")
        val_data = torch.load("data/processed/val_data.pt")
        
        # Create datasets
        train_dataset = Seq2SeqDataset(train_data)
        val_dataset = Seq2SeqDataset(val_data)
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=collate_fn, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=0
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def build_model(self):
        """Build and initialize the model"""
        print("Building model...")
        
        src_vocab_size = len(self.src_vocab['vocab'])
        tgt_vocab_size = len(self.tgt_vocab['vocab'])
        
        # Create model
        self.model = create_model(self.config, src_vocab_size, tgt_vocab_size)
        self.model.to(self.device)
        
        # Print model info
        param_count = count_parameters(self.model)
        print(f"Model created with {param_count:,} trainable parameters")
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['word2idx']['<PAD>'])
        
        # Initialize optimizer
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize learning rate scheduler
        self.scheduler = LearningRateScheduler(self.optimizer, 'plateau', patience=3)
        
        # Initialize early stopping
        patience = self.config['training']['patience']
        self.early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
        
        print("Model, optimizer, and scheduler initialized")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (src_batch, tgt_batch) in enumerate(pbar):
            # Move to device
            src_batch = src_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            teacher_forcing_ratio = self.config['training']['teacher_forcing_ratio']
            outputs = self.model(src_batch, tgt_batch, teacher_forcing_ratio)
            
            # Calculate loss (exclude first token which is SOS)
            loss = self.criterion(
                outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
                tgt_batch[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_value = self.config['training']['gradient_clip']
            clip_grad_norm_(self.model.parameters(), clip_value)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'PPL': f'{calculate_perplexity(avg_loss):.2f}'
            })
        
        return epoch_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for src_batch, tgt_batch in pbar:
                # Move to device
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # Forward pass (no teacher forcing during validation)
                outputs = self.model(src_batch, tgt_batch, teacher_forcing_ratio=1.0)
                
                # Calculate loss
                loss = self.criterion(
                    outputs[:, 1:].contiguous().view(-1, outputs.size(-1)),
                    tgt_batch[:, 1:].contiguous().view(-1)
                )
                
                val_loss += loss.item()
                avg_loss = val_loss / (len(pbar.desc) if hasattr(pbar, 'desc') else 1)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{avg_loss:.4f}',
                    'PPL': f'{calculate_perplexity(avg_loss):.2f}'
                })
        
        return val_loss / num_batches
    
    def train(self, resume_from: str = None) -> Dict:
        """Main training loop"""
        print("Starting training...")
        
        # Load data and build model
        self.load_data()
        self.build_model()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            self.model, self.optimizer, start_epoch, _ = load_checkpoint(
                resume_from, self.model, self.optimizer, self.device
            )
            print(f"Resumed training from epoch {start_epoch}")
        
        # Training configuration
        num_epochs = self.config['training']['epochs']
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = get_lr(self.optimizer)
            
            # Update progress tracker
            self.progress_tracker.update(epoch, train_loss, val_loss, current_lr)
            
            # Calculate metrics
            train_ppl = calculate_perplexity(train_loss)
            val_ppl = calculate_perplexity(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print("-" * 50)
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.models_dir, "best_model.pt")
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path,
                    additional_info={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_ppl': train_ppl,
                        'val_ppl': val_ppl,
                        'config': self.config,
                        'src_vocab_size': len(self.src_vocab['vocab']),
                        'tgt_vocab_size': len(self.tgt_vocab['vocab'])
                    }
                )
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(self.models_dir, f"checkpoint_epoch_{epoch}.pt")
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final results
        results = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'final_train_ppl': train_ppl,
            'final_val_ppl': val_ppl,
            'epochs_trained': epoch,
            'total_parameters': count_parameters(self.model)
        }
        
        # Save progress and results
        progress_path = os.path.join(self.logs_dir, "training_progress.json")
        self.progress_tracker.save_progress(progress_path)
        
        results_path = os.path.join(self.logs_dir, "training_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Results saved to: {results_path}")
        
        return results
    
    def generate_sample_translations(self, num_samples: int = 5) -> None:
        """Generate sample translations for qualitative evaluation"""
        print(f"\nGenerating {num_samples} sample translations...")
        
        # Load test data
        test_data = torch.load("data/processed/test_data.pt")
        test_dataset = Seq2SeqDataset(test_data[:num_samples])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for i, (src_batch, tgt_batch) in enumerate(test_loader):
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # Generate translation
                generated_seq, attention_weights = self.model.generate(src_batch, max_length=100)
                
                # Convert sequences to text
                src_text = self._sequence_to_text(src_batch[0], self.src_vocab['idx2word'])
                tgt_text = self._sequence_to_text(tgt_batch[0], self.tgt_vocab['idx2word'])
                pred_text = self._sequence_to_text(generated_seq[0], self.tgt_vocab['idx2word'])
                
                sample = {
                    'source': src_text,
                    'target': tgt_text,
                    'prediction': pred_text
                }
                samples.append(sample)
                
                print(f"\nSample {i+1}:")
                print(f"Source: {src_text}")
                print(f"Target: {tgt_text}")
                print(f"Predicted: {pred_text}")
        
        # Save samples
        samples_path = os.path.join(self.logs_dir, "sample_translations.json")
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"\nSample translations saved to: {samples_path}")
    
    def _sequence_to_text(self, sequence: torch.Tensor, idx2word: Dict) -> str:
        """Convert sequence of indices to text"""
        words = []
        for idx in sequence.cpu().tolist():
            word = idx2word.get(idx, '<UNK>')
            if word in ['<PAD>', '<SOS>']:
                continue
            if word == '<EOS>':
                break
            words.append(word)
        
        # Join characters or words
        if len(words) > 0 and len(words[0]) == 1:  # Character-level
            return ''.join(words)
        else:  # Word-level
            return ' '.join(words)


def main():
    parser = argparse.ArgumentParser(description="Train Urdu to Roman Urdu Seq2Seq Model")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment", default=None, help="Experiment name for logging")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    try:
        # Run training
        results = trainer.train(resume_from=args.resume)
        
        # Generate sample translations
        trainer.generate_sample_translations()
        
        # Log experiment if name provided
        if args.experiment:
            log_experiment(trainer.config, results, f"experiments_{args.experiment}.json")
        
        print("\nTraining pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()