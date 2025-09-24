import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import pickle
from typing import Dict, List, Tuple, Any


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str, 
                   additional_info: Dict[str, Any] = None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: nn.Module, 
                   optimizer: torch.optim.Optimizer = None, 
                   device: torch.device = None):
    """Load model checkpoint"""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded from {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
    return model, optimizer, epoch, loss


def load_vocabularies(vocab_path: str) -> Tuple[Dict, Dict]:
    """Load vocabularies from pickle file"""
    with open(vocab_path, 'rb') as f:
        vocabs = pickle.load(f)
    return vocabs['urdu_vocab'], vocabs['roman_vocab']


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: Dict, filepath: str):
    """Save experiment results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filepath}")


def create_padding_mask(sequences: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Create padding mask for sequences"""
    return (sequences != pad_idx).float()


def sequence_to_text(sequence: List[int], idx2word: Dict[int, str], 
                    remove_special: bool = True) -> str:
    """Convert sequence of indices back to text"""
    words = []
    for idx in sequence:
        word = idx2word.get(idx, '<UNK>')
        if remove_special and word in ['<PAD>', '<SOS>', '<EOS>']:
            if word == '<EOS>':
                break
            continue
        words.append(word)
    
    # Join based on tokenization type (if characters, join without spaces)
    if len(words) > 0 and len(words[0]) == 1:  # Character-level
        return ''.join(words)
    else:  # Word-level
        return ' '.join(words)


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return np.exp(loss)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop early"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 scheduler_type: str = 'plateau', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'plateau':
            # Extract patience if provided in kwargs, otherwise use default
            patience = kwargs.pop('patience', 5)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience, **kwargs)
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5, **kwargs)
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: float = None):
        """Step the scheduler"""
        if self.scheduler_type == 'plateau':
            self.scheduler.step(metric)
        else:
            self.scheduler.step()


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = None):
    """Print model summary"""
    print(f"\nModel Summary:")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model structure:")
    print(model)
    
    if input_shape and torch.cuda.is_available():
        try:
            # Try to estimate model size
            model.eval()
            dummy_input = torch.randint(0, 100, input_shape).cuda()
            with torch.no_grad():
                _ = model(dummy_input, dummy_input)
            print(f"Model successfully processes input shape: {input_shape}")
        except Exception as e:
            print(f"Could not process dummy input: {e}")


def log_experiment(config: Dict, results: Dict, log_file: str = "experiment_log.json"):
    """Log experiment configuration and results"""
    experiment_data = {
        'timestamp': str(torch.tensor(1).cpu().numpy()),  # Simple timestamp
        'config': config,
        'results': results
    }
    
    # Load existing log or create new one
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    else:
        log_data = {'experiments': []}
    
    log_data['experiments'].append(experiment_data)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"Experiment logged to {log_file}")


def visualize_attention(attention_weights: np.ndarray, 
                       source_tokens: List[str], 
                       target_tokens: List[str],
                       save_path: str = None):
    """Visualize attention weights (placeholder for matplotlib implementation)"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, 
                   xticklabels=source_tokens, 
                   yticklabels=target_tokens,
                   cmap='Blues', 
                   cbar=True)
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Tokens')
        plt.title('Attention Weights')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib/Seaborn not available for attention visualization")


def batch_decode_sequences(sequences: torch.Tensor, 
                         idx2word: Dict[int, str],
                         remove_special: bool = True) -> List[str]:
    """Decode a batch of sequences to text"""
    batch_size = sequences.shape[0]
    decoded_texts = []
    
    for i in range(batch_size):
        sequence = sequences[i].cpu().tolist()
        text = sequence_to_text(sequence, idx2word, remove_special)
        decoded_texts.append(text)
    
    return decoded_texts


class ProgressTracker:
    """Track training progress"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Update tracking metrics"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_perplexities.append(calculate_perplexity(train_loss))
        self.val_perplexities.append(calculate_perplexity(val_loss))
        self.learning_rates.append(lr)
    
    def save_progress(self, filepath: str):
        """Save progress to file"""
        progress_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'learning_rates': self.learning_rates
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"Training progress saved to {filepath}")
    
    def plot_progress(self, save_path: str = None):
        """Plot training progress"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss
            ax1.plot(self.epochs, self.train_losses, label='Train Loss')
            ax1.plot(self.epochs, self.val_losses, label='Val Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Perplexity
            ax2.plot(self.epochs, self.train_perplexities, label='Train Perplexity')
            ax2.plot(self.epochs, self.val_perplexities, label='Val Perplexity')
            ax2.set_title('Training and Validation Perplexity')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Perplexity')
            ax2.legend()
            ax2.grid(True)
            
            # Learning Rate
            ax3.plot(self.epochs, self.learning_rates)
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
            
            # Loss difference
            loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
            ax4.plot(self.epochs, loss_diff)
            ax4.set_title('Train-Val Loss Difference')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('|Train Loss - Val Loss|')
            ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Progress plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")