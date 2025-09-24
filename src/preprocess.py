import os
import json
import re
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import unicodedata

class UrduRomanDataset:
    """Dataset class for Urdu to Roman Urdu transliteration"""
    
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.data_dir = self.config['paths']['data_dir']
        self.urdu_text = []
        self.roman_text = []
        
        # Special tokens
        self.SOS_token = '<SOS>'
        self.EOS_token = '<EOS>'
        self.PAD_token = '<PAD>'
        self.UNK_token = '<UNK>'
        
        # Vocabularies
        self.urdu_vocab = None
        self.roman_vocab = None
        self.urdu_word2idx = None
        self.urdu_idx2word = None
        self.roman_word2idx = None
        self.roman_idx2word = None
        
    def normalize_urdu_text(self, text):
        """Clean and normalize Urdu text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra punctuation except basic ones
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\.\,\:\;\!\?\'\"]', '', text)
        
        return text.strip()
    
    def normalize_roman_text(self, text):
        """Clean and normalize Roman Urdu text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Keep only alphanumeric characters and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\:\;\!\?\'\"\-]', '', text)
        
        # Remove diacritics commonly used in roman urdu transliteration
        text = re.sub(r'[āīūēōḥṇṭḍṛṣẓġñḳḷṯẇȳḇṕṽźĵč]', lambda m: {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
            'ḥ': 'h', 'ṇ': 'n', 'ṭ': 't', 'ḍ': 'd', 'ṛ': 'r',
            'ṣ': 's', 'ẓ': 'z', 'ġ': 'gh', 'ñ': 'n', 'ḳ': 'k',
            'ḷ': 'l', 'ṯ': 't', 'ẇ': 'w', 'ȳ': 'y', 'ḇ': 'b',
            'ṕ': 'p', 'ṽ': 'v', 'ź': 'z', 'ĵ': 'j', 'č': 'ch'
        }.get(m.group(0), m.group(0)), text)
        
        return text.strip()
    
    def load_data(self):
        """Load data from the urdu_ghazals_rekhta dataset"""
        print("Loading Urdu-Roman data from dataset...")
        
        dataset_path = os.path.join(self.data_dir, "dataset", "dataset")
        poets = os.listdir(dataset_path)
        
        for poet in tqdm(poets, desc="Processing poets"):
            poet_path = os.path.join(dataset_path, poet)
            if not os.path.isdir(poet_path):
                continue
                
            urdu_path = os.path.join(poet_path, "ur")
            roman_path = os.path.join(poet_path, "en")
            
            if not (os.path.exists(urdu_path) and os.path.exists(roman_path)):
                continue
                
            urdu_files = os.listdir(urdu_path)
            roman_files = os.listdir(roman_path)
            
            # Match files
            for urdu_file in urdu_files:
                if urdu_file in roman_files:
                    try:
                        # Read Urdu text
                        with open(os.path.join(urdu_path, urdu_file), 'r', encoding='utf-8') as f:
                            urdu_content = f.read().strip()
                        
                        # Read Roman text
                        with open(os.path.join(roman_path, urdu_file), 'r', encoding='utf-8') as f:
                            roman_content = f.read().strip()
                        
                        # Split into lines and process
                        urdu_lines = [line.strip() for line in urdu_content.split('\n') if line.strip()]
                        roman_lines = [line.strip() for line in roman_content.split('\n') if line.strip()]
                        
                        # Only process if we have matching number of lines
                        if len(urdu_lines) == len(roman_lines):
                            for urdu_line, roman_line in zip(urdu_lines, roman_lines):
                                # Clean and normalize
                                urdu_clean = self.normalize_urdu_text(urdu_line)
                                roman_clean = self.normalize_roman_text(roman_line)
                                
                                # Skip very short or long sequences
                                if (5 <= len(urdu_clean) <= 200 and 
                                    5 <= len(roman_clean) <= 200):
                                    self.urdu_text.append(urdu_clean)
                                    self.roman_text.append(roman_clean)
                                    
                    except Exception as e:
                        print(f"Error processing {poet}/{urdu_file}: {e}")
                        continue
        
        print(f"Loaded {len(self.urdu_text)} parallel sentences")
        return self.urdu_text, self.roman_text
    
    def tokenize_text(self, texts, tokenization_type="char"):
        """Tokenize text based on specified method"""
        if tokenization_type == "char":
            return [list(text) for text in texts]
        elif tokenization_type == "word":
            return [text.split() for text in texts]
        else:
            raise ValueError("Tokenization type must be 'char' or 'word'")
    
    def build_vocabulary(self, tokenized_texts, max_vocab_size=None, min_freq=1):
        """Build vocabulary from tokenized texts"""
        # Count token frequencies
        token_counts = Counter()
        for tokens in tokenized_texts:
            token_counts.update(tokens)
        
        # Filter by minimum frequency
        vocab = [token for token, count in token_counts.items() if count >= min_freq]
        
        # Limit vocabulary size
        if max_vocab_size:
            vocab = vocab[:max_vocab_size-4]  # Reserve space for special tokens
        
        # Add special tokens
        vocab = [self.PAD_token, self.UNK_token, self.SOS_token, self.EOS_token] + vocab
        
        # Create mappings
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        
        return vocab, word2idx, idx2word
    
    def text_to_sequence(self, tokens, word2idx, max_length=None):
        """Convert tokenized text to sequence of indices"""
        sequence = [word2idx.get(token, word2idx[self.UNK_token]) for token in tokens]
        
        # Add SOS and EOS tokens
        sequence = [word2idx[self.SOS_token]] + sequence + [word2idx[self.EOS_token]]
        
        # Pad or truncate to max_length
        if max_length:
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                sequence.extend([word2idx[self.PAD_token]] * (max_length - len(sequence)))
        
        return sequence
    
    def preprocess_and_split(self):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        if not self.urdu_text:
            raise ValueError("No data loaded. Check dataset path and format.")
        
        # Tokenize
        tokenization_type = self.config['data']['tokenization']
        print(f"Tokenizing with {tokenization_type}-level tokenization...")
        
        urdu_tokenized = self.tokenize_text(self.urdu_text, tokenization_type)
        roman_tokenized = self.tokenize_text(self.roman_text, tokenization_type)
        
        # Build vocabularies
        print("Building vocabularies...")
        max_vocab = self.config['data']['vocab_size_limit']
        min_freq = self.config['data']['min_freq']
        
        self.urdu_vocab, self.urdu_word2idx, self.urdu_idx2word = self.build_vocabulary(
            urdu_tokenized, max_vocab, min_freq)
        
        self.roman_vocab, self.roman_word2idx, self.roman_idx2word = self.build_vocabulary(
            roman_tokenized, max_vocab, min_freq)
        
        print(f"Urdu vocabulary size: {len(self.urdu_vocab)}")
        print(f"Roman vocabulary size: {len(self.roman_vocab)}")
        
        # Convert to sequences
        print("Converting to sequences...")
        max_length = self.config['model']['max_sequence_length']
        
        urdu_sequences = [self.text_to_sequence(tokens, self.urdu_word2idx, max_length) 
                         for tokens in tqdm(urdu_tokenized, desc="Converting Urdu")]
        
        roman_sequences = [self.text_to_sequence(tokens, self.roman_word2idx, max_length) 
                          for tokens in tqdm(roman_tokenized, desc="Converting Roman")]
        
        # Create dataset
        data_pairs = list(zip(urdu_sequences, roman_sequences))
        
        # Train/validation/test split
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        # First split: train + val vs test
        train_val, test_data = train_test_split(
            data_pairs, test_size=(1 - train_split - val_split), random_state=42)
        
        # Second split: train vs val
        train_data, val_data = train_test_split(
            train_val, test_size=(val_split / (train_split + val_split)), random_state=42)
        
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'urdu_vocab': {
                'vocab': self.urdu_vocab,
                'word2idx': self.urdu_word2idx,
                'idx2word': self.urdu_idx2word
            },
            'roman_vocab': {
                'vocab': self.roman_vocab,
                'word2idx': self.roman_word2idx,
                'idx2word': self.roman_idx2word
            }
        }
    
    def save_preprocessed_data(self, data, save_dir="data/processed"):
        """Save preprocessed data to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save train/val/test splits
        for split_name, split_data in [('train', data['train']), 
                                      ('val', data['val']), 
                                      ('test', data['test'])]:
            torch.save(split_data, os.path.join(save_dir, f"{split_name}_data.pt"))
        
        # Save vocabularies
        with open(os.path.join(save_dir, "vocabularies.pkl"), 'wb') as f:
            pickle.dump({
                'urdu_vocab': data['urdu_vocab'],
                'roman_vocab': data['roman_vocab']
            }, f)
        
        # Save metadata
        metadata = {
            'urdu_vocab_size': len(data['urdu_vocab']['vocab']),
            'roman_vocab_size': len(data['roman_vocab']['vocab']),
            'train_size': len(data['train']),
            'val_size': len(data['val']),
            'test_size': len(data['test']),
            'tokenization': self.config['data']['tokenization'],
            'max_sequence_length': self.config['model']['max_sequence_length']
        }
        
        with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessed data saved to {save_dir}")


class Seq2SeqDataset(Dataset):
    """PyTorch Dataset for sequence-to-sequence training"""
    
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        src_seq, tgt_seq = self.data_pairs[idx]
        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    src_seqs, tgt_seqs = zip(*batch)
    
    # Convert to tensors and stack
    src_batch = torch.stack(src_seqs)
    tgt_batch = torch.stack(tgt_seqs)
    
    return src_batch, tgt_batch


def main():
    """Main preprocessing function"""
    print("Starting Urdu to Roman Urdu preprocessing...")
    
    # Initialize dataset processor
    dataset = UrduRomanDataset()
    
    # Preprocess data
    preprocessed_data = dataset.preprocess_and_split()
    
    # Save processed data
    dataset.save_preprocessed_data(preprocessed_data)
    
    # Create sample DataLoader for testing
    train_dataset = Seq2SeqDataset(preprocessed_data['train'][:100])  # Sample for testing
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    print("\nSample batch from DataLoader:")
    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        print(f"Source batch shape: {src_batch.shape}")
        print(f"Target batch shape: {tgt_batch.shape}")
        print(f"Sample source sequence: {src_batch[0][:20]}...")  # First 20 tokens
        print(f"Sample target sequence: {tgt_batch[0][:20]}...")  # First 20 tokens
        break
    
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()