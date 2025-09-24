import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import unicodedata


class UrduRomanDataCleaner:
    """Advanced data cleaning for Urdu-Roman transliteration pairs"""
    
    def __init__(self):
        self.urdu_to_roman_mapping = {
            # Basic consonants
            'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
            'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
            'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z',
            'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z',
            'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f',
            'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm',
            'ن': 'n', 'و': 'o', 'ہ': 'h', 'ھ': 'h', 'ی': 'y',
            'ے': 'e',
            
            # Vowels and diacritics
            'ا': 'a', 'آ': 'aa', 'ؤ': 'o', 'ئ': 'y', 'ء': '',
            
            # Additional mappings
            'ّ': '',  # Shadda (gemination)
            'ً': 'an', 'ٌ': 'un', 'ٍ': 'in',
            'َ': 'a', 'ُ': 'u', 'ِ': 'i',
            'ْ': '',  # Sukun (no vowel)
            'ٰ': 'aa',  # Alif superscript
        }
        
        # Common Urdu words and their Roman equivalents for validation
        self.common_words = {
            'اور': 'aur', 'ہے': 'hai', 'کے': 'ke', 'میں': 'me',
            'کو': 'ko', 'سے': 'se', 'پر': 'par', 'کا': 'ka',
            'کی': 'ki', 'نے': 'ne', 'تھا': 'tha', 'ہو': 'ho',
            'گا': 'ga', 'گی': 'gi', 'گے': 'ge', 'رہا': 'raha',
            'رہی': 'rahi', 'رہے': 'rahe', 'کیا': 'kya', 'کوئی': 'koi'
        }
    
    def normalize_urdu_text(self, text: str) -> str:
        """Normalize Urdu text"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize some common variations
        text = text.replace('ي', 'ی')  # Normalize yeh
        text = text.replace('ك', 'ک')  # Normalize kaf
        text = text.replace('ء', 'ء')  # Normalize hamza
        
        # Remove unwanted characters but keep essential punctuation
        allowed_chars = set('ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھیےءآؤئًٌٍَُِّْٰ ۔،؍؎')
        text = ''.join(c for c in text if c in allowed_chars)
        
        return text.strip()
    
    def normalize_roman_text(self, text: str) -> str:
        """Normalize Roman text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = text.replace('-', '-')  # Normalize hyphens
        text = text.replace('.', '.')   # Keep periods for abbreviations
        
        # Remove unwanted characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyz -.')
        text = ''.join(c for c in text if c in allowed_chars)
        
        return text.strip()
    
    def validate_pair(self, urdu: str, roman: str) -> bool:
        """Validate if Urdu-Roman pair is reasonable"""
        if not urdu or not roman:
            return False
        
        # Check length ratio (Roman should not be more than 3x Urdu length)
        if len(roman) > 3 * len(urdu) or len(urdu) > 2 * len(roman):
            return False
        
        # Check for excessive repetition
        if self._has_excessive_repetition(roman):
            return False
        
        # Check for valid character ratios
        urdu_chars = len([c for c in urdu if c.strip()])
        roman_chars = len([c for c in roman if c.strip()])
        
        if roman_chars == 0 or urdu_chars == 0:
            return False
        
        # Check for reasonable character ratio
        ratio = roman_chars / urdu_chars
        if ratio < 0.3 or ratio > 4.0:
            return False
        
        return True
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check if text has excessive character repetition"""
        if len(text) < 3:
            return False
        
        # Check for 4+ consecutive identical characters
        if re.search(r'(.)\1{3,}', text):
            return True
        
        # Check for excessive repetition of character patterns
        repetition_ratio = len(set(text)) / len(text) if text else 1
        return repetition_ratio < 0.3
    
    def create_mapping_from_data(self, urdu_texts: List[str], roman_texts: List[str]) -> Dict[str, str]:
        """Create character mapping from parallel data"""
        char_mappings = {}
        
        for urdu, roman in zip(urdu_texts, roman_texts):
            if len(urdu) == len(roman):  # Simple 1:1 alignment for same length
                for u_char, r_char in zip(urdu, roman):
                    if u_char != ' ' and r_char != ' ':
                        if u_char not in char_mappings:
                            char_mappings[u_char] = Counter()
                        char_mappings[u_char][r_char] += 1
        
        # Select most frequent mapping for each character
        final_mapping = {}
        for u_char, r_counter in char_mappings.items():
            if r_counter:
                final_mapping[u_char] = r_counter.most_common(1)[0][0]
        
        return final_mapping
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the entire dataset"""
        print(f"Original dataset size: {len(df)}")
        
        # Normalize texts
        df['urdu'] = df['urdu'].apply(self.normalize_urdu_text)
        df['roman'] = df['roman'].apply(self.normalize_roman_text)
        
        # Remove empty pairs
        df = df[(df['urdu'].str.len() > 0) & (df['roman'].str.len() > 0)]
        print(f"After removing empty pairs: {len(df)}")
        
        # Validate pairs
        valid_mask = df.apply(lambda row: self.validate_pair(row['urdu'], row['roman']), axis=1)
        df = df[valid_mask]
        print(f"After validation: {len(df)}")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['urdu', 'roman'])
        print(f"After removing duplicates: {len(df)}")
        
        # Sort by length for better training
        df = df.sort_values(by='urdu', key=lambda x: x.str.len())
        
        return df.reset_index(drop=True)


class ImprovedVocabularyBuilder:
    """Improved vocabulary builder with better handling"""
    
    def __init__(self, tokenization: str = "char"):
        self.tokenization = tokenization
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<SOS>': 2,
            '<EOS>': 3
        }
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 1) -> Dict[str, int]:
        """Build vocabulary from texts"""
        if self.tokenization == "char":
            return self._build_char_vocab(texts, min_freq)
        else:
            return self._build_word_vocab(texts, min_freq)
    
    def _build_char_vocab(self, texts: List[str], min_freq: int) -> Dict[str, int]:
        """Build character-level vocabulary"""
        char_counts = Counter()
        
        for text in texts:
            char_counts.update(text)
        
        # Filter by frequency
        filtered_chars = {char for char, count in char_counts.items() if count >= min_freq}
        
        # Create vocabulary
        vocab = self.special_tokens.copy()
        for char in sorted(filtered_chars):
            if char not in vocab:
                vocab[char] = len(vocab)
        
        return vocab
    
    def _build_word_vocab(self, texts: List[str], min_freq: int) -> Dict[str, int]:
        """Build word-level vocabulary"""
        word_counts = Counter()
        
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Filter by frequency
        filtered_words = {word for word, count in word_counts.items() if count >= min_freq}
        
        # Create vocabulary
        vocab = self.special_tokens.copy()
        for word in sorted(filtered_words):
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def tokenize(self, text: str, vocab: Dict[str, int]) -> List[int]:
        """Tokenize text using vocabulary"""
        if self.tokenization == "char":
            tokens = list(text)
        else:
            tokens = text.split()
        
        # Convert to indices
        indices = []
        for token in tokens:
            indices.append(vocab.get(token, vocab['<UNK>']))
        
        return indices
    
    def detokenize(self, indices: List[int], vocab: Dict[str, int]) -> str:
        """Convert indices back to text"""
        # Create reverse vocabulary
        idx_to_token = {idx: token for token, idx in vocab.items()}
        
        # Convert indices to tokens
        tokens = []
        for idx in indices:
            token = idx_to_token.get(idx, '<UNK>')
            if token not in self.special_tokens:
                tokens.append(token)
        
        # Join tokens
        if self.tokenization == "char":
            return ''.join(tokens)
        else:
            return ' '.join(tokens)


class ImprovedSeq2SeqDataset(Dataset):
    """Improved dataset with better preprocessing"""
    
    def __init__(self, urdu_texts: List[str], roman_texts: List[str], 
                 urdu_vocab: Dict[str, int], roman_vocab: Dict[str, int],
                 tokenization: str = "char", max_length: int = 100):
        
        self.urdu_texts = urdu_texts
        self.roman_texts = roman_texts
        self.urdu_vocab = urdu_vocab
        self.roman_vocab = roman_vocab
        self.tokenization = tokenization
        self.max_length = max_length
        
        self.vocab_builder = ImprovedVocabularyBuilder(tokenization)
        
        # Pre-tokenize all texts
        self.urdu_sequences = []
        self.roman_sequences = []
        
        for urdu, roman in zip(urdu_texts, roman_texts):
            urdu_tokens = self.vocab_builder.tokenize(urdu, urdu_vocab)
            roman_tokens = self.vocab_builder.tokenize(roman, roman_vocab)
            
            # Add SOS and EOS tokens to target
            roman_tokens = [roman_vocab['<SOS>']] + roman_tokens + [roman_vocab['<EOS>']]
            
            # Skip sequences that are too long
            if len(urdu_tokens) <= max_length and len(roman_tokens) <= max_length:
                self.urdu_sequences.append(urdu_tokens)
                self.roman_sequences.append(roman_tokens)
    
    def __len__(self):
        return len(self.urdu_sequences)
    
    def __getitem__(self, idx):
        return {
            'urdu': torch.tensor(self.urdu_sequences[idx], dtype=torch.long),
            'roman': torch.tensor(self.roman_sequences[idx], dtype=torch.long),
            'urdu_length': len(self.urdu_sequences[idx]),
            'roman_length': len(self.roman_sequences[idx])
        }


def improved_collate_fn(batch):
    """Improved collate function with proper padding"""
    # Sort batch by source length (descending) for better packing
    batch = sorted(batch, key=lambda x: x['urdu_length'], reverse=True)
    
    # Get lengths
    urdu_lengths = torch.tensor([item['urdu_length'] for item in batch])
    roman_lengths = torch.tensor([item['roman_length'] for item in batch])
    
    # Pad sequences
    max_urdu_len = max(urdu_lengths)
    max_roman_len = max(roman_lengths)
    
    urdu_seqs = torch.zeros(len(batch), max_urdu_len, dtype=torch.long)
    roman_seqs = torch.zeros(len(batch), max_roman_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        urdu_len = item['urdu_length']
        roman_len = item['roman_length']
        
        urdu_seqs[i, :urdu_len] = item['urdu']
        roman_seqs[i, :roman_len] = item['roman']
    
    return {
        'src': urdu_seqs,
        'tgt': roman_seqs,
        'src_lengths': urdu_lengths,
        'tgt_lengths': roman_lengths
    }


def create_improved_datasets(data_path: str, config: dict) -> Tuple[Dataset, Dataset, Dataset, Dict, Dict]:
    """Create improved datasets with better preprocessing"""
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = pd.read_csv(data_path)
    
    # Initialize cleaner and clean data
    cleaner = UrduRomanDataCleaner()
    df = cleaner.clean_dataset(df)
    
    print(f"Final cleaned dataset size: {len(df)}")
    
    # Split data
    train_size = int(len(df) * config['data']['train_split'])
    val_size = int(len(df) * config['data']['val_split'])
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Build vocabularies
    print("Building vocabularies...")
    vocab_builder = ImprovedVocabularyBuilder(config['data']['tokenization'])
    
    urdu_vocab = vocab_builder.build_vocabulary(
        train_df['urdu'].tolist(), 
        config['data']['min_freq']
    )
    roman_vocab = vocab_builder.build_vocabulary(
        train_df['roman'].tolist(), 
        config['data']['min_freq']
    )
    
    print(f"Urdu vocabulary size: {len(urdu_vocab)}")
    print(f"Roman vocabulary size: {len(roman_vocab)}")
    
    # Create datasets
    train_dataset = ImprovedSeq2SeqDataset(
        train_df['urdu'].tolist(), train_df['roman'].tolist(),
        urdu_vocab, roman_vocab, config['data']['tokenization'],
        config['model']['max_sequence_length']
    )
    
    val_dataset = ImprovedSeq2SeqDataset(
        val_df['urdu'].tolist(), val_df['roman'].tolist(),
        urdu_vocab, roman_vocab, config['data']['tokenization'],
        config['model']['max_sequence_length']
    )
    
    test_dataset = ImprovedSeq2SeqDataset(
        test_df['urdu'].tolist(), test_df['roman'].tolist(),
        urdu_vocab, roman_vocab, config['data']['tokenization'],
        config['model']['max_sequence_length']
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, urdu_vocab, roman_vocab


if __name__ == "__main__":
    # Test the improved preprocessing
    import json
    
    with open('../config.json', 'r') as f:
        config = json.load(f)
    
    # Test data cleaning
    test_data = [
        {"urdu": "میں نے کیا", "roman": "mai ne kya"},
        {"urdu": "آپ کیسے ہیں", "roman": "aap kaise hai"},
        {"urdu": "بہت اچھا", "roman": "bohat acha"}
    ]
    
    df = pd.DataFrame(test_data)
    cleaner = UrduRomanDataCleaner()
    cleaned_df = cleaner.clean_dataset(df)
    
    print("Original data:")
    print(df)
    print("\nCleaned data:")
    print(cleaned_df)