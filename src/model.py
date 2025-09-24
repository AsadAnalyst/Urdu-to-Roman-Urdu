import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple, Optional


class Attention(nn.Module):
    """Additive attention mechanism for seq2seq model"""
    
    def __init__(self, hidden_size: int, attention_dim: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        # Linear layers for attention computation
        self.encoder_projection = nn.Linear(hidden_size * 2, attention_dim)  # *2 for bidirectional
        self.decoder_projection = nn.Linear(hidden_size, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention mechanism
        
        Args:
            encoder_outputs: [batch_size, seq_len, hidden_size * 2]
            decoder_hidden: [batch_size, hidden_size]
            mask: [batch_size, seq_len] - mask for padding tokens
            
        Returns:
            context_vector: [batch_size, hidden_size * 2]
            attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Project encoder outputs and decoder hidden state
        encoder_proj = self.encoder_projection(encoder_outputs)  # [batch, seq_len, attn_dim]
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)  # [batch, 1, attn_dim]
        
        # Compute attention scores
        scores = self.attention_vector(self.tanh(encoder_proj + decoder_proj))  # [batch, seq_len, 1]
        scores = scores.squeeze(2)  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -float('inf'))
        
        # Compute attention weights
        attention_weights = self.softmax(scores)  # [batch, seq_len]
        
        # Compute context vector
        context_vector = torch.sum(encoder_outputs * attention_weights.unsqueeze(2), dim=1)  # [batch, hidden_size * 2]
        
        return context_vector, attention_weights


class Encoder(nn.Module):
    """Bidirectional LSTM Encoder"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, dropout: float = 0.1, pad_idx: int = 0):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Embedding layer with padding index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through encoder
        
        Args:
            input_seq: [batch_size, seq_len]
            input_lengths: [batch_size] - actual lengths of sequences
            
        Returns:
            outputs: [batch_size, seq_len, hidden_size * 2]
            hidden: tuple of (h_n, c_n) each [num_layers * 2, batch_size, hidden_size]
        """
        batch_size, seq_len = input_seq.shape
        
        # Embedding
        embedded = self.embedding(input_seq)  # [batch, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack sequences if lengths are provided
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        outputs, hidden = self.lstm(embedded)
        
        # Unpack sequences if they were packed
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class Decoder(nn.Module):
    """LSTM Decoder with Attention"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, attention_dim: int, dropout: float = 0.1, pad_idx: int = 0):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Attention mechanism
        self.attention = Attention(hidden_size, attention_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_size * 2,  # embedding + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.out_projection = nn.Linear(hidden_size + hidden_size * 2, vocab_size)  # hidden + context
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor, encoder_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for one decoding step
        
        Args:
            input_token: [batch_size, 1]
            hidden: tuple of (h, c) each [num_layers, batch_size, hidden_size]
            encoder_outputs: [batch_size, seq_len, hidden_size * 2]
            encoder_mask: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, 1, vocab_size]
            hidden: tuple of (h, c) each [num_layers, batch_size, hidden_size]
            attention_weights: [batch_size, seq_len]
        """
        batch_size = input_token.shape[0]
        
        # Embedding
        embedded = self.embedding(input_token)  # [batch, 1, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Get attention context using the last layer hidden state
        decoder_hidden = hidden[0][-1]  # [batch_size, hidden_size]
        context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden, encoder_mask)
        
        # Concatenate embedded input with context vector
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)  # [batch, 1, embedding_dim + hidden_size * 2]
        
        # LSTM forward pass
        lstm_output, hidden = self.lstm(lstm_input, hidden)  # [batch, 1, hidden_size]
        
        # Concatenate LSTM output with context for final projection
        output_input = torch.cat([lstm_output, context_vector.unsqueeze(1)], dim=2)  # [batch, 1, hidden_size + hidden_size * 2]
        
        # Output projection
        output = self.out_projection(output_input)  # [batch, 1, vocab_size]
        
        return output, hidden, attention_weights


class Seq2SeqModel(nn.Module):
    """Complete Sequence-to-Sequence Model with BiLSTM Encoder and LSTM Decoder"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embedding_dim: int, 
                 hidden_size: int, encoder_layers: int, decoder_layers: int, 
                 attention_dim: int, dropout: float = 0.1, 
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0, 
                 sos_idx: int = 2, eos_idx: int = 3):
        super(Seq2SeqModel, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        # Encoder and Decoder
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size, 
                              encoder_layers, dropout, src_pad_idx)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_size, 
                              decoder_layers, attention_dim, dropout, tgt_pad_idx)
        
        # Bridge layers to transform encoder hidden state to decoder initial state
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def _init_decoder_hidden(self, encoder_hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize decoder hidden state from encoder hidden state"""
        encoder_h, encoder_c = encoder_hidden
        
        # Combine forward and backward hidden states
        # encoder_h/c shape: [num_layers * 2, batch, hidden_size]
        batch_size = encoder_h.shape[1]
        
        # Take the last layer of encoder (both directions)
        encoder_h_last = encoder_h[-2:].transpose(0, 1).contiguous()  # [batch, 2, hidden_size]
        encoder_c_last = encoder_c[-2:].transpose(0, 1).contiguous()  # [batch, 2, hidden_size]
        
        # Concatenate forward and backward states
        encoder_h_combined = encoder_h_last.view(batch_size, -1)  # [batch, hidden_size * 2]
        encoder_c_combined = encoder_c_last.view(batch_size, -1)  # [batch, hidden_size * 2]
        
        # Transform to decoder hidden size
        decoder_h = self.bridge_h(encoder_h_combined)  # [batch, hidden_size]
        decoder_c = self.bridge_c(encoder_c_combined)  # [batch, hidden_size]
        
        # Repeat for all decoder layers
        decoder_h = decoder_h.unsqueeze(0).repeat(self.decoder_layers, 1, 1)  # [decoder_layers, batch, hidden_size]
        decoder_c = decoder_c.unsqueeze(0).repeat(self.decoder_layers, 1, 1)  # [decoder_layers, batch, hidden_size]
        
        return (decoder_h, decoder_c)
    
    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass during training
        
        Args:
            src_seq: [batch_size, src_len]
            tgt_seq: [batch_size, tgt_len]
            teacher_forcing_ratio: probability of using teacher forcing
            
        Returns:
            outputs: [batch_size, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt_seq.shape
        
        # Create source mask
        src_mask = (src_seq != self.src_pad_idx).float()
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        
        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src_seq.device)
        
        # First decoder input is SOS token
        decoder_input = tgt_seq[:, 0:1]  # [batch_size, 1]
        
        # Decode step by step
        for t in range(1, tgt_len):
            # Decoder forward pass
            output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, src_mask
            )
            
            # Store output
            outputs[:, t:t+1] = output
            
            # Teacher forcing: use ground truth or predicted token
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt_seq[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1)
        
        return outputs
    
    def generate(self, src_seq: torch.Tensor, max_length: int = 100, 
                 beam_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate translation using greedy decoding or beam search
        
        Args:
            src_seq: [batch_size, src_len]
            max_length: maximum output length
            beam_size: beam size for beam search (1 = greedy)
            
        Returns:
            generated_seq: [batch_size, output_len]
            attention_weights: [batch_size, output_len, src_len]
        """
        self.eval()
        with torch.no_grad():
            if beam_size == 1:
                return self._greedy_decode(src_seq, max_length)
            else:
                return self._beam_search(src_seq, max_length, beam_size)
    
    def _greedy_decode(self, src_seq: torch.Tensor, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding"""
        batch_size = src_seq.shape[0]
        device = src_seq.device
        
        # Create source mask
        src_mask = (src_seq != self.src_pad_idx).float()
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        
        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Initialize output sequence with SOS token
        generated_seq = torch.full((batch_size, 1), self.sos_idx, device=device)
        attention_weights_list = []
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Decoder forward pass
            output, decoder_hidden, attention_weights = self.decoder(
                generated_seq[:, -1:], decoder_hidden, encoder_outputs, src_mask
            )
            
            # Get next token
            next_token = output.argmax(dim=-1)
            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            attention_weights_list.append(attention_weights)
            
            # Check if all sequences have produced EOS token
            if (next_token == self.eos_idx).all():
                break
        
        # Stack attention weights
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            attention_weights = torch.zeros(batch_size, 1, src_seq.shape[1], device=device)
        
        return generated_seq, attention_weights
    
    def _beam_search(self, src_seq: torch.Tensor, max_length: int, beam_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding - simplified implementation"""
        # For now, return greedy decoding
        # TODO: Implement full beam search
        return self._greedy_decode(src_seq, max_length)


def create_model(config: dict, src_vocab_size: int, tgt_vocab_size: int) -> Seq2SeqModel:
    """Create model from configuration"""
    model_config = config['model']
    
    model = Seq2SeqModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=model_config['embedding_dim'],
        hidden_size=model_config['hidden_size'],
        encoder_layers=model_config['encoder_layers'],
        decoder_layers=model_config['decoder_layers'],
        attention_dim=model_config['attention_dim'],
        dropout=model_config['dropout']
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    import json
    
    with open('../config.json', 'r') as f:
        config = json.load(f)
    
    # Create dummy model
    model = create_model(config, src_vocab_size=100, tgt_vocab_size=80)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    print("\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 15
    
    src_seq = torch.randint(0, 100, (batch_size, src_len))
    tgt_seq = torch.randint(0, 80, (batch_size, tgt_len))
    
    # Forward pass
    outputs = model(src_seq, tgt_seq)
    print(f"\nOutput shape: {outputs.shape}")
    
    # Generation test
    generated_seq, attention_weights = model.generate(src_seq[:1], max_length=20)
    print(f"Generated sequence shape: {generated_seq.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")