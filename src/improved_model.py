import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple, Optional
import numpy as np


class ImprovedAttention(nn.Module):
    """Improved attention mechanism with better normalization"""
    
    def __init__(self, hidden_size: int, attention_dim: int):
        super(ImprovedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        # Linear layers for attention computation
        self.encoder_projection = nn.Linear(hidden_size * 2, attention_dim)  # *2 for bidirectional
        self.decoder_projection = nn.Linear(hidden_size, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(attention_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        self.tanh = nn.Tanh()
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.encoder_projection, self.decoder_projection, self.attention_vector]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
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
        
        # Add projections and apply layer norm
        combined = self.layer_norm(encoder_proj + decoder_proj)  # [batch, seq_len, attn_dim]
        combined = self.dropout(combined)
        
        # Compute attention scores
        scores = self.attention_vector(self.tanh(combined))  # [batch, seq_len, 1]
        scores = scores.squeeze(2)  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask matches the sequence length
            mask = mask[:, :scores.size(1)]
            scores.masked_fill_(mask == 0, -float('inf'))
        
        # Compute attention weights with temperature scaling
        attention_weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden_size * 2]
        
        return context_vector, attention_weights


class ImprovedEncoder(nn.Module):
    """Improved Bidirectional LSTM Encoder with better regularization"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, dropout: float = 0.1, pad_idx: int = 0):
        super(ImprovedEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Embedding layer with padding index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Layer normalization for embeddings
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        
        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization for LSTM outputs
        self.output_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dropout layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
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
        
        # Embedding with normalization and dropout
        embedded = self.embedding(input_seq)  # [batch, seq_len, embedding_dim]
        embedded = self.embedding_norm(embedded)
        embedded = self.embedding_dropout(embedded)
        
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
        
        # Apply layer normalization and dropout
        outputs = self.output_norm(outputs)
        outputs = self.output_dropout(outputs)
        
        return outputs, hidden


class ImprovedDecoder(nn.Module):
    """Improved LSTM Decoder with better attention and regularization"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, attention_dim: int, dropout: float = 0.1, pad_idx: int = 0):
        super(ImprovedDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        
        # Attention mechanism
        self.attention = ImprovedAttention(hidden_size, attention_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_size * 2,  # embedding + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization for LSTM output
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        # Output projection with residual connection
        self.context_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Dropout layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
        
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
        
        # Embedding with normalization and dropout
        embedded = self.embedding(input_token)  # [batch, 1, embedding_dim]
        embedded = self.embedding_norm(embedded)
        embedded = self.embedding_dropout(embedded)
        
        # Get attention context using the last layer hidden state
        decoder_hidden = hidden[0][-1]  # [batch_size, hidden_size]
        context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden, encoder_mask)
        
        # Apply dropout to context
        context_vector = self.context_dropout(context_vector)
        
        # Concatenate embedded input with context vector
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)  # [batch, 1, embedding_dim + hidden_size * 2]
        
        # LSTM forward pass
        lstm_output, hidden = self.lstm(lstm_input, hidden)  # [batch, 1, hidden_size]
        lstm_output = self.lstm_norm(lstm_output)
        
        # Project context to same dimension as LSTM output for residual connection
        projected_context = self.context_projection(context_vector.unsqueeze(1))  # [batch, 1, hidden_size]
        
        # Residual connection
        combined_output = lstm_output + projected_context
        combined_output = self.output_dropout(combined_output)
        
        # Output projection
        output = self.output_projection(combined_output)  # [batch, 1, vocab_size]
        
        return output, hidden, attention_weights


class ImprovedSeq2SeqModel(nn.Module):
    """Improved Sequence-to-Sequence Model with better architecture"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embedding_dim: int, 
                 hidden_size: int, encoder_layers: int, decoder_layers: int, 
                 attention_dim: int, dropout: float = 0.1, 
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0, 
                 sos_idx: int = 2, eos_idx: int = 3):
        super(ImprovedSeq2SeqModel, self).__init__()
        
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
        self.encoder = ImprovedEncoder(src_vocab_size, embedding_dim, hidden_size, 
                                     encoder_layers, dropout, src_pad_idx)
        self.decoder = ImprovedDecoder(tgt_vocab_size, embedding_dim, hidden_size, 
                                     decoder_layers, attention_dim, dropout, tgt_pad_idx)
        
        # Bridge layers to transform encoder hidden state to decoder initial state
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization for bridge
        self.bridge_norm_h = nn.LayerNorm(hidden_size)
        self.bridge_norm_c = nn.LayerNorm(hidden_size)
        
        # Dropout for bridge
        self.bridge_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize bridge layers
        nn.init.xavier_uniform_(self.bridge_h.weight)
        nn.init.xavier_uniform_(self.bridge_c.weight)
        nn.init.zeros_(self.bridge_h.bias)
        nn.init.zeros_(self.bridge_c.bias)
    
    def _init_decoder_hidden(self, encoder_hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize decoder hidden state from encoder hidden state"""
        encoder_h, encoder_c = encoder_hidden
        
        # Combine forward and backward hidden states
        batch_size = encoder_h.shape[1]
        
        # Take the last layer of encoder (both directions)
        encoder_h_last = encoder_h[-2:].transpose(0, 1).contiguous()  # [batch, 2, hidden_size]
        encoder_c_last = encoder_c[-2:].transpose(0, 1).contiguous()  # [batch, 2, hidden_size]
        
        # Concatenate forward and backward states
        encoder_h_combined = encoder_h_last.view(batch_size, -1)  # [batch, hidden_size * 2]
        encoder_c_combined = encoder_c_last.view(batch_size, -1)  # [batch, hidden_size * 2]
        
        # Transform to decoder hidden size with normalization and dropout
        decoder_h = self.bridge_h(encoder_h_combined)  # [batch, hidden_size]
        decoder_c = self.bridge_c(encoder_c_combined)  # [batch, hidden_size]
        
        decoder_h = self.bridge_norm_h(decoder_h)
        decoder_c = self.bridge_norm_c(decoder_c)
        
        decoder_h = self.bridge_dropout(decoder_h)
        decoder_c = self.bridge_dropout(decoder_c)
        
        # Repeat for all decoder layers
        decoder_h = decoder_h.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_c = decoder_c.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        
        return (decoder_h, decoder_c)
    
    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5, src_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass during training with curriculum learning
        
        Args:
            src_seq: [batch_size, src_len]
            tgt_seq: [batch_size, tgt_len]
            teacher_forcing_ratio: probability of using teacher forcing
            src_lengths: [batch_size] - actual source sequence lengths
            
        Returns:
            outputs: [batch_size, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt_seq.shape
        
        # Create source mask
        src_mask = (src_seq != self.src_pad_idx).float()
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src_seq, src_lengths)
        
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
            
            # Teacher forcing with curriculum learning
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing and self.training:
                decoder_input = tgt_seq[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1)
        
        return outputs
    
    def generate(self, src_seq: torch.Tensor, max_length: int = 100, 
                 beam_size: int = 1, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate translation using greedy decoding or beam search
        
        Args:
            src_seq: [batch_size, src_len]
            max_length: maximum output length
            beam_size: beam size for beam search (1 = greedy)
            src_lengths: [batch_size] - actual source sequence lengths
            
        Returns:
            generated_seq: [batch_size, output_len]
            attention_weights: [batch_size, output_len, src_len]
        """
        self.eval()
        with torch.no_grad():
            return self._greedy_decode(src_seq, max_length, src_lengths)
    
    def _greedy_decode(self, src_seq: torch.Tensor, max_length: int, 
                      src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Improved greedy decoding with early stopping"""
        batch_size = src_seq.shape[0]
        device = src_seq.device
        
        # Create source mask
        src_mask = (src_seq != self.src_pad_idx).float()
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src_seq, src_lengths)
        
        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Initialize output sequence with SOS token
        generated_seq = torch.full((batch_size, 1), self.sos_idx, device=device)
        attention_weights_list = []
        
        # Track which sequences are still generating
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for step in range(max_length - 1):
            if not active_mask.any():
                break
                
            # Decoder forward pass
            output, decoder_hidden, attention_weights = self.decoder(
                generated_seq[:, -1:], decoder_hidden, encoder_outputs, src_mask
            )
            
            # Get next token with temperature sampling for better diversity
            probs = F.softmax(output.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, 1) if self.training else output.argmax(dim=-1)
            
            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            attention_weights_list.append(attention_weights)
            
            # Update active mask (stop generating for sequences that produced EOS)
            eos_produced = (next_token.squeeze(-1) == self.eos_idx)
            active_mask = active_mask & ~eos_produced
        
        # Stack attention weights
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            attention_weights = torch.zeros(batch_size, 1, src_seq.shape[1], device=device)
        
        return generated_seq, attention_weights
    
    def forward_with_attention(self, src_seq: torch.Tensor, tgt_seq: Optional[torch.Tensor] = None, 
                              teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both outputs and attention weights for analysis
        
        Args:
            src_seq: [batch_size, src_len]
            tgt_seq: [batch_size, tgt_len] (optional, for training)
            teacher_forcing_ratio: probability of using teacher forcing
            
        Returns:
            outputs: [batch_size, tgt_len, vocab_size]
            attention_weights: [batch_size, tgt_len, src_len]
        """
        if tgt_seq is not None:
            # Training mode - use provided target sequence
            batch_size = src_seq.shape[0]
            tgt_len = tgt_seq.shape[1]
            
            # Create source mask
            src_mask = (src_seq != self.src_pad_idx).float()
            
            # Encode source sequence
            encoder_outputs, encoder_hidden = self.encoder(src_seq)
            
            # Initialize decoder hidden state
            decoder_hidden = self._init_decoder_hidden(encoder_hidden)
            
            # Initialize outputs and attention weights
            outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src_seq.device)
            attention_weights_list = []
            
            # First decoder input is SOS token
            decoder_input = tgt_seq[:, 0:1]
            
            # Decode step by step
            for t in range(1, tgt_len):
                # Decoder forward pass
                output, decoder_hidden, attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, src_mask
                )
                
                # Store output and attention
                outputs[:, t:t+1] = output
                attention_weights_list.append(attention_weights.squeeze(1))
                
                # Teacher forcing decision
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing and self.training:
                    decoder_input = tgt_seq[:, t:t+1]
                else:
                    decoder_input = output.argmax(dim=-1)
            
            # Stack attention weights
            if attention_weights_list:
                attention_weights = torch.stack(attention_weights_list, dim=1)
            else:
                attention_weights = torch.zeros(batch_size, tgt_len-1, src_seq.shape[1], device=src_seq.device)
            
            return outputs, attention_weights
        
        else:
            # Inference mode - generate sequence
            return self.generate(src_seq, max_length=100)


def create_improved_model(config: dict, src_vocab_size: int, tgt_vocab_size: int) -> ImprovedSeq2SeqModel:
    """Create improved model from configuration"""
    model_config = config['model']
    
    model = ImprovedSeq2SeqModel(
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
    # Test improved model creation
    import json
    
    with open('../config.json', 'r') as f:
        config = json.load(f)
    
    # Create improved model
    model = create_improved_model(config, src_vocab_size=100, tgt_vocab_size=80)
    
    print(f"Improved model created with {count_parameters(model):,} trainable parameters")
    print("\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 15
    
    src_seq = torch.randint(1, 100, (batch_size, src_len))  # Avoid pad token
    tgt_seq = torch.randint(1, 80, (batch_size, tgt_len))   # Avoid pad token
    src_lengths = torch.randint(10, src_len, (batch_size,))
    
    # Forward pass
    outputs = model(src_seq, tgt_seq, teacher_forcing_ratio=0.7, src_lengths=src_lengths)
    print(f"\nOutput shape: {outputs.shape}")
    
    # Generation test
    generated_seq, attention_weights = model.generate(src_seq[:1], max_length=20, src_lengths=src_lengths[:1])
    print(f"Generated sequence shape: {generated_seq.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")