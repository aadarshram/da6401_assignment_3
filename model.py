"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Google Drive file id for the released pretrained checkpoint (.pt)
_PRETRAINED_CHECKPOINT_DRIVE_ID = "1eQ4_uIMU-3cmsSQvc_NhWlQ-qNEbem93"

# Multi30k vocab sizes (bentrevett/multi30k, spacy blank tokenizers, train split)
MULTI30K_SRC_VOCAB_SIZE = 18_669  # German (de)
MULTI30K_TGT_VOCAB_SIZE = 9_797   # English (en)
INFER_MAX_LEN = 40


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # Find score
    score = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        score = score / math.sqrt(Q.size(-1))
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.device != score.device:
            mask = mask.to(score.device)
        score = score.masked_fill(mask, float('-1e9'))

    # Attention weights
    attn_w = F.softmax(score, dim=-1) # shape: (..., seq_q, seq_k)

    # Apply attention weights to values
    output = torch.matmul(attn_w, V) # shape: (..., seq_q, d_v)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # shape: [batch, 1, 1, src_len]
    return mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    # Padding mask
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # shape: [batch, 1, 1, tgt_len]
    # Causal mask (allocated on same device as tgt)
    seq_len = tgt.size(-1)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).view(1, 1, seq_len, seq_len)  # shape: [1, 1, tgt_len, tgt_len]

    # Combine masks
    mask = pad_mask | causal_mask # shape: [batch, 1, tgt_len, tgt_len], True if either pad or causal mask is True

    return mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.d_v       = d_model // num_heads   # depth per head (same as d_k)
        self.scale_attention = scale_attention
        self.dropout = nn.Dropout(p=dropout)

        # Learnable projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self._incremental = False
        self._kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def start_incremental(self) -> None:
        """Enable KV caching for fast autoregressive inference (infer only)."""
        self._incremental = True
        self._kv_cache = None

    def stop_incremental(self) -> None:
        self._incremental = False
        self._kv_cache = None

    def _project(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        batch_size = x.size(0)
        return self.dropout(
            linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        Q = self._project(query, self.W_Q)
        is_self_attn = query is key

        if self._incremental and self._kv_cache is not None:
            if is_self_attn:
                K_new = self._project(key, self.W_K)
                V_new = self._project(value, self.W_V)
                K = torch.cat([self._kv_cache[0], K_new], dim=2)
                V = torch.cat([self._kv_cache[1], V_new], dim=2)
                self._kv_cache = (K, V)
            else:
                K, V = self._kv_cache
        else:
            K = self._project(key, self.W_K)
            V = self._project(value, self.W_V)
            if self._incremental:
                self._kv_cache = (K, V)

        if mask is not None and self._incremental and Q.size(2) == 1 and mask.dim() == 4:
            mask = mask[:, :, -1:, :]

        # Apply scaled dot-product attention to each head
        attn_output, attn_w = scaled_dot_product_attention(
            Q, K, V, mask, scale=self.scale_attention
        )  # shape: [batch, num_heads, seq_q, d_v]
        if getattr(self, 'store_attn_weights', False):
            self.last_attn_weights = attn_w.detach()
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_model)
        attn_output = self.W_O(attn_output) # shape: [batch, seq_q, d_model]

        return attn_output
#   ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        # Pre-compute positional encodings
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # For pytorch optimizations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0) # shape: [batch, seq_len, d_model]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (alternative to sinusoidal PE)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pe(positions)
        return self.dropout(x)


def build_positional_encoding(
    encoding_type: str,
    d_model: int,
    dropout: float = 0.1,
    max_len: int = 5000,
) -> nn.Module:
    if encoding_type == 'learned':
        return LearnedPositionalEncoding(d_model, dropout, max_len)
    if encoding_type == 'sinusoidal':
        return PositionalEncoding(d_model, dropout, max_len)
    raise ValueError(f"Unknown positional encoding type: {encoding_type}")


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        # Initialize submodules
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale_attention)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Self-attention sub-layer
        attn_output = self.self_attn(x, x, x, src_mask) # shape: [batch, src_len, d_model]
        x = self.norm_attn(x + attn_output) # Add & Norm

        # FFN sub-layer
        ffn_output = self.ffn(x) # shape: [batch, src_len, d_model]
        x = self.norm_ffn(x + ffn_output) # Add & Norm

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        # Initialize submodules
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, scale_attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, scale_attention)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.norm_cross_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Masked self-attention sub-layer
        self_attn_output = self.self_attn(x, x, x, tgt_mask) # shape: [batch, tgt_len, d_model]
        x = self.norm_self_attn(x + self_attn_output) # Add & Norm

        # Cross-attention sub-layer
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask) # shape: [batch, tgt_len, d_model]
        x = self.norm_cross_attn(x + cross_attn_output) # Add & Norm

        # FFN sub-layer
        ffn_output = self.ffn(x) # shape: [batch, tgt_len, d_model]
        x = self.norm_ffn(x + ffn_output) # Add & Norm

        return x

    def forward_step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single decoder step with KV cache (x is [batch, 1, d_model])."""
        self_attn_output = self.self_attn(x, x, x, None)
        x = self.norm_self_attn(x + self_attn_output)
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm_cross_attn(x + cross_attn_output)
        ffn_output = self.ffn(x)
        return self.norm_ffn(x + ffn_output)


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    def forward_step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_step(x, memory, src_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size (default: Multi30k de = 18669).
        tgt_vocab_size (int)  : Target vocabulary size (default: Multi30k en = 9797).
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = 18_669,
        tgt_vocab_size: int = 9_797,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        scale_attention: bool = True,
        positional_encoding_type: str = 'sinusoidal',
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.scale_attention = scale_attention
        self.positional_encoding_type = positional_encoding_type
        # Embeddings and positional encodings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = build_positional_encoding(
            positional_encoding_type, d_model, dropout
        )
        # Encoder and decoder stacks
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, scale_attention)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, scale_attention)
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        # Final linear layer to project decoder output to target vocab size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        _root = os.path.dirname(__file__)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(_root, "model_weights.pt")

        if not os.path.isfile(checkpoint_path):
            gdown.download(
                id=_PRETRAINED_CHECKPOINT_DRIVE_ID,
                output=checkpoint_path,
                quiet=False,
            )
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
            self._checkpoint_meta = {
                k: ckpt[k]
                for k in ("epoch", "val_bleu", "model_config", "train_config")
                if k in ckpt
            }
        else:
            self.load_state_dict(ckpt)
            self._checkpoint_meta = None

        self._tokenizer = None  # lazy: fast Transformer() init for autograder

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from dataset import get_infer_vocab
            self._tokenizer = get_infer_vocab()
        return self._tokenizer

    def _infer_begin(self) -> None:
        for layer in self.decoder.layers:
            layer.self_attn.start_incremental()
            layer.cross_attn.start_incremental()

    def _infer_end(self) -> None:
        for layer in self.decoder.layers:
            layer.self_attn.stop_incremental()
            layer.cross_attn.stop_incremental()

    def _embed_tgt(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embedding(tokens) * math.sqrt(self.d_model)
        return self.positional_encoding(x)

    def _embed_tgt_step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        """Embed a single target token with the correct positional index."""
        x = self.tgt_embedding(token) * math.sqrt(self.d_model)
        pe = self.positional_encoding
        if hasattr(pe, "pe") and pe.pe.dim() == 2:
            # Sinusoidal: registered buffer [max_len, d_model]
            x = x + pe.pe[position : position + 1].unsqueeze(0)
        else:
            # Learned positional embedding
            pos = torch.tensor([position], device=token.device)
            x = x + pe.pe(pos).unsqueeze(0)
        return x

    def _decode_step(
        self,
        token: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one greedy decode step; returns logits [batch, 1, vocab]."""
        x = self._embed_tgt_step(token, position)
        x = self.decoder.forward_step(x, memory, src_mask)
        return self.output_projection(x)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) # shape: [batch, src_len, d_model] ; scale by sqrt(d_model) to preserve token meaning
        src_emb = self.positional_encoding(src_emb) # shape: [batch, src_len, d_model]
        memory = self.encoder(src_emb, src_mask) # shape: [batch, src_len, d_model]
        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model) # shape: [batch, tgt_len, d_model]
        tgt_emb = self.positional_encoding(tgt_emb) # shape: [batch, tgt_len, d_model]
        decoder_output = self.decoder(tgt_emb, memory, src_mask, tgt_mask) # shape: [batch, tgt_len, d_model]
        logits = self.output_projection(decoder_output) # shape: [batch, tgt_len, tgt_vocab_size]
        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        return logits


    @torch.inference_mode()
    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        tokenizer = self._get_tokenizer()
        pad_idx = tokenizer.special_tokens['<pad>']
        sos = tokenizer.special_tokens['<sos>']
        eos = tokenizer.special_tokens['<eos>']
        unk = tokenizer.special_tokens['<unk>']
        src_tokens = [
            tokenizer.de_stoi.get(token.text.lower(), unk)
            for token in tokenizer.spacy_de(src_sentence)
        ]
        src_tokens = [sos] + src_tokens + [eos]
        device = next(self.parameters()).device
        src_tensor = torch.tensor(src_tokens, device=device).unsqueeze(0)
        src_mask = make_src_mask(src_tensor, pad_idx=pad_idx)
        memory = self.encode(src_tensor, src_mask)

        self._infer_begin()
        try:
            token = torch.tensor([[sos]], dtype=torch.long, device=device)
            generated = [sos]
            for step in range(1, INFER_MAX_LEN):
                logits = self._decode_step(token, step - 1, memory, src_mask)
                next_token = logits[:, 0, :].argmax(dim=-1).item()
                if next_token == eos:
                    break
                generated.append(next_token)
                token = torch.tensor([[next_token]], dtype=torch.long, device=device)
        finally:
            self._infer_end()

        tgt_words = [tokenizer.en_itos.get(idx, '<unk>') for idx in generated[1:]]
        return ' '.join(tgt_words)

