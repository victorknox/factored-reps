from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import math

from transformer_lens.hook_points import HookPoint, HookedRootModule


@dataclass
class AttentionOnlyConfig:
    """Configuration for attention-only transformer."""

    d_vocab: int
    d_model: int
    n_heads: int
    n_layers: int
    n_ctx: int
    d_head: Optional[int] = None  # If None, d_head = d_model // n_heads
    dropout: float = 0.0
    normalization_type: Literal["LN", "RMS", None] = "LN"
    device: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.d_head is None:
            assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
            self.d_head = self.d_model // self.n_heads


class Attention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, cfg: AttentionOnlyConfig):
        super().__init__()
        self.cfg = cfg
        d_head = cfg.d_head
        assert d_head is not None

        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, d_head))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, d_head))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, d_head))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, d_head, cfg.d_model))

        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, d_head))
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, d_head))
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, d_head))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

        # Initialize weights
        for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(W, std=0.02)

        # Register causal mask buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx), diagonal=1).bool(),
        )

        # Hook points for attention internals (TransformerLens style)
        self.hook_q = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_k = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_v = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, n_heads, pos, pos]
        self.hook_pattern = HookPoint()  # [batch, n_heads, pos, pos] (after softmax)
        self.hook_z = HookPoint()  # [batch, pos, n_heads, d_head]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, pos, d_model)

        Returns:
            Output tensor of shape (batch, pos, d_model)
        """
        batch, pos, _ = x.shape

        # Compute Q, K, V: [batch, pos, n_heads, d_head]
        q = torch.einsum("bpd,hdk->bphk", x, self.W_Q) + self.b_Q
        k = torch.einsum("bpd,hdk->bphk", x, self.W_K) + self.b_K
        v = torch.einsum("bpd,hdk->bphk", x, self.W_V) + self.b_V

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        # Attention scores: [batch, n_heads, pos, pos]
        d_head = self.cfg.d_head
        assert d_head is not None
        attn_scores = torch.einsum("bphk,bqhk->bhpq", q, k) / math.sqrt(d_head)
        attn_scores = self.hook_attn_scores(attn_scores)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(self.mask[:pos, :pos], float("-inf"))

        # Softmax to get attention pattern
        pattern = torch.softmax(attn_scores, dim=-1)
        pattern = self.hook_pattern(pattern)

        # Apply attention to values: [batch, pos, n_heads, d_head]
        z = torch.einsum("bhpq,bqhk->bphk", pattern, v)
        z = self.hook_z(z)

        # Project back to d_model: [batch, pos, d_model]
        out = torch.einsum("bphk,hkd->bpd", z, self.W_O) + self.b_O

        return out


class AttentionOnlyBlock(nn.Module):
    """A single attention-only transformer block."""

    def __init__(self, cfg: AttentionOnlyConfig, block_idx: int):
        super().__init__()
        self.cfg = cfg
        self.block_idx = block_idx

        # Layer normalization before attention (pre-norm architecture)
        if cfg.normalization_type == "LN":
            self.ln = nn.LayerNorm(cfg.d_model)
        elif cfg.normalization_type == "RMS":
            self.ln = RMSNorm(cfg.d_model)
        else:
            self.ln = nn.Identity()

        self.attn = Attention(cfg)

        # Hook points for residual stream
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, pos, d_model)

        Returns:
            Output tensor of shape (batch, pos, d_model)
        """
        resid_pre = self.hook_resid_pre(x)
        normalized = self.ln(resid_pre)
        attn_out = self.attn(normalized)
        attn_out = self.hook_attn_out(attn_out)
        resid_post = self.hook_resid_post(resid_pre + attn_out)
        return resid_post


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AttentionOnly(HookedRootModule):
    """Attention-only transformer model.

    A simplified transformer with only attention mechanisms (no MLP/FFN layers).
    Supports TransformerLens-style run_with_cache for activation analysis.

    Hook naming convention:
        - blocks.{i}.hook_resid_pre: Input to block i
        - blocks.{i}.hook_attn_out: Output of attention in block i
        - blocks.{i}.hook_resid_post: Output of block i
        - blocks.{i}.attn.hook_q/k/v: Query/Key/Value after projection
        - blocks.{i}.attn.hook_pattern: Attention pattern after softmax
        - blocks.{i}.attn.hook_z: Attention output before projection
        - ln_final.hook_normalized: Final layer norm output (if normalization enabled)
    """

    def __init__(self, cfg: AttentionOnlyConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        # Token embedding
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)

        # Positional embedding
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)

        # Hook for embeddings
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        self.hook_pos_embed = HookPoint()  # [batch, pos, d_model]

        # Attention blocks
        self.blocks = nn.ModuleList(
            [AttentionOnlyBlock(cfg, i) for i in range(cfg.n_layers)]
        )

        # Final layer normalization
        if cfg.normalization_type == "LN":
            self.ln_final = nn.LayerNorm(cfg.d_model)
        elif cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(cfg.d_model)
        else:
            self.ln_final = nn.Identity()

        # Hook for final layer norm
        self.hook_ln_final = HookPoint()

        # Unembedding (output projection)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        # Dropout
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else None

        if cfg.device is not None:
            self.to(cfg.device)

        # Setup hook points - MUST be called after all modules are defined
        self.setup()

    def forward(
        self, x: torch.Tensor, return_type: Optional[str] = "logits"
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices.
            return_type: Type of output to return (only "logits" supported).

        Returns:
            Logits tensor of shape (batch_size, seq_len, d_vocab).
        """
        batch, pos = x.shape

        # Token embeddings
        tok_embed = self.embed(x)
        tok_embed = self.hook_embed(tok_embed)

        # Positional embeddings
        positions = torch.arange(pos, device=x.device)
        pos_embed = self.pos_embed(positions)
        pos_embed = self.hook_pos_embed(pos_embed)

        # Combine embeddings
        h = tok_embed + pos_embed

        if self.dropout is not None:
            h = self.dropout(h)

        # Pass through attention blocks
        for block in self.blocks:
            h = block(h)
            if self.dropout is not None:
                h = self.dropout(h)

        # Final layer norm
        h = self.ln_final(h)
        h = self.hook_ln_final(h)

        # Project to vocabulary
        logits = self.unembed(h)

        return logits
