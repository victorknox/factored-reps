from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch
import torch.nn as nn


@dataclass
class RNNConfig:
    d_vocab: int
    d_model: int
    n_layers: int
    n_ctx: int # Not used for RNN but relevant for generative process
    rnn_type: Literal["rnn", "lstm", "gru"] = "gru"
    dropout: float = 0.0
    device: Optional[str] = None
    seed: Optional[int] = None


class RNN(nn.Module):
    """A simple RNN model for sequence prediction.

    Supports vanilla RNN, LSTM, and GRU architectures. Uses stacked single-layer
    RNNs to enable capturing hidden states at each layer for activation analysis.
    """

    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)

        rnn_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[cfg.rnn_type]

        # Use stacked single-layer RNNs to capture per-layer hidden states
        self.rnn_layers = nn.ModuleList([
            rnn_cls(
                input_size=cfg.d_model,
                hidden_size=cfg.d_model,
                num_layers=1,
                batch_first=True,
            )
            for _ in range(cfg.n_layers)
        ])

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else None
        self.output_proj = nn.Linear(cfg.d_model, cfg.d_vocab)

        self._init_weights()

        if cfg.device is not None:
            self.to(cfg.device)

    def _init_weights(self):
        """Initialize weights following best practices for RNNs.

        - Orthogonal initialization for hidden-to-hidden weights
        - Xavier initialization for input-to-hidden weights
        - Zero initialization for biases
        - For LSTM: forget gate bias set to 1.0 (Jozefowicz et al., 2015)
        """
        for module in self.rnn_layers:
            rnn_layer = cast(nn.RNNBase, module)
            # Xavier for input-to-hidden weights
            nn.init.xavier_uniform_(rnn_layer.weight_ih_l0)
            # Orthogonal for hidden-to-hidden weights
            nn.init.orthogonal_(rnn_layer.weight_hh_l0)
            # Zero biases
            nn.init.zeros_(rnn_layer.bias_ih_l0)
            nn.init.zeros_(rnn_layer.bias_hh_l0)

            # For LSTM: set forget gate bias to 1.0
            if self.cfg.rnn_type == "lstm":
                hidden_size = self.cfg.d_model
                # LSTM bias layout: [input_gate, forget_gate, cell_gate, output_gate]
                rnn_layer.bias_hh_l0.data[hidden_size : 2 * hidden_size].fill_(1.0)

        # Xavier for output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            Logits tensor of shape (batch_size, seq_len, d_vocab).
        """
        h = self.embedding(x)

        for i, rnn_layer in enumerate(self.rnn_layers):
            h, _ = rnn_layer(h)
            if self.dropout is not None and i < len(self.rnn_layers) - 1:
                h = self.dropout(h)

        logits = self.output_proj(h)
        return logits

    def run_with_cache(
        self, x: torch.Tensor, return_type: str = "logits"
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with activation caching for analysis.

        Accumulates hidden states at each timestep for each layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices.
            return_type: Type of output to return (only "logits" supported).

        Returns:
            Tuple of (logits, cache) where cache contains intermediate activations
            with shape (batch_size, seq_len, d_model) for each layer.
        """
        cache = {}

        h = self.embedding(x)
        cache["embed"] = h

        for i, rnn_layer in enumerate(self.rnn_layers):
            h, _ = rnn_layer(h)
            cache[f"rnn.{i}.hidden"] = h
            if self.dropout is not None and i < len(self.rnn_layers) - 1:
                h = self.dropout(h)

        logits = self.output_proj(h)

        return logits, cache
