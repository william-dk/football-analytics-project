"""Masked-window route autoencoder with a small bottleneck z (PyTorch).

Encoder: one token per frame (knocked-out frames are replaced by a learned mask token) plus a
learned time embedding and a [CLS] token -> Transformer -> z (default 16 numbers).
Decoder: one query per time step, built only from a time embedding and z -> Transformer over the
route's real time steps -> output channels.

The decoder never receives input frames. Everything it reconstructs, including the knocked-out
window, has to pass through z, so z has to describe the whole route.
"""
from __future__ import annotations

import torch
from torch import nn


def _layer(d_model: int, n_heads: int, dim_ff: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, activation="gelu",
                                      batch_first=True, norm_first=True)


class RouteAutoencoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, max_len: int = 77, d_model: int = 64, n_heads: int = 4,
                 enc_layers: int = 3, dec_layers: int = 3, z_dim: int = 16, dim_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.config = dict(n_in=n_in, n_out=n_out, max_len=max_len, d_model=d_model, n_heads=n_heads,
                           enc_layers=enc_layers, dec_layers=dec_layers, z_dim=z_dim, dim_ff=dim_ff,
                           dropout=dropout)
        self.in_proj = nn.Linear(n_in, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.enc_time = nn.Embedding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(_layer(d_model, n_heads, dim_ff, dropout), enc_layers,
                                             enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d_model)
        self.to_z = nn.Linear(d_model, z_dim)
        self.from_z = nn.Linear(z_dim, d_model)
        self.dec_time = nn.Embedding(max_len, d_model)
        self.decoder = nn.TransformerEncoder(_layer(d_model, n_heads, dim_ff, dropout), dec_layers,
                                             enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_out)

    def encode(self, x: torch.Tensor, valid: torch.Tensor, knocked_out: torch.Tensor) -> torch.Tensor:
        """x (B, T, n_in) scaled features; valid and knocked_out (B, T) bool. Returns z (B, z_dim)."""
        batch, steps, _ = x.shape
        tokens = self.in_proj(x)
        tokens = torch.where(knocked_out.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        tokens = tokens + self.enc_time.weight[:steps].unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(batch, 1, -1), tokens], dim=1)
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        hidden = self.encoder(tokens, src_key_padding_mask=padding)
        return self.to_z(self.enc_norm(hidden[:, 0]))

    def decode(self, z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Rebuild every time step from z alone. valid (B, T) says which steps the route has."""
        steps = valid.shape[1]
        queries = self.dec_time.weight[:steps].unsqueeze(0) + self.from_z(z).unsqueeze(1)
        hidden = self.decoder(queries, src_key_padding_mask=~valid)
        return self.out(self.dec_norm(hidden))

    def forward(self, x, valid, knocked_out):
        z = self.encode(x, valid, knocked_out)
        return self.decode(z, valid), z


def masked_mse(pred, target, valid, knocked_out, visible_weight: float = 0.5):
    """Squared error summed over channels, averaged over knocked-out frames (main term) plus
    visible_weight times the average over the visible real frames. Padding is ignored."""
    err = (pred - target).pow(2).sum(-1)
    hidden = knocked_out & valid
    shown = valid & ~knocked_out
    hidden_loss = (err * hidden).sum() / hidden.sum().clamp(min=1)
    shown_loss = (err * shown).sum() / shown.sum().clamp(min=1)
    return hidden_loss + visible_weight * shown_loss, hidden_loss, shown_loss


@torch.no_grad()
def run_batches(model: RouteAutoencoder, x, valid, knocked_out, batch_size: int = 1024):
    """Evaluation-mode forward pass over a whole split. Returns (pred, z)."""
    model.eval()
    preds, zs = [], []
    for i in range(0, len(x), batch_size):
        pred, z = model(x[i:i + batch_size], valid[i:i + batch_size], knocked_out[i:i + batch_size])
        preds.append(pred)
        zs.append(z)
    return torch.cat(preds), torch.cat(zs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
