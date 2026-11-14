"""Masked-window route autoencoder with a small bottleneck z (PyTorch).

Encoder (BERT-style): one token per frame (knocked-out frames are replaced by a learned mask
token) plus a learned time embedding and a [CLS] token -> Transformer -> z (default 16 numbers).
  readout "cls"     z is read from the [CLS] token only (first version).
  readout "frames"  z also gets a linear readout of every frame's output state, including the
                    states at knocked-out frames, which the Transformer has to fill from context.
                    This gives z an exact linear view of the route, which [CLS] alone cannot learn
                    to the precision a 16-number code allows.
Decoder: built only from z.
  decoder "transformer"  one query per time step (time embedding + z) -> Transformer (first version).
  decoder "linear"       positions = bias + basis @ z: every z dimension is one fixed route
                         deformation, so the mean of several z decodes to the mean route.
  decoder "mlp"          the linear decoder plus a small nonlinear correction.
  basis K > 0            (linear and mlp) the decoder writes K coefficients of fixed "eigen-routes"
                         (principal components of the training routes, set with set_basis) instead
                         of one free number per frame. Routes are smooth, so their positions at
                         neighbouring frames are almost copies of each other; in the eigen-route
                         basis the outputs are uncorrelated and equally scaled, which is what makes
                         gradient training reach fine detail, and the result is smooth by design.

The decoder never receives input frames. Everything it reconstructs, including the knocked-out
window, has to pass through z, so z has to describe the whole route.
"""
from __future__ import annotations

import torch
from torch import nn

READOUTS = ("cls", "frames")
DECODERS = ("transformer", "linear", "mlp")


def _layer(d_model: int, n_heads: int, dim_ff: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, activation="gelu",
                                      batch_first=True, norm_first=True)


class RouteAutoencoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, max_len: int = 77, d_model: int = 64, n_heads: int = 4,
                 enc_layers: int = 3, dec_layers: int = 3, z_dim: int = 16, dim_ff: int = 128,
                 dropout: float = 0.1, readout: str = "cls", decoder: str = "transformer",
                 dec_hidden: int = 256, basis: int = 0):
        super().__init__()
        if readout not in READOUTS or decoder not in DECODERS:
            raise ValueError(f"readout must be one of {READOUTS} and decoder one of {DECODERS}")
        self.config = dict(n_in=n_in, n_out=n_out, max_len=max_len, d_model=d_model, n_heads=n_heads,
                           enc_layers=enc_layers, dec_layers=dec_layers, z_dim=z_dim, dim_ff=dim_ff,
                           dropout=dropout, readout=readout, decoder=decoder, dec_hidden=dec_hidden,
                           basis=basis)
        self.max_len, self.n_out, self.readout, self.decoder_kind = max_len, n_out, readout, decoder
        self.basis_size = basis if decoder != "transformer" else 0
        self.in_proj = nn.Linear(n_in, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.enc_time = nn.Embedding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(_layer(d_model, n_heads, dim_ff, dropout), enc_layers,
                                             enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d_model)
        self.to_z = nn.Linear(d_model, z_dim)
        if readout == "frames":
            self.frame_readout = nn.Linear(max_len * d_model, z_dim, bias=False)
        if decoder == "transformer":
            self.from_z = nn.Linear(z_dim, d_model)
            self.dec_time = nn.Embedding(max_len, d_model)
            self.decoder = nn.TransformerEncoder(_layer(d_model, n_heads, dim_ff, dropout), dec_layers,
                                                 enable_nested_tensor=False)
            self.dec_norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, n_out)
        else:
            width = self.basis_size or max_len * n_out
            if self.basis_size:
                self.register_buffer("basis_mean", torch.zeros(max_len * n_out))
                self.register_buffer("basis_vectors", torch.zeros(self.basis_size, max_len * n_out))
                self.register_buffer("basis_scale", torch.ones(self.basis_size))
            self.dec_linear = nn.Linear(z_dim, width)
            if decoder == "mlp":
                self.dec_mlp = nn.Sequential(nn.Linear(z_dim, dec_hidden), nn.GELU(),
                                             nn.Linear(dec_hidden, dec_hidden), nn.GELU(),
                                             nn.Linear(dec_hidden, width))
                nn.init.zeros_(self.dec_mlp[-1].weight)       # start as the plain linear decoder
                nn.init.zeros_(self.dec_mlp[-1].bias)

    def encode(self, x: torch.Tensor, valid: torch.Tensor, knocked_out: torch.Tensor) -> torch.Tensor:
        """x (B, T, n_in) scaled features; valid and knocked_out (B, T) bool. Returns z (B, z_dim)."""
        batch, steps, _ = x.shape
        tokens = self.in_proj(x)
        tokens = torch.where(knocked_out.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        tokens = tokens + self.enc_time.weight[:steps].unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(batch, 1, -1), tokens], dim=1)
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        hidden = self.encoder(tokens, src_key_padding_mask=padding)
        z = self.to_z(self.enc_norm(hidden[:, 0]))
        if self.readout == "frames":
            frames = hidden[:, 1:] * valid.unsqueeze(-1)          # padded frames contribute nothing
            if steps < self.max_len:
                frames = nn.functional.pad(frames, (0, 0, 0, self.max_len - steps))
            z = z + self.frame_readout(frames.flatten(1))
        return z

    def decode(self, z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Rebuild every time step from z alone. valid (B, T) says which steps the route has."""
        steps = valid.shape[1]
        if self.decoder_kind == "transformer":
            queries = self.dec_time.weight[:steps].unsqueeze(0) + self.from_z(z).unsqueeze(1)
            hidden = self.decoder(queries, src_key_padding_mask=~valid)
            return self.out(self.dec_norm(hidden))
        out = self.dec_linear(z)
        if self.decoder_kind == "mlp":
            out = out + self.dec_mlp(z)
        if self.basis_size:
            out = self.basis_mean + (out * self.basis_scale) @ self.basis_vectors
        return out.view(len(z), self.max_len, self.n_out)[:, :steps]

    def set_basis(self, mean, vectors, scale) -> None:
        """Fix the eigen-route basis: mean (T * n_out,), vectors (K, T * n_out) orthonormal rows,
        scale (K,) the standard deviation of each coefficient. All in scaled units, time-major."""
        self.basis_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.basis_vectors.copy_(torch.as_tensor(vectors, dtype=torch.float32))
        self.basis_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

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


def velocity_mse(pred, target, valid, knocked_out, visible_weight: float = 0.5, fps: float = 10.0):
    """Squared error between the frame-to-frame steps of the prediction and of the target, as a
    velocity (scaled units per second). It is zero for a perfect reconstruction and penalises
    jitter, which a position error alone barely sees. A step counts as knocked out when either of
    its two frames is; visible steps get visible_weight, as in masked_mse."""
    step_err = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).mul(fps).pow(2).sum(-1)
    pair = valid[:, 1:] & valid[:, :-1]
    hidden = pair & (knocked_out[:, 1:] | knocked_out[:, :-1])
    shown = pair & ~hidden
    hidden_loss = (step_err * hidden).sum() / hidden.sum().clamp(min=1)
    shown_loss = (step_err * shown).sum() / shown.sum().clamp(min=1)
    return hidden_loss + visible_weight * shown_loss


@torch.no_grad()
def run_batches(model: RouteAutoencoder, x, valid, knocked_out, batch_size: int = 1024):
    """Evaluation-mode forward pass over a whole split. Returns (pred, z).

    Each batch is cut to its longest route before the forward pass (padding cannot influence real
    frames) and the prediction is padded back with zeros, so sorting rows by length makes this
    much faster without changing any real frame."""
    model.eval()
    steps = x.shape[1]
    preds, zs = [], []
    for i in range(0, len(x), batch_size):
        v = valid[i:i + batch_size]
        used = max(1, int(v.sum(1).max()))
        pred, z = model(x[i:i + batch_size, :used], v[:, :used], knocked_out[i:i + batch_size, :used])
        if used < steps:
            pred = torch.nn.functional.pad(pred, (0, 0, 0, steps - used))
        preds.append(pred)
        zs.append(z)
    return torch.cat(preds), torch.cat(zs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
