import torch
import torch.nn as nn


class SceneQueryExchange(nn.Module):
    """Exchange detection-query information within, but never across, scenes."""

    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=1024,
                 dropout=0.1, residual_init=1e-3):
        super().__init__()
        if embed_dims <= 0 or ffn_dims <= 0:
            raise ValueError('embedding dimensions must be positive')
        if num_heads <= 0 or embed_dims % num_heads:
            raise ValueError('num_heads must divide embed_dims')
        self.attn_norm = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout))
        self.gamma_attn = nn.Parameter(torch.full(
            (), float(residual_init)))
        self.gamma_ffn = nn.Parameter(torch.full(
            (), float(residual_init)))

    def forward(self, query):
        if query.ndim != 4:
            raise ValueError('query must have shape [B, V, Q, D]')
        batch_size, num_views, num_queries, embed_dims = query.shape
        if embed_dims != self.attn.embed_dim:
            raise ValueError('query embedding dimension does not match module')
        if num_views == 1:
            return query

        tokens = query.reshape(batch_size, num_views * num_queries, embed_dims)
        normalized = self.attn_norm(tokens)
        attended, _ = self.attn(normalized, normalized, normalized,
                                need_weights=False)
        tokens = tokens + self.gamma_attn * attended
        tokens = tokens + self.gamma_ffn * self.ffn(self.ffn_norm(tokens))
        return tokens.reshape(batch_size, num_views, num_queries, embed_dims)
