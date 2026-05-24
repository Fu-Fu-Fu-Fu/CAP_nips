"""
Ablation 3: w/o Cross-Attention

Replaces cross-attention layers with mean-pooling of context embeddings.
The rest of the architecture (embedding, self-attention, actor/critic heads) is identical.

Two ways to activate:
  1. Set env var CAP_NO_CROSS_ATTN=1 before importing train_rl (auto-patch on import)
  2. Call apply_patch() manually after importing train_rl
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .train_rl import TransformerBlock, AttentionPooling


class NoCrossAttnDualTowerSelector(nn.Module):
    """
    ImprovedDualTowerSelector with cross-attention replaced by mean-pooling.

    Instead of N cross-attention layers that let candidates attend to context,
    we simply mean-pool the context tower output and broadcast-add it to each
    candidate embedding.  Everything else is identical.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        n_self_attn_layers: int = 3,
        n_cross_attn_layers: int = 3,   # accepted but ignored
        n_heads: int = 8,
        dropout: float = 0.1,
        max_steps: int = 20,
        use_taf_feature: bool = False,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.use_taf_feature = use_taf_feature

        # Context features: [x, y_rank] -> (coord_dim + 1)
        context_input_dim = coord_dim + 1

        # Candidate features: [x, mu, sigma, is_persistent] -> (coord_dim + 3)
        # With TAF rank: [x, mu, sigma, is_persistent, taf_rank] -> (coord_dim + 4)
        candidate_input_dim = coord_dim + (4 if use_taf_feature else 3)

        # Embedding layers
        self.context_embed = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.candidate_embed = nn.Sequential(
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Step embedding
        self.step_embed = nn.Embedding(max_steps + 1, hidden_dim)

        # Context tower: Self-Attention
        self.context_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_self_attn_layers)
        ])

        # Candidate tower: Self-Attention only (NO cross-attention)
        self.candidate_self_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_self_attn_layers)
        ])

        # NO self.cross_layers — this is the ablation

        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Critic Head (attention pooling)
        self.context_pool = AttentionPooling(hidden_dim)
        self.candidate_pool = AttentionPooling(hidden_dim)

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context, candidates, step, context_mask=None):
        """
        Args:
            context: (batch, n_context, context_dim)
            candidates: (batch, n_candidates, candidate_dim)
            step: (batch,)
            context_mask: optional
        """
        # Embedding
        ctx_emb = self.context_embed(context)
        cand_emb = self.candidate_embed(candidates)

        # Step embedding
        step_emb = self.step_embed(step)  # (batch, hidden_dim)
        ctx_emb = ctx_emb + step_emb.unsqueeze(1)
        cand_emb = cand_emb + step_emb.unsqueeze(1)

        # Context tower: self-attention
        for layer in self.context_layers:
            ctx_emb = layer(ctx_emb, mask=context_mask)

        # Candidate tower: self-attention
        for layer in self.candidate_self_layers:
            cand_emb = layer(cand_emb)

        # ABLATION: mean-pool context → broadcast-add to candidates
        # (replaces cross-attention layers)
        if context_mask is not None:
            mask_expanded = context_mask.unsqueeze(-1).float()  # (batch, n_ctx, 1)
            ctx_sum = (ctx_emb * mask_expanded).sum(dim=1, keepdim=True)
            ctx_count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            ctx_mean = ctx_sum / ctx_count  # (batch, 1, hidden_dim)
        else:
            ctx_mean = ctx_emb.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        cand_emb = cand_emb + ctx_mean

        # Actor: per-candidate logit
        logits = self.actor_head(cand_emb).squeeze(-1)  # (batch, n_candidates)

        # Critic: pooled features
        ctx_pooled = self.context_pool(ctx_emb, mask=context_mask)
        cand_pooled = self.candidate_pool(cand_emb)

        combined = torch.cat([ctx_pooled, cand_pooled], dim=-1)
        value = self.critic_head(combined).squeeze(-1)

        return logits, value

    def get_action(self, context, candidates, step, context_mask=None):
        logits, value = self.forward(context, candidates, step, context_mask)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, context, candidates, step, actions, context_mask=None):
        logits, value = self.forward(context, candidates, step, context_mask)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, value


def apply_patch():
    """Replace ImprovedDualTowerSelector with NoCrossAttnDualTowerSelector in train_rl module.

    Normally not needed — the auto-patch in myrl/rl/__init__.py handles this
    when CAP_NO_CROSS_ATTN=1 env var is set.
    """
    from . import train_rl as _mod
    if _mod.ImprovedDualTowerSelector is not NoCrossAttnDualTowerSelector:
        _mod.ImprovedDualTowerSelector = NoCrossAttnDualTowerSelector
