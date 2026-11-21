"""Permutation-equivariant actor-critic for CityTaxi-v0.

A flat MLP treats "rider in slot 2" and "rider in slot 4" as unrelated inputs, so a
lesson like *don't take airport runs at rush hour* must be relearned for every slot
from rare examples. Here one shared network scores every candidate rider and another
scores every zone; the action logits are those scores. Anything learned about one
rider transfers to all of them, and the policy is invariant to slot order.

The critic pools the per-rider / per-zone embeddings (mean + max) with the global
context, so it is permutation-invariant too.
"""

from __future__ import annotations

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

N_GLOBAL = 7
SLOT_F = 8
ZONE_F = 5


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ScoringExtractor(nn.Module):
    """Plays the role of SB3's ``mlp_extractor``: returns (logits, value-latent)."""

    def __init__(self, n_slots: int, n_zones: int, hidden: int = 128, emb: int = 64):
        super().__init__()
        self.n_slots, self.n_zones = n_slots, n_zones
        self.ctx = mlp(N_GLOBAL, hidden, emb)
        self.slot_emb = mlp(SLOT_F + N_GLOBAL, hidden, emb)
        self.zone_emb = mlp(ZONE_F + N_GLOBAL + n_zones, hidden, emb)
        self.slot_head = nn.Sequential(nn.Tanh(), nn.Linear(emb * 2, 1))
        self.zone_head = nn.Sequential(nn.Tanh(), nn.Linear(emb * 2, 1))
        self.wait_head = nn.Sequential(nn.Tanh(), nn.Linear(emb, 1))
        self.vf = nn.Sequential(nn.Linear(emb * 5, hidden), nn.Tanh())
        self.latent_dim_pi = n_slots + n_zones + 1
        self.latent_dim_vf = hidden
        self.register_buffer("zone_id", torch.eye(n_zones))

    def _split(self, obs: torch.Tensor):
        b = obs.shape[0]
        g = obs[:, :N_GLOBAL]
        o = N_GLOBAL
        slots = obs[:, o : o + self.n_slots * SLOT_F].reshape(b, self.n_slots, SLOT_F)
        o += self.n_slots * SLOT_F
        zones = obs[:, o : o + self.n_zones * ZONE_F].reshape(b, self.n_zones, ZONE_F)
        return g, slots, zones

    def _embed(self, obs: torch.Tensor):
        g, slots, zones = self._split(obs)
        b = obs.shape[0]
        c = self.ctx(g)  # (B, E)
        gs = g[:, None, :].expand(b, self.n_slots, N_GLOBAL)
        gz = g[:, None, :].expand(b, self.n_zones, N_GLOBAL)
        se = self.slot_emb(torch.cat([slots, gs], -1))  # (B, S, E)
        zid = self.zone_id[None].expand(b, -1, -1)
        ze = self.zone_emb(torch.cat([zones, gz, zid], -1))  # (B, Z, E)
        valid = slots[:, :, :1]  # 1 if the slot holds a rider
        return c, se, ze, valid

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        c, se, ze, _ = self._embed(obs)
        cs = c[:, None, :]
        s_logit = self.slot_head(torch.cat([se, cs.expand_as(se)], -1)).squeeze(-1)
        z_logit = self.zone_head(torch.cat([ze, cs.expand_as(ze)], -1)).squeeze(-1)
        w_logit = self.wait_head(c)
        return torch.cat([s_logit, z_logit, w_logit], -1)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        c, se, ze, valid = self._embed(obs)
        n = valid.sum(1).clamp(min=1.0)
        s_mean = (se * valid).sum(1) / n
        s_max = (se * valid - 1e4 * (1 - valid)).max(1).values * (valid.sum(1) > 0)
        return self.vf(torch.cat([c, s_mean, s_max, ze.mean(1), ze.max(1).values], -1))

    def forward(self, obs: torch.Tensor):
        return self.forward_actor(obs), self.forward_critic(obs)


class ScoringPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose logits are per-candidate scores."""

    def __init__(self, *args, n_slots: int = 5, n_zones: int = 8, **kwargs):
        self._n_slots, self._n_zones = n_slots, n_zones
        kwargs["share_features_extractor"] = True
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ScoringExtractor(self._n_slots, self._n_zones)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Logits come straight from the scoring heads.
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
