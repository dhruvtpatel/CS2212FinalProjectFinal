import torch
import torch.nn as nn
from torch.distributions import Normal

MU_LOW   = 0.1
MU_HIGH  = 3.0
MU_RANGE = MU_HIGH - MU_LOW


class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeEncoder(nn.Module):

    def __init__(self, feat_dim: int = 10, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [_Block(feat_dim, hidden)]
        for _ in range(depth - 1):
            layers.append(_Block(hidden, hidden))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCritic(nn.Module):

    def __init__(
        self,
        feat_dim: int   = 10,
        hidden:   int   = 128,
        depth:    int   = 3,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.encoder  = EdgeEncoder(feat_dim, hidden, depth)

        self.mu_head  = nn.Linear(hidden, 1)

        self.log_std  = nn.Parameter(torch.full((1,), log_std_init))

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        h       = self.encoder(obs)
        raw_mu  = self.mu_head(h).squeeze(-1)
        std     = self.log_std.exp().expand(raw_mu.shape)
        dist    = Normal(raw_mu, std)

        h_global = h.mean(dim=0, keepdim=True)
        value    = self.value_head(h_global).squeeze()
        return dist, value

    @staticmethod
    def squash(raw: torch.Tensor) -> torch.Tensor:
        return MU_LOW + MU_RANGE * torch.sigmoid(raw)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        raw_action  = dist.mean if deterministic else dist.rsample()
        log_prob    = dist.log_prob(raw_action).mean()
        mu_action   = self.squash(raw_action)
        return raw_action, mu_action, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob    = dist.log_prob(raw_action).mean()
        entropy     = dist.entropy().mean()
        return log_prob, value, entropy
