# flake8: noqa
# %%
import torch as t
import torch
import torch.nn as nn
import einops
from einops import rearrange
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import dataclass, field
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Union, Callable

t.manual_seed(2)

W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)
device = "cuda"


def constant_lr(*_):
    return 1.0


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    # n_features: int = 5
    groups: list[int] = field(default_factory=lambda: [2, 2])
    n_hidden: int = 2

    @property
    def n_features(self) -> int:
        return sum(self.groups)


# Toy model of superposition
class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device=device,
    ):
        super().__init__()
        self.cfg = cfg

        n_features = sum(cfg.groups)
        self.n_features = n_features

        if feature_probability is None:
            feature_probability = t.ones(())
        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_instances, n_features)
        )
        if importance is None:
            importance = t.ones(())
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to(
            (cfg.n_instances, n_features)
        )

        self.W = nn.Parameter(
            nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, n_features)))
        )
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, n_features)))
        self.to(device)

    def forward(
        self, features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = einops.einsum(
            features,
            self.W,
            "... instances features, instances hidden features -> ... instances hidden",
        )
        out = einops.einsum(
            hidden,
            self.W,
            "... instances hidden, instances hidden features -> ... instances features",
        )
        return F.relu(out + self.b_final)

    def generate_batch(
        self, batch_size
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of data. We'll return to this function later when we apply correlations.
        """
        return torch.cat(
            [
                rearrange(
                    F.one_hot(
                        torch.randint(
                            0,
                            i,
                            (
                                batch_size,
                                self.cfg.n_instances,
                            ),
                        )
                    ),
                    "",
                )
                for i in self.cfg.groups
            ],
            dim=-1,
        )

    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Remember, `self.importance` will always have shape (n_instances, n_features).
        """
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(
            error, "batch instances features -> instances", "mean"
        ).sum()
        return loss

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / self.cfg.n_instances, lr=step_lr
                )


# %%
cfg = Config(n_instances=8, n_hidden=2, groups=[2, 2])
model = Model(cfg)
print(model)

# importance varies within features for each instance
# importance = 1 ** torch.ones(cfg.n_features)
# importance = einops.rearrange(importance, "features -> () features")

# sparsity is the same for all features in a given instance, but varies over instances
# feature_probability = 50 ** -torch.linspace(0, 1, cfg.n_instances)
# feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

# line(importance.squeeze(), width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})
# line(feature_probability.squeeze(), width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
# %%
