"""
model.py
========
Phase 3 - Lightweight PyTorch CNN policy/value network.

The network takes a 4-channel grid tensor (see ``GameState.to_tensor``) and
emits:

    * a **policy** over the 4 cardinal moves (Up, Down, Left, Right).
    * a scalar **value** in [-1, 1] estimating the probability that the
      current state is on a winning trajectory.

The architecture is intentionally tiny - a Zip board is small enough that a
3-conv + 2-FC head suffices and stays well under a megabyte of weights. The
network is used by ``solver.py`` as a *move-ordering heuristic* on top of a
strict DFS backtracker, so perfect accuracy is not required: any reasonable
prior that ranks the promising moves first will reduce the search space
dramatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CHANNELS: int = 4      # cell type / number / visited / position
NUM_ACTIONS: int = 4       # up, down, left, right
DEFAULT_WEIGHTS_PATH: Path = Path(__file__).resolve().parent / "zip_cnn.pth"


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ZipCNN(nn.Module):
    """
    Convolutional policy/value network for the Zip game.

    Layout
    ------
    * Conv block 1: 4 -> 32 channels, 3x3 kernel, padding=1, ReLU
    * Conv block 2: 32 -> 64 channels, 3x3 kernel, padding=1, ReLU
    * Conv block 3: 64 -> 64 channels, 3x3 kernel, padding=1, ReLU
    * Policy head: flatten -> 256 -> NUM_ACTIONS (softmax over moves)
    * Value head:  flatten -> 256 -> 1 (tanh)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS,
                 num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Policy head
        self.policy_fc = nn.Linear(64, 256)
        self.policy_head = nn.Linear(256, num_actions)

        # Value head
        self.value_fc = nn.Linear(64, 256)
        self.value_head = nn.Linear(256, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, in_channels, H, W)``.

        Returns
        -------
        (policy_logits, value)
            ``policy_logits`` has shape ``(batch, num_actions)`` - apply
            softmax to get a probability distribution.
            ``value`` has shape ``(batch, 1)`` and is in ``(-1, 1)`` after
            tanh.
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))

        # Global average pool -> (batch, 64)
        pooled = h.mean(dim=(2, 3))

        # Policy
        p = F.relu(self.policy_fc(pooled))
        policy_logits = self.policy_head(p)

        # Value
        v = F.relu(self.value_fc(pooled))
        value = torch.tanh(self.value_head(v))
        return policy_logits, value

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Single-state inference helper.

        Parameters
        ----------
        state_tensor : torch.Tensor
            Either shape ``(4, H, W)`` or ``(1, 4, H, W)``.

        Returns
        -------
        (policy_probs, value)
            ``policy_probs`` is a 1-D float tensor of length ``num_actions``.
            ``value`` is a Python float in ``(-1, 1)``.
        """
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        self.eval()
        logits, value = self.forward(state_tensor)
        policy_probs = F.softmax(logits, dim=1).squeeze(0)
        return policy_probs, float(value.item())


# ---------------------------------------------------------------------------
# Weights I/O
# ---------------------------------------------------------------------------

def load_model(weights_path: Path = DEFAULT_WEIGHTS_PATH,
               device: str = "cpu") -> ZipCNN:
    """
    Instantiate the model and load weights.

    If the weights file does not exist, a freshly-initialised model is
    returned - the solver still works, it just has no learned prior and
    falls back to plain DFS.
    """
    model = ZipCNN().to(device)
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict):
            # Allow both raw state_dicts and {"model": ...} wrappers.
            state_dict = state_dict.get("model", state_dict)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # Weights are optional - any incompatibility just means we use
            # the random initialisation.
            pass
    model.eval()
    return model


def save_model(model: ZipCNN,
               weights_path: Path = DEFAULT_WEIGHTS_PATH) -> None:
    """Persist the model's state_dict to ``weights_path``."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)


# ---------------------------------------------------------------------------
# Self-supervised pre-training stub
# ---------------------------------------------------------------------------

def train_on_samples(
    samples: list,
    epochs: int = 10,
    lr: float = 1e-3,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    device: str = "cpu",
) -> ZipCNN:
    """
    Tiny training loop.

    Parameters
    ----------
    samples : list of (state_tensor, policy_target, value_target)
        ``state_tensor``  : float32 tensor (4, H, W)
        ``policy_target`` : float32 tensor (4,)  - action probabilities
        ``value_target``  : float scalar in {-1, 1}

    The training objective is the canonical AlphaZero-style composite loss:

        L = (z - v)^2 - pi^T * log(p) + c * ||theta||^2

    This is mainly intended for self-play fine-tuning once a corpus of
    solved games has been collected - it is not required for the solver to
    function.
    """
    if not samples:
        return load_model(weights_path, device)

    model = load_model(weights_path, device).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    states = torch.stack([s[0] for s in samples]).to(device)
    policies = torch.stack([s[1] for s in samples]).to(device)
    values = torch.tensor([s[2] for s in samples],
                          dtype=torch.float32, device=device).unsqueeze(1)

    for epoch in range(epochs):
        opt.zero_grad()
        logits, v = model(states)
        policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(v, values)
        loss = policy_loss + value_loss
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 5) == 0:
            print(f"[model] epoch {epoch:3d}  "
                  f"policy={policy_loss.item():.4f}  "
                  f"value={value_loss.item():.4f}")

    save_model(model, weights_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    m = load_model()
    dummy = torch.zeros(1, NUM_CHANNELS, 8, 8)
    logits, value = m(dummy)
    print("Policy logits:", logits)
    print("Value        :", value)
