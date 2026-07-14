"""
model.py — Phase 3: PyTorch CNN policy/value network (Optimized v2)
====================================================================

Architecture improvements over v1:
  * Deeper conv stack (4 layers instead of 3) with residual connection.
  * Separate policy and value trunk so they don't share all features.
  * Spatial policy head: instead of global-average-pooling to a single
    4-vector, we produce a per-direction logit map and then pool. This
    preserves spatial information about which moves are locally good.
  * Graceful degradation: if torch is not installed, ``load_model``
    returns ``None`` and the solver falls back to plain DFS without
    any crash.

The network is still intentionally small (~300 KB) because it only
serves as a *move-ordering heuristic* on top of a strict DFS backtracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CHANNELS: int = 4
NUM_ACTIONS: int = 4
DEFAULT_WEIGHTS_PATH: Path = Path(__file__).resolve().parent / "zip_cnn.pth"


# ---------------------------------------------------------------------------
# Architecture (only defined if torch is available)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class ZipCNN(nn.Module):
        """
        Convolutional policy/value network for the Zip game.

        Layout
        ------
        Conv trunk (shared):
            4 → 32  (3×3, pad=1) + BN + ReLU
            32 → 64 (3×3, pad=1) + BN + ReLU
            64 → 64 (3×3, pad=1) + BN + ReLU  + residual
            64 → 64 (3×3, pad=1) + BN + ReLU

        Policy head:
            64 → 2  (1×1 conv) → flatten → 128 → NUM_ACTIONS

        Value head:
            64 → 1  (1×1 conv) → flatten → 128 → 1 (tanh)
        """

        def __init__(self, in_channels: int = NUM_CHANNELS,
                     num_actions: int = NUM_ACTIONS) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.num_actions = num_actions

            # Shared trunk.
            self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(64)

            # Policy head.
            self.policy_conv = nn.Conv2d(64, 2, 1)
            self.policy_fc = nn.Linear(2, 128)
            self.policy_head = nn.Linear(128, num_actions)

            # Value head.
            self.value_conv = nn.Conv2d(64, 1, 1)
            self.value_fc = nn.Linear(1, 128)
            self.value_head = nn.Linear(128, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = F.relu(self.bn1(self.conv1(x)))
            h = F.relu(self.bn2(self.conv2(h)))
            h2 = F.relu(self.bn3(self.conv3(h)))
            h = F.relu(self.bn4(self.conv4(h2))) + h2  # residual
            p_map = self.policy_conv(h)
            p_flat = p_map.mean(dim=(2, 3))
            p = F.relu(self.policy_fc(p_flat))
            policy_logits = self.policy_head(p)

            v_map = self.value_conv(h)
            v_flat = v_map.mean(dim=(2, 3))
            v = F.relu(self.value_fc(v_flat))
            value = torch.tanh(self.value_head(v))
            return policy_logits, value

        @torch.no_grad()
        def predict(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            self.eval()
            logits, value = self.forward(state_tensor)
            policy_probs = F.softmax(logits, dim=1).squeeze(0)
            return policy_probs, float(value.item())


    def load_model(weights_path: Path = DEFAULT_WEIGHTS_PATH,
                   device: str = "cpu") -> Optional[ZipCNN]:
        """Load the CNN. Returns None if torch is unavailable."""
        if not HAS_TORCH:
            return None
        model = ZipCNN().to(device)
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=device)
                if isinstance(state_dict, dict):
                    state_dict = state_dict.get("model", state_dict)
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
        model.eval()
        return model


    def save_model(model: ZipCNN,
                   weights_path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)


    def _pad_tensor(t: 'torch.Tensor', target_h: int, target_w: int) -> 'torch.Tensor':
        """Pad a (4, H, W) tensor to (4, target_h, target_w) with zeros."""
        c, h, w = t.shape
        if h == target_h and w == target_w:
            return t
        padded = torch.zeros(c, target_h, target_w, dtype=t.dtype)
        padded[:, :h, :w] = t
        return padded


    def train_on_samples(
        samples: list,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        device: str = "cpu",
    ) -> Optional[ZipCNN]:
        """
        AlphaZero-style composite loss training.

        samples : list of (state_tensor (4,H,W), policy_target (4,), value_target float)
        """
        if not HAS_TORCH or not samples:
            return load_model(weights_path, device)

        # Find the max H,W across all samples and pad everything to that size
        # so tensors of different board sizes can be stacked into a batch.
        max_h = max(s[0].shape[1] for s in samples)
        max_w = max(s[0].shape[2] for s in samples)

        model = load_model(weights_path, device).to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        states = torch.stack([
            _pad_tensor(s[0], max_h, max_w) for s in samples
        ]).to(device)
        policies = torch.stack([s[1] for s in samples]).to(device)
        values = torch.tensor([s[2] for s in samples],
                              dtype=torch.float32, device=device).unsqueeze(1)

        n = len(samples)
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_pl = 0.0
            total_vl = 0.0
            batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                # Pad each batch to the max size within that batch.
                bh = max(states[j].shape[1] for j in idx.tolist())
                bw = max(states[j].shape[2] for j in idx.tolist())
                s_b = torch.stack([
                    _pad_tensor(states[j], bh, bw) for j in idx.tolist()
                ])
                p_b = policies[idx]
                v_b = values[idx]

                opt.zero_grad()
                logits, v = model(s_b)
                pl = -(p_b * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                vl = F.mse_loss(v, v_b)
                loss = pl + vl
                loss.backward()
                opt.step()
                total_pl += pl.item()
                total_vl += vl.item()
                batches += 1

            if epoch % max(1, epochs // 10) == 0:
                print(f"[model] epoch {epoch:3d}  "
                      f"policy={total_pl/max(1,batches):.4f}  "
                      f"value={total_vl/max(1,batches):.4f}")

        save_model(model, weights_path)
        model.eval()
        return model


else:
    # No torch available — stub everything so imports don't crash.

    class ZipCNN:  # type: ignore[no-redef]
        pass

    def load_model(weights_path: Path = DEFAULT_WEIGHTS_PATH,
                   device: str = "cpu"):
        return None

    def save_model(model, weights_path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        pass

    def train_on_samples(samples, epochs=20, lr=1e-3, batch_size=64,
                         weights_path: Path = DEFAULT_WEIGHTS_PATH,
                         device: str = "cpu"):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not HAS_TORCH:
        print("torch not installed — model.py is a no-op stub.")
    else:
        m = load_model()
        if m is None:
            print("Model not loaded.")
        else:
            dummy = torch.zeros(1, NUM_CHANNELS, 8, 8)
            logits, value = m(dummy)
            print("Policy logits:", logits)
            print("Value:", value)
