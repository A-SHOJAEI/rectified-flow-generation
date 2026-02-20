"""Rectified flow / conditional flow matching implementation.

Implements the core training and sampling algorithms for rectified flow:
- Linear interpolation: x_t = (1-t)*x_0 + t*x_1
- Velocity target: v = x_1 - x_0
- CFM loss: ||v_theta(x_t, t) - v||^2
"""

import torch
import torch.nn.functional as F


class RectifiedFlow:
    """Rectified flow matching for generative modeling.

    The flow transports samples from noise x_0 ~ N(0,I) to data x_1 ~ p_data
    along straight-line paths. The model learns the velocity field v_theta(x_t, t)
    that defines this transport.
    """

    def __init__(self, model, time_distribution="uniform"):
        """
        Args:
            model: Neural network v_theta(x, t) -> velocity
            time_distribution: How to sample training timesteps
                - "uniform": U[0, 1]
                - "logit_normal": Logit-normal (biased toward mid-range, used by SD3)
                - "u_shaped": Beta(0.5, 0.5) — biased toward endpoints (used by RF++)
        """
        self.model = model
        self.time_distribution = time_distribution

    def sample_timesteps(self, batch_size, device):
        """Sample training timesteps according to the chosen distribution."""
        if self.time_distribution == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.time_distribution == "logit_normal":
            # Logit-normal: t = sigmoid(Normal(0, 1))
            z = torch.randn(batch_size, device=device)
            t = torch.sigmoid(z)
        elif self.time_distribution == "u_shaped":
            # Beta(0.5, 0.5) gives U-shaped distribution
            dist = torch.distributions.Beta(0.5, 0.5)
            t = dist.sample((batch_size,)).to(device)
        else:
            raise ValueError(f"Unknown time distribution: {self.time_distribution}")

        # Clamp to avoid exact 0 or 1
        return t.clamp(1e-5, 1 - 1e-5)

    def compute_loss(self, x_1, x_0=None):
        """Compute the conditional flow matching loss.

        Args:
            x_1: Data samples [B, C, H, W] in [-1, 1]
            x_0: Noise samples [B, C, H, W] ~ N(0, I). If None, sampled automatically.

        Returns:
            loss: Scalar MSE loss
            info: Dict with diagnostics
        """
        B = x_1.shape[0]
        device = x_1.device

        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # Sample timesteps
        t = self.sample_timesteps(B, device)

        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Target velocity: v = x_1 - x_0
        v_target = x_1 - x_0

        # Predicted velocity
        v_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        info = {
            "loss": loss.item(),
            "v_pred_norm": v_pred.detach().flatten(1).norm(dim=1).mean().item(),
            "v_target_norm": v_target.detach().flatten(1).norm(dim=1).mean().item(),
        }

        return loss, info

    @torch.no_grad()
    def sample(self, shape, device, num_steps=100, return_trajectory=False):
        """Generate samples using Euler ODE solver.

        Solves dx/dt = v_theta(x, t) from t=0 (noise) to t=1 (data).

        Args:
            shape: Output shape [B, C, H, W]
            device: Device to generate on
            num_steps: Number of Euler steps (NFE)
            return_trajectory: If True, return all intermediate states

        Returns:
            x: Generated samples [B, C, H, W] in [-1, 1]
            trajectory: (optional) List of intermediate states
        """
        self.model.eval()
        dt = 1.0 / num_steps
        x = torch.randn(shape, device=device)

        trajectory = [x] if return_trajectory else None

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            v = self.model(x, t)
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x)

        self.model.train()

        if return_trajectory:
            return x.clamp(-1, 1), trajectory
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_heun(self, shape, device, num_steps=50):
        """Generate samples using Heun's method (2nd-order ODE solver).

        More accurate than Euler for the same number of function evaluations,
        but uses 2 NFE per step.

        Args:
            shape: Output shape [B, C, H, W]
            device: Device
            num_steps: Number of Heun steps (2*num_steps NFE total)

        Returns:
            x: Generated samples [B, C, H, W]
        """
        self.model.eval()
        dt = 1.0 / num_steps
        x = torch.randn(shape, device=device)

        for i in range(num_steps):
            t_i = i * dt
            t_next = (i + 1) * dt

            t_batch = torch.full((shape[0],), t_i, device=device)
            v_i = self.model(x, t_batch)

            # Euler predictor
            x_euler = x + v_i * dt

            # Corrector
            t_next_batch = torch.full((shape[0],), min(t_next, 1.0 - 1e-5), device=device)
            v_next = self.model(x_euler, t_next_batch)

            # Heun's update
            x = x + 0.5 * (v_i + v_next) * dt

        self.model.train()
        return x.clamp(-1, 1)

    @torch.no_grad()
    def generate_reflow_pairs(self, num_pairs, shape, device, num_steps=50, batch_size=256):
        """Generate (noise, data) pairs for reflow training.

        Runs the learned ODE to create coupled pairs (z_0, z_1) where:
        - z_0 is the initial noise
        - z_1 is the generated sample from that noise

        Training a new model on these pairs produces straighter trajectories.

        Args:
            num_pairs: Total number of pairs to generate
            shape: Single sample shape (C, H, W)
            device: Device
            num_steps: Euler steps for generation
            batch_size: Batch size for generation

        Returns:
            z0_all: Noise samples [N, C, H, W]
            z1_all: Generated samples [N, C, H, W]
        """
        self.model.eval()
        z0_list, z1_list = [], []
        remaining = num_pairs

        while remaining > 0:
            B = min(batch_size, remaining)
            z0 = torch.randn(B, *shape, device=device)

            # Run ODE from z0
            x = z0.clone()
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                v = self.model(x, t)
                x = x + v * dt

            z1 = x.clamp(-1, 1)
            z0_list.append(z0.cpu())
            z1_list.append(z1.cpu())
            remaining -= B

        self.model.train()
        return torch.cat(z0_list), torch.cat(z1_list)
