"""Core Hakuli optimizer implementation.

Hakuli is an independent, clean-room implementation developed by XAEDRO,
written from the published algorithmic description of low-rank-moment
optimizers. No source code from any reference implementation is included.
See the NOTICE file at the repository root for academic attributions.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer

__all__ = ["Hakuli"]


class Hakuli(Optimizer):
    r"""Memory-efficient optimizer for full-parameter LLM training.

    Hakuli keeps Adam-style first- and second-moment estimates in a low-rank
    auxiliary subspace built from a Gaussian random projection of the
    gradient, then applies a channel-wise (or tensor-wise) rescaling of the
    *full* gradient. Optimizer memory is therefore close to SGD while the
    weights receive full-rank updates. No singular value decomposition is
    performed at any point.

    For a 2-D parameter :math:`G_t \in \mathbb{R}^{m \times n}` the update is

    .. math::

        R_t &= P_t^{\top} G_t \\
        M_t &= \beta_1 M_{t-1} + (1 - \beta_1) R_t \\
        V_t &= \beta_2 V_{t-1} + (1 - \beta_2) R_t^2 \\
        S_t &= \frac{M_t}{\sqrt{V_t} + \epsilon} \\
        s_t &= \frac{\lVert S_t \rVert}{\lVert R_t \rVert}
               \quad \text{(per output channel)} \\
        \hat{G}_t &= \sqrt{\lambda}\, s_t \odot G_t \\
        W_t &= W_{t-1} - \eta_t \hat{U}_t,
        \qquad
        \eta_t = \eta \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}

    where :math:`P_t \in \mathbb{R}^{m \times r}` is resampled from a standard
    Gaussian every ``update_proj_gap`` steps, :math:`r` is ``rank`` and
    :math:`\lambda` is ``scale``. When the parameter is wider than it is tall
    the projection is applied on the right and channel norms are taken along
    rows. Parameters whose smallest dimension does not exceed ``rank``
    (including all 1-D parameters) are updated with the bias-corrected
    Adam-style direction :math:`S_t` directly.

    A relative norm-growth limiter (``nl_threshold``) caps the update norm at
    ``nl_threshold`` times the previous step's update norm, converting abrupt
    gradient spikes into smooth growth.

    Args:
        params: Iterable of parameters or parameter-group dictionaries. The
            hyperparameters below can be overridden per group.
        lr: Learning rate (default: ``1e-5``).
        rank: Rank of the auxiliary random subspace (default: ``256``).
        update_proj_gap: Steps between projection resamples (default: ``200``).
        scale: Global multiplier on the structured update, applied as
            ``sqrt(scale)`` (default: ``1.0``).
        scale_type: ``"channel"`` for per-channel rescaling or ``"tensor"``
            for a single scalar per tensor (default: ``"channel"``).
        betas: Moment decay coefficients (default: ``(0.9, 0.999)``).
        eps: Numerical stability term (default: ``1e-8``).
        weight_decay: Decoupled weight decay (default: ``0.0``).
        nl_threshold: Relative cap on update-norm growth between consecutive
            steps (default: ``1.01``).

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from hakuli import Hakuli
        >>> model = nn.Linear(128, 128)
        >>> optimizer = Hakuli(model.parameters(), lr=1e-4, rank=64)
        >>> optimizer.zero_grad()
        >>> loss = model(torch.randn(8, 128)).pow(2).mean()
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        Hakuli was designed for fine-tuning large transformers (attention and
        MLP projection matrices, embeddings). Route 1-D parameters and
        normalization layers to the same optimizer or any standard optimizer;
        Hakuli handles both cases transparently.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-5,
        rank: int = 256,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        scale_type: str = "channel",
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        nl_threshold: float = 1.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank < 0:
            raise ValueError(f"Invalid rank: {rank}")
        if update_proj_gap < 1:
            raise ValueError(f"Invalid update_proj_gap: {update_proj_gap}")
        if scale < 0.0:
            raise ValueError(f"Invalid scale: {scale}")
        if scale_type not in ("channel", "tensor"):
            raise ValueError(
                f"Invalid scale_type: {scale_type!r} (expected 'channel' or 'tensor')"
            )
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nl_threshold <= 0.0:
            raise ValueError(f"Invalid nl_threshold: {nl_threshold}")

        defaults = dict(
            lr=lr,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type=scale_type,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nl_threshold=nl_threshold,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Args:
            closure: Optional callable that re-evaluates the model and
                returns the loss. Executed under ``torch.enable_grad()``.

        Returns:
            The loss returned by ``closure``, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rank = group["rank"]
            eps = group["eps"]
            lr = group["lr"]
            scale = group["scale"]
            weight_decay = group["weight_decay"]
            scale_type = group["scale_type"]
            update_proj_gap = group["update_proj_gap"]
            nl_threshold = group["nl_threshold"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Hakuli does not support sparse gradients")

                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                t = state["step"]

                projected = grad.dim() == 2 and rank > 0 and min(grad.shape) > rank

                if projected:
                    rows, cols = grad.shape
                    if rows >= cols:
                        if "proj" not in state or t % update_proj_gap == 0:
                            state["proj"] = torch.randn(
                                rows, rank, device=grad.device, dtype=torch.bfloat16
                            )
                        low = torch.mm(state["proj"].t().to(grad.dtype), grad).float()
                        channel_dim = 0
                    else:
                        if "proj" not in state or t % update_proj_gap == 0:
                            state["proj"] = torch.randn(
                                cols, rank, device=grad.device, dtype=torch.bfloat16
                            )
                        low = torch.mm(grad, state["proj"].to(grad.dtype)).float()
                        channel_dim = 1

                    exp_avg = state.get("exp_avg")
                    if exp_avg is None:
                        exp_avg = torch.zeros_like(low)
                        exp_avg_sq = torch.zeros_like(low)
                    else:
                        exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(low, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(low, low, value=1.0 - beta2)
                    state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq

                    norm_grad = exp_avg / (exp_avg_sq.sqrt() + eps)
                    if scale_type == "channel":
                        numer = norm_grad.norm(dim=channel_dim)
                        denom = low.norm(dim=channel_dim).clamp(min=1e-8)
                        scaling = numer / denom
                        if channel_dim == 1:
                            scaling = scaling.unsqueeze(1)
                    else:
                        scaling = norm_grad.norm() / low.norm().clamp(min=1e-8)
                    update = grad * scaling.to(grad.dtype)
                else:
                    exp_avg = state.get("exp_avg")
                    if exp_avg is None:
                        exp_avg = torch.zeros_like(grad)
                        exp_avg_sq = torch.zeros_like(grad)
                    else:
                        exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq

                    update = exp_avg / (exp_avg_sq.sqrt() + eps)

                update = update * math.sqrt(max(scale, 0.0))

                update_norm = update.norm().item()
                prev_norm = state.get("prev_norm")
                if prev_norm is not None and prev_norm > 0.0:
                    limiter = max(update_norm / (prev_norm + 1e-8), nl_threshold) / nl_threshold
                    update = update / limiter
                    state["prev_norm"] = update_norm / limiter
                else:
                    state["prev_norm"] = update_norm

                step_size = lr * math.sqrt(1.0 - beta2**t) / max(1.0 - beta1**t, 1e-16)
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-step_size)

        return loss
