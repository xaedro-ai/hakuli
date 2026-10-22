"""Core Hakuli optimizer implementation.

Hakuli is an independent, clean-room implementation developed by XAEDRO,
written from the published algorithmic description of low-rank-moment
optimizers. No source code from any reference implementation is included.
See the NOTICE file at the repository root for academic attributions.

Version 2.0 is an engineering and efficiency revision of the 1.x algorithm.
The optimizer mathematics is unchanged; what changed is how it is executed:

- The step is synchronization-free: no ``.item()`` / ``float(tensor)`` /
  truth-value round-trips to the host anywhere in ``step()``, so devices
  never stall waiting for scalar results.
- The parameter update is applied as a single fused ``addcmul_`` pass
  (``w <- w - eta * g * s``) instead of materializing and rescaling a
  full-size update tensor several times.
- The projection is stored in matmul-ready layout (transposed once at
  resample time, never per step) and in the gradient's own dtype, so no
  per-step casts of ``rank x max(m, n)`` buffers are needed.
- The bias/norm path (non-2-D and small parameters) can execute through
  batched ``torch._foreach_*`` kernels (``foreach=True``).
- ``limiter_norm="projected"`` evaluates the norm-growth limiter in the
  ``rank``-dimensional projection subspace, removing the last full-size
  temporary from the step and making the limiter cost independent of the
  parameter size.
- ``transport_moments`` (default off, experimental) re-expresses the stored
  moments in the new basis at every resample using the ``rank x rank``
  overlap between consecutive projections. Measured consistently worse than
  letting the old-basis content decay, so it is kept only as a documented
  negative result and an option for further research.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

__all__ = ["Hakuli"]


def _limiter_factor(
    update_norm: torch.Tensor, prev_norm: torch.Tensor, threshold: float
) -> torch.Tensor:
    raw = torch.clamp_min(update_norm / (prev_norm + 1e-8), threshold) / threshold
    return torch.where(prev_norm > 0, raw, torch.ones_like(raw))


def _device_scalar(state: dict, grad: torch.Tensor) -> Optional[torch.Tensor]:
    prev = state.get("prev_norm")
    if prev is None or isinstance(prev, torch.Tensor):
        return prev
    return torch.as_tensor(prev, dtype=torch.float32, device=grad.device)


def _draw_proj(a: int, b: int, grad: torch.Tensor) -> torch.Tensor:
    dtype = grad.dtype if grad.dtype.is_floating_point else torch.float32
    return torch.randn(a, b, device=grad.device, dtype=dtype)


def _transport_moments(
    state: dict, old_proj: torch.Tensor, new_proj: torch.Tensor, left: bool
) -> None:
    exp_avg = state.get("exp_avg")
    if exp_avg is None:
        return
    old = old_proj.float()
    new = new_proj.float()
    if left:
        a = (new @ old.t()) / old.shape[0]
    else:
        a = (old.t() @ new) / old.shape[1]
    a = a.to(exp_avg.dtype)
    if left:
        state["exp_avg"] = a @ exp_avg
        state["exp_avg_sq"] = (a * a) @ state["exp_avg_sq"]
    else:
        state["exp_avg"] = exp_avg @ a
        state["exp_avg_sq"] = state["exp_avg_sq"] @ (a * a)


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
    gradient spikes into smooth growth. With ``limiter_norm="projected"`` the
    limiter operates on the estimate

    .. math::

        \lVert U_t \rVert \approx
        \sqrt{\lambda}\,\frac{\lVert s_t \odot R_t \rVert}{\sqrt{r}},

    which follows from :math:`\mathbb{E}\,\lVert P_t^{\top} g \rVert^2 =
    r\,\lVert g \rVert^2` (Johnson-Lindenstrauss norm preservation), instead
    of materializing the full-size update to measure its norm.

    Version 2.0 details: the step never synchronizes the host, the update is
    applied with one fused ``addcmul_``, projections are cached in
    matmul-ready layout and gradient dtype, and the fallback path can run
    through batched ``torch._foreach_*`` kernels.

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
        limiter_norm: ``"exact"`` measures the update norm over the full
            parameter; ``"projected"`` estimates it in the projection
            subspace, removing the full-size temporary at the cost of
            Johnson-Lindenstrauss distortion (default: ``"exact"``).
        foreach: Batch the bias/norm update path with ``torch._foreach_*``
            kernels when more than one parameter shares a device and dtype
            (default: ``False``).
        transport_moments: At every projection resample, re-express the
            stored moments in the new basis with the ``rank x rank`` overlap
            operator between the old and new projections,
            :math:`M \leftarrow A M` with
            :math:`A = P_{\text{new}} P_{\text{old}}^{\top} / r` (and
            :math:`V \leftarrow (A \odot A) V` as its diagonal second-moment
            counterpart), instead of letting the old-basis content decay
            exponentially. Experimental: measured consistently worse than
            decay (+0.08 nats at gap=50, +0.007 at gap=200, three seeds, tiny
            GPT), because the transported
            old-basis content is mostly moment noise. Default: ``False``.
        denom: Denominator of the channel scale. ``"grad"`` (default) uses
            projected-gradient norms (APOLLO-style); ``"update"`` uses an
            AdaDelta-style EMA of the applied subspace-update norms, which is
            unit-consistent and removes the rank-dependent ``scale``
            sensitivity. Experimental.
        direction: ``"adam"`` (default) or ``"atan2"``: ``atan2(m, sqrt(v))``
            is bounded, epsilon-free and exactly gradient-scale invariant
            inside the subspace. Experimental.
        norm_estimator: For ``limiter_norm="projected"``: ``"sqrt_rank"``
            (default) or ``"chi"``, the unbiased norm estimator
            ``E[chi_r] = sqrt(2) * Gamma((r+1)/2) / Gamma(r/2)`` (Lehmann-
            Scheffe for Gaussian sketches; matters at small rank).
            Experimental.
        anti_windup: When the norm limiter saturates, correct the moment
            accumulators so they integrate the applied (attenuated) step
            instead of the raw one (control-theory anti-windup). Experimental.
        nl_schedule: ``"fixed"`` (default) or ``"record"``: the growth
            threshold relaxes toward 1 as ``1 + (thr-1)*sqrt(10/t)``,
            record-process style. Experimental.
        delta_rho: EMA rate of the ``denom="update"`` accumulator.
        scale_type: also accepts ``"two_sided"``: channel scale multiplied by
            a per-row/column norm balance (uses one full-size temporary).
            Experimental.

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
        limiter_norm: str = "exact",
        foreach: bool = False,
        transport_moments: bool = False,
        denom: str = "grad",
        direction: str = "adam",
        norm_estimator: str = "sqrt_rank",
        anti_windup: bool = False,
        nl_schedule: str = "fixed",
        delta_rho: float = 0.95,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank < 0:
            raise ValueError(f"Invalid rank: {rank}")
        if update_proj_gap < 1:
            raise ValueError(f"Invalid update_proj_gap: {update_proj_gap}")
        if scale < 0.0:
            raise ValueError(f"Invalid scale: {scale}")
        if scale_type not in ("channel", "tensor", "two_sided"):
            raise ValueError(
                f"Invalid scale_type: {scale_type!r} "
                "(expected 'channel', 'tensor' or 'two_sided')"
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
        if limiter_norm not in ("exact", "projected"):
            raise ValueError(
                f"Invalid limiter_norm: {limiter_norm!r} (expected 'exact' or 'projected')"
            )
        if not isinstance(foreach, bool):
            raise ValueError(f"Invalid foreach: {foreach!r} (expected bool)")
        if not isinstance(transport_moments, bool):
            raise ValueError(
                f"Invalid transport_moments: {transport_moments!r} (expected bool)"
            )
        if denom not in ("grad", "update"):
            raise ValueError(
                f"Invalid denom: {denom!r} (expected 'grad' or 'update')"
            )
        if direction not in ("adam", "atan2"):
            raise ValueError(
                f"Invalid direction: {direction!r} (expected 'adam' or 'atan2')"
            )
        if norm_estimator not in ("sqrt_rank", "chi"):
            raise ValueError(
                f"Invalid norm_estimator: {norm_estimator!r} "
                "(expected 'sqrt_rank' or 'chi')"
            )
        if not isinstance(anti_windup, bool):
            raise ValueError(f"Invalid anti_windup: {anti_windup!r} (expected bool)")
        if nl_schedule not in ("fixed", "record"):
            raise ValueError(
                f"Invalid nl_schedule: {nl_schedule!r} (expected 'fixed' or 'record')"
            )
        if not 0.0 < delta_rho < 1.0:
            raise ValueError(f"Invalid delta_rho: {delta_rho}")

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
            limiter_norm=limiter_norm,
            foreach=foreach,
            transport_moments=transport_moments,
            denom=denom,
            direction=direction,
            norm_estimator=norm_estimator,
            anti_windup=anti_windup,
            nl_schedule=nl_schedule,
            delta_rho=delta_rho,
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
            if not group["params"]:
                continue
            rank = group["rank"]
            projected: List[torch.Tensor] = []
            fallback: List[torch.Tensor] = []
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Hakuli does not support sparse gradients")
                if grad.dim() == 2 and rank > 0 and min(grad.shape) > rank:
                    projected.append(p)
                else:
                    fallback.append(p)
            for p in projected:
                self._step_projected(p, group)
            if group["foreach"] and len(fallback) > 1:
                self._step_fallback_foreach(fallback, group)
            else:
                for p in fallback:
                    self._step_fallback_single(p, group)

        return loss

    def _step_projected(self, p: torch.Tensor, group: dict) -> None:
        grad = p.grad
        state = self.state[p]
        t = state.get("step", 0) + 1
        state["step"] = t

        beta1, beta2 = group["betas"]
        rank = group["rank"]
        eps = group["eps"]
        lr = group["lr"]
        scale = math.sqrt(max(group["scale"], 0.0))
        weight_decay = group["weight_decay"]
        gap = group["update_proj_gap"]
        threshold = group["nl_threshold"]

        rows, cols = grad.shape
        if rows >= cols:
            if "proj" not in state:
                state["proj"] = _draw_proj(rank, rows, grad)
            elif t % gap == 0:
                new_proj = _draw_proj(rank, rows, grad)
                if group["transport_moments"]:
                    _transport_moments(state, state["proj"], new_proj, left=True)
                state["proj"] = new_proj
            low = torch.mm(state["proj"], grad)
            channel_dim = 0
        else:
            if "proj" not in state:
                state["proj"] = _draw_proj(cols, rank, grad)
            elif t % gap == 0:
                new_proj = _draw_proj(cols, rank, grad)
                if group["transport_moments"]:
                    _transport_moments(state, state["proj"], new_proj, left=False)
                state["proj"] = new_proj
            low = torch.mm(grad, state["proj"])
            channel_dim = 1
        if low.dtype != torch.float32:
            low = low.float()

        exp_avg = state.get("exp_avg")
        if exp_avg is None:
            exp_avg = torch.zeros_like(low)
            exp_avg_sq = torch.zeros_like(low)
        else:
            exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(low, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(low, low, value=1.0 - beta2)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        if group["direction"] == "atan2":
            direction = torch.atan2(exp_avg, exp_avg_sq.sqrt())
        else:
            direction = exp_avg / (exp_avg_sq.sqrt() + eps)

        threshold_eff = threshold
        if group["nl_schedule"] == "record":
            threshold_eff = 1.0 + (threshold - 1.0) * math.sqrt(10.0 / max(t, 10))

        scale_type = group["scale_type"]
        two_sided = scale_type == "two_sided"
        if scale_type == "channel":
            numer = torch.linalg.vector_norm(direction, dim=channel_dim)
            if group["denom"] == "update":
                prev_delta = state.get("delta_rms")
                if prev_delta is None:
                    denom_vec = torch.linalg.vector_norm(
                        low, dim=channel_dim
                    ).clamp_min(1e-8)
                else:
                    denom_vec = prev_delta.clamp_min(0.0).sqrt().clamp_min(1e-8)
                scaling = numer / denom_vec
                low_cn = torch.linalg.vector_norm(low, dim=channel_dim)
                delta_sq = (scaling * low_cn) ** 2
                rho = group["delta_rho"]
                if prev_delta is None:
                    state["delta_rms"] = delta_sq
                else:
                    state["delta_rms"] = rho * prev_delta + (1.0 - rho) * delta_sq
            else:
                denom_vec = torch.linalg.vector_norm(
                    low, dim=channel_dim
                ).clamp_min(1e-8)
                scaling = numer / denom_vec
            scaling = scaling.unsqueeze(1) if channel_dim == 1 else scaling.unsqueeze(0)
        elif two_sided:
            numer = torch.linalg.vector_norm(direction, dim=channel_dim)
            denom_vec = torch.linalg.vector_norm(low, dim=channel_dim).clamp_min(1e-8)
            scaling = numer / denom_vec
            scaling = scaling.unsqueeze(1) if channel_dim == 1 else scaling.unsqueeze(0)
            other = torch.linalg.vector_norm(grad, dim=1 - channel_dim).clamp_min(1e-8)
            s_other = other.mean() / other
            s_other = s_other / s_other.mean()
            s_other = s_other.unsqueeze(1) if channel_dim == 0 else s_other.unsqueeze(0)
            scaling = scaling * s_other
        else:
            scaling = direction.norm() / low.norm().clamp_min(1e-8)

        if two_sided:
            scaled = grad * scaling.to(grad.dtype)
            update_norm = torch.linalg.vector_norm(scaled).float() * scale
            prev = _device_scalar(state, grad)
            if prev is None:
                state["prev_norm"] = update_norm
                scaled_out = scaled * scale
            else:
                factor = _limiter_factor(update_norm, prev, threshold_eff)
                state["prev_norm"] = update_norm / factor
                scaled_out = scaled * (scale / factor)
            step_size = lr * math.sqrt(1.0 - beta2**t) / max(1.0 - beta1**t, 1e-16)
            if weight_decay > 0:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(scaled_out, alpha=-step_size)
            return

        if group["limiter_norm"] == "projected":
            r_eff = math.sqrt(rank)
            if group["norm_estimator"] == "chi":
                r_eff = math.sqrt(2.0) * math.exp(
                    math.lgamma((rank + 1.0) / 2.0) - math.lgamma(rank / 2.0)
                )
            update_norm = torch.linalg.vector_norm(low * scaling) / r_eff * scale
        else:
            gs = grad * scaling.to(grad.dtype)
            update_norm = torch.linalg.vector_norm(gs).float() * scale

        prev = _device_scalar(state, grad)
        if prev is None:
            state["prev_norm"] = update_norm
            scaling_out = scaling * scale
        else:
            factor = _limiter_factor(update_norm, prev, threshold_eff)
            state["prev_norm"] = update_norm / factor
            scaling_out = scaling * (scale / factor)
            if group["anti_windup"]:
                excess = (1.0 - 1.0 / factor).clamp_min(0.0).to(low.dtype)
                exp_avg.sub_(low * excess, alpha=1.0 - beta1)
                excess_v = (1.0 - 1.0 / (factor * factor)).clamp_min(0.0).to(low.dtype)
                exp_avg_sq.sub_(low * low * excess_v, alpha=1.0 - beta2)

        step_size = lr * math.sqrt(1.0 - beta2**t) / max(1.0 - beta1**t, 1e-16)
        if weight_decay > 0:
            p.mul_(1.0 - lr * weight_decay)
        p.addcmul_(grad, scaling_out.to(grad.dtype), value=-step_size)

    def _step_fallback_single(self, p: torch.Tensor, group: dict) -> None:
        grad = p.grad
        state = self.state[p]
        t = state.get("step", 0) + 1
        state["step"] = t

        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        scale = math.sqrt(max(group["scale"], 0.0))
        weight_decay = group["weight_decay"]
        threshold = group["nl_threshold"]

        exp_avg = state.get("exp_avg")
        if exp_avg is None:
            exp_avg = torch.zeros_like(grad)
            exp_avg_sq = torch.zeros_like(grad)
        else:
            exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        update = exp_avg / (exp_avg_sq.sqrt() + eps)
        update_norm = update.norm().float() * scale

        prev = _device_scalar(state, grad)
        if prev is None:
            state["prev_norm"] = update_norm
            update.mul_(scale)
        else:
            factor = _limiter_factor(update_norm, prev, threshold)
            state["prev_norm"] = update_norm / factor
            update.mul_(scale).div_(factor.to(update.dtype))

        step_size = lr * math.sqrt(1.0 - beta2**t) / max(1.0 - beta1**t, 1e-16)
        if weight_decay > 0:
            p.mul_(1.0 - lr * weight_decay)
        p.add_(update, alpha=-step_size)

    def _step_fallback_foreach(self, params: List[torch.Tensor], group: dict) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        scale = math.sqrt(max(group["scale"], 0.0))
        weight_decay = group["weight_decay"]
        threshold = group["nl_threshold"]

        buckets: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = {}
        singles: List[torch.Tensor] = []
        for p in params:
            if p.grad.dtype == p.dtype:
                buckets.setdefault((p.device, p.dtype), []).append(p)
            else:
                singles.append(p)
        for bucket in buckets.values():
            if len(bucket) > 1:
                self._foreach_bucket(bucket, group)
            else:
                singles.extend(bucket)
        for p in singles:
            self._step_fallback_single(p, group)

    def _foreach_bucket(self, bucket: List[torch.Tensor], group: dict) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        scale = math.sqrt(max(group["scale"], 0.0))
        weight_decay = group["weight_decay"]
        threshold = group["nl_threshold"]

        grads = [p.grad for p in bucket]
        states = [self.state[p] for p in bucket]
        steps = []
        for st in states:
            st["step"] = t = st.get("step", 0) + 1
            steps.append(t)

        avgs: List[torch.Tensor] = []
        avgsq: List[torch.Tensor] = []
        for i, st in enumerate(states):
            if "exp_avg" in st:
                avgs.append(st["exp_avg"])
                avgsq.append(st["exp_avg_sq"])
            else:
                avgs.append(torch.zeros_like(grads[i]))
                avgsq.append(torch.zeros_like(grads[i]))
                st["exp_avg"], st["exp_avg_sq"] = avgs[-1], avgsq[-1]

        torch._foreach_mul_(avgs, beta1)
        torch._foreach_add_(avgs, grads, alpha=1.0 - beta1)
        torch._foreach_mul_(avgsq, beta2)
        torch._foreach_addcmul_(avgsq, grads, grads, value=1.0 - beta2)
        dens = torch._foreach_sqrt(avgsq)
        torch._foreach_add_(dens, eps)
        updates = torch._foreach_div(avgs, dens)
        norms = torch._foreach_norm(updates)

        for p, g, st, u, un, t in zip(bucket, grads, states, updates, norms, steps):
            update_norm = un.float() * scale
            prev = _device_scalar(st, g)
            if prev is None:
                st["prev_norm"] = update_norm
                u.mul_(scale)
            else:
                factor = _limiter_factor(update_norm, prev, threshold)
                st["prev_norm"] = update_norm / factor
                u.mul_(scale).div_(factor.to(u.dtype))
            step_size = lr * math.sqrt(1.0 - beta2**t) / max(1.0 - beta1**t, 1e-16)
            if weight_decay > 0:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(u, alpha=-step_size)
