"""Integration tests: end-to-end training behaviour of Hakuli."""

import math

import torch

from hakuli import Hakuli


def _make_problem(seed=0, n=1024, d=32):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    w = torch.randn(d, d, generator=generator) / math.sqrt(d)
    y = x @ w
    return x, y


def test_regression_loss_decreases():
    torch.manual_seed(0)
    x, y = _make_problem()
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 32),
    )
    optimizer = Hakuli(model.parameters(), lr=3e-3, rank=8, update_proj_gap=50)

    def loss_fn():
        return (model(x) - y).pow(2).mean()

    initial_loss = loss_fn().item()
    assert math.isfinite(initial_loss)

    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn().item()
    assert math.isfinite(final_loss)
    assert final_loss < 0.2 * initial_loss


def test_closure_is_supported_and_returns_loss():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    optimizer = Hakuli(model.parameters(), lr=1e-4, rank=0)
    data = torch.randn(16, 8)

    def closure():
        optimizer.zero_grad()
        loss = model(data).pow(2).mean()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None
    assert math.isfinite(loss.item())


def test_gradient_accumulation_matches_single_accumulated_step():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 4)
    params_a = [p.detach().clone().requires_grad_() for p in base.parameters()]
    params_b = [p.detach().clone().requires_grad_() for p in base.parameters()]

    opt_a = Hakuli(params_a, lr=1e-3, rank=0)
    opt_b = Hakuli(params_b, lr=1e-3, rank=0)

    batch1 = torch.randn(8, 8)
    batch2 = torch.randn(8, 8)
    names = ("weight", "bias")

    opt_a.zero_grad()
    loss = torch.func.functional_call(base, dict(zip(names, params_a)), (batch1,))
    (loss.pow(2).mean() / 2).backward()
    loss = torch.func.functional_call(base, dict(zip(names, params_a)), (batch2,))
    (loss.pow(2).mean() / 2).backward()
    opt_a.step()

    opt_b.zero_grad()
    fused_input = torch.cat([batch1, batch2], dim=0)
    loss = torch.func.functional_call(base, dict(zip(names, params_b)), (fused_input,))
    loss.pow(2).mean().backward()
    opt_b.step()

    for pa, pb in zip(params_a, params_b):
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-5, rtol=1e-3)
