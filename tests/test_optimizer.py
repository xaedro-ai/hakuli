"""Unit tests for the Hakuli optimizer (v2)."""

import copy
import math

import pytest
import torch

from hakuli import Hakuli


def _step_all(opt):
    opt.step()


def test_validation_errors():
    params = [torch.nn.Parameter(torch.zeros(4, 4))]
    with pytest.raises(ValueError):
        Hakuli(params, lr=-1.0)
    with pytest.raises(ValueError):
        Hakuli(params, rank=-1)
    with pytest.raises(ValueError):
        Hakuli(params, update_proj_gap=0)
    with pytest.raises(ValueError):
        Hakuli(params, scale=-1.0)
    with pytest.raises(ValueError):
        Hakuli(params, scale_type="diagonal")
    with pytest.raises(ValueError):
        Hakuli(params, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        Hakuli(params, eps=-1e-8)
    with pytest.raises(ValueError):
        Hakuli(params, weight_decay=-0.1)
    with pytest.raises(ValueError):
        Hakuli(params, nl_threshold=0.0)
    with pytest.raises(ValueError):
        Hakuli(params, limiter_norm="quantum")
    with pytest.raises(ValueError):
        Hakuli(params, foreach=1)


def test_step_updates_projected_param_and_state_shapes():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=10)
    before = p.detach().clone()

    p.grad = torch.randn(32, 16)
    opt.step()

    assert not torch.equal(p.detach(), before)
    state = opt.state[p]
    assert state["proj"].shape == (4, 32)
    assert state["proj"].dtype == p.dtype
    assert state["exp_avg"].shape == (4, 16)
    assert state["exp_avg_sq"].shape == (4, 16)
    assert state["exp_avg"].dtype == torch.float32


def test_wide_param_projects_on_the_right():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 32))
    opt = Hakuli([p], lr=1e-3, rank=4)
    p.grad = torch.randn(16, 32)
    opt.step()

    state = opt.state[p]
    assert state["proj"].shape == (32, 4)
    assert state["exp_avg"].shape == (16, 4)


def test_small_and_1d_params_use_adam_path():
    torch.manual_seed(0)
    small = torch.nn.Parameter(torch.randn(4, 4))
    vector = torch.nn.Parameter(torch.randn(8))
    opt = Hakuli([small, vector], lr=1e-3, rank=8)

    small.grad = torch.randn(4, 4)
    vector.grad = torch.randn(8)
    opt.step()

    for p in (small, vector):
        state = opt.state[p]
        assert "proj" not in state
        assert state["exp_avg"].shape == p.shape


def test_no_svd_is_performed(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Hakuli must not call any SVD routine")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)
    monkeypatch.setattr(torch, "svd", forbidden)

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=5)
    for _ in range(6):
        p.grad = torch.randn(32, 16)
        opt.step()


def test_no_host_synchronization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("step() must not synchronize with the host")

    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(32, 16)),
        torch.nn.Parameter(torch.randn(16, 32)),
        torch.nn.Parameter(torch.randn(24)),
        torch.nn.Parameter(torch.randn(6, 6)),
    ]
    opt = Hakuli(
        params, lr=1e-3, rank=4, update_proj_gap=3, limiter_norm="projected"
    )
    grads = [torch.randn_like(p) for p in params]

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    if hasattr(torch.Tensor, "__float__"):
        monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    try:
        for _ in range(4):
            for p, g in zip(params, grads):
                p.grad = g.clone()
            opt.step()

        params2 = [
            torch.nn.Parameter(torch.randn(32, 16)),
            torch.nn.Parameter(torch.randn(24)),
            torch.nn.Parameter(torch.randn(6, 6)),
        ]
        opt2 = Hakuli(params2, lr=1e-3, rank=4, foreach=True)
        for _ in range(3):
            for p in params2:
                p.grad = torch.randn_like(p)
            opt2.step()
    finally:
        monkeypatch.undo()


def test_projection_resampled_at_gap():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=3)

    p.grad = torch.randn(32, 16)
    opt.step()
    first_projection = opt.state[p]["proj"].clone()

    p.grad = torch.randn(32, 16)
    opt.step()
    assert torch.equal(opt.state[p]["proj"], first_projection)

    torch.manual_seed(12345)
    p.grad = torch.randn(32, 16)
    opt.step()
    assert not torch.equal(opt.state[p]["proj"], first_projection)


def test_norm_growth_limiter_caps_update_growth():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, nl_threshold=1.01)

    p.grad = torch.randn(32, 16)
    opt.step()

    opt.state[p]["prev_norm"] = 1e-6
    p.grad = torch.randn(32, 16)
    opt.step()

    assert float(opt.state[p]["prev_norm"]) == pytest.approx(1.01e-6, rel=0.05)


def test_norm_growth_limiter_projected_matches_exact_estimate():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(512, 384))
    opt = Hakuli([p], lr=1e-3, rank=128, limiter_norm="projected")
    p.grad = torch.randn(512, 384)
    opt.step()

    state = opt.state[p]
    low = torch.mm(state["proj"], p.grad)
    direction = state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1e-8)
    scaling = direction.norm(dim=0) / low.norm(dim=0).clamp_min(1e-8)
    exact = (p.grad * scaling.unsqueeze(0)).norm()
    estimate = float(state["prev_norm"])
    assert estimate == pytest.approx(float(exact), rel=0.25)


def test_zero_gradient_step_does_not_freeze_parameter():
    torch.manual_seed(0)
    weight = torch.randn(16, 8)
    p = torch.nn.Parameter(weight.clone())
    opt = Hakuli([p], lr=1e-3, rank=4)

    p.grad = torch.zeros_like(p)
    opt.step()
    assert float(opt.state[p]["prev_norm"]) == 0.0

    p.grad = torch.randn(16, 8)
    before = p.detach().clone()
    opt.step()
    assert (p.detach() - before).norm() > 1e-6
    assert float(opt.state[p]["prev_norm"]) > 1e-3


def test_weight_decay_decays_param_without_grad():
    p = torch.nn.Parameter(torch.ones(8))
    opt = Hakuli([p], lr=0.1, weight_decay=0.5, rank=0)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert p.mean().item() == pytest.approx(1.0 - 0.1 * 0.5, rel=1e-4)


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt1 = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=2)
    grad = torch.randn(16, 8)

    for step in range(3):
        p.grad = grad.clone()
        torch.manual_seed(100 + step)
        opt1.step()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = Hakuli([p2], lr=1e-3, rank=4, update_proj_gap=2)
    opt2.load_state_dict(copy.deepcopy(opt1.state_dict()))

    p.grad = grad.clone()
    p2.grad = grad.clone()
    torch.manual_seed(200)
    opt1.step()
    torch.manual_seed(200)
    opt2.step()

    assert torch.allclose(p.detach(), p2.detach(), atol=1e-6)


def test_foreach_matches_single_path():
    torch.manual_seed(0)
    sizes = [4, 8, 16]
    shapes = [(s, 4) if i % 2 else (4, s) for i, s in enumerate(sizes)]
    shapes += [(s,) for s in sizes]

    base = [torch.randn(*s) for s in shapes]
    params_a = [torch.nn.Parameter(w.clone()) for w in base]
    params_b = [torch.nn.Parameter(w.clone()) for w in base]
    opt_a = Hakuli(params_a, lr=1e-3, rank=8, foreach=False, weight_decay=0.01)
    opt_b = Hakuli(params_b, lr=1e-3, rank=8, foreach=True, weight_decay=0.01)

    torch.manual_seed(42)
    grads = [torch.randn_like(p) for p in params_a]
    for _ in range(5):
        for pa, pb, g in zip(params_a, params_b, grads):
            pa.grad = g.clone()
            pb.grad = g.clone()
        opt_a.step()
        opt_b.step()

    for pa, pb in zip(params_a, params_b):
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-6)
        assert float(opt_a.state[pa]["prev_norm"]) == pytest.approx(
            float(opt_b.state[pb]["prev_norm"]), rel=1e-5
        )


def test_foreach_with_non_2d_only_and_weight_decay():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(32)) for _ in range(6)]
    opt = Hakuli(params, lr=1e-2, foreach=True, weight_decay=0.1)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    assert all("proj" not in opt.state[p] for p in params)


def test_tensor_scale_type():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(48, 32))
    opt = Hakuli([p], lr=1e-3, rank=8, scale_type="tensor")
    before = p.detach().clone()
    p.grad = torch.randn(48, 32)
    opt.step()
    assert not torch.equal(p.detach(), before)
    assert opt.state[p]["exp_avg"].shape == (8, 32)


def test_param_groups_override():
    torch.manual_seed(0)
    big = torch.nn.Parameter(torch.randn(32, 16))
    small = torch.nn.Parameter(torch.randn(16, 8))
    opt = Hakuli(
        [
            {"params": [big], "rank": 4, "scale": 2.0},
            {"params": [small], "rank": 64},
        ],
        lr=1e-3,
    )
    big.grad = torch.randn(32, 16)
    small.grad = torch.randn(16, 8)
    opt.step()
    assert "proj" in opt.state[big]
    assert "proj" not in opt.state[small]


def test_grad_none_is_skipped():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(16, 8))
    p2 = torch.nn.Parameter(torch.randn(16, 8))
    opt = Hakuli([p1, p2], lr=1e-3, rank=4)
    p1.grad = torch.randn(16, 8)
    opt.step()
    assert "step" in opt.state[p1]
    assert p2 not in opt.state or "step" not in opt.state[p2]


def test_sparse_gradients_raise():
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Hakuli([p], lr=1e-3, rank=2)
    p.grad = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), (8, 8)
    )
    with pytest.raises(RuntimeError):
        opt.step()


def test_closure_returns_loss():
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Hakuli([p], lr=1e-3, rank=4)

    def closure():
        opt.zero_grad()
        loss = p.pow(2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert loss.item() > 0


def test_convergence_smoke():
    torch.manual_seed(0)
    x = torch.randn(256, 8)
    w_true = torch.randn(8, 4)
    y = x @ w_true
    w = torch.nn.Parameter(torch.zeros(8, 4))
    opt = Hakuli([w], lr=1e-2, rank=0)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = (x @ w - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.25 * losses[0]
