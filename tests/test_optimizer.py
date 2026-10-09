"""Unit tests for the Hakuli optimizer."""

import copy

import torch
import pytest

from hakuli import Hakuli


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


def test_step_updates_projected_param_and_state_shapes():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=10)
    before = p.detach().clone()

    p.grad = torch.randn(32, 16)
    opt.step()

    assert not torch.equal(p.detach(), before)
    state = opt.state[p]
    assert state["proj"].shape == (32, 4)
    assert state["proj"].dtype == torch.bfloat16
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

    assert opt.state[p]["prev_norm"] == pytest.approx(1.01e-6, rel=0.05)


def test_zero_gradient_step_does_not_freeze_parameter():
    torch.manual_seed(0)
    weight = torch.randn(16, 8)
    p = torch.nn.Parameter(weight.clone())
    opt = Hakuli([p], lr=1e-3, rank=4)

    p.grad = torch.zeros_like(p)
    opt.step()
    assert opt.state[p]["prev_norm"] == 0.0

    p.grad = torch.randn(16, 8)
    before = p.detach().clone()
    opt.step()
    assert (p.detach() - before).norm() > 1e-6
    assert opt.state[p]["prev_norm"] > 1e-3


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
