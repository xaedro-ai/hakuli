"""Numerical equivalence of the fused v2 step against a literal fp64
implementation of the documented update rule."""

import math

import pytest
import torch

from hakuli import Hakuli


def reference_run(p_init, grads, proj, lr, rank, scale, betas, eps, wd, thr,
                  limiter_norm, scale_type="channel"):
    beta1, beta2 = betas
    p = p_init.double().clone()
    proj64 = proj.double()
    tall = p_init.shape[0] >= p_init.shape[1]
    m, n = p_init.shape
    exp_avg = None
    exp_avg_sq = None
    prev = None

    for t, g in enumerate(grads, start=1):
        g = g.double()
        low = proj64 @ g if tall else g @ proj64
        if exp_avg is None:
            exp_avg = (1.0 - beta1) * low
            exp_avg_sq = (1.0 - beta2) * low * low
        else:
            exp_avg = beta1 * exp_avg + (1.0 - beta1) * low
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * low * low
        direction = exp_avg / (exp_avg_sq.sqrt() + eps)
        if scale_type == "channel":
            dim = 0 if tall else 1
            s = direction.norm(dim=dim) / low.norm(dim=dim).clamp_min(1e-8)
            s_view = s.unsqueeze(0) if tall else s.unsqueeze(1)
        else:
            s_view = direction.norm() / low.norm().clamp_min(1e-8)

        c = math.sqrt(scale)
        if limiter_norm == "projected":
            un = (low * s_view).norm() / math.sqrt(rank) * c
        else:
            un = (g * s_view).norm() * c

        if prev is None or prev == 0.0:
            factor = 1.0
        else:
            factor = max(float(un) / prev, thr) / thr
        prev = float(un) / factor

        eta = lr * math.sqrt(1.0 - beta2**t) / max(1.0 - beta1**t, 1e-16)
        p = p * (1.0 - lr * wd) if wd > 0 else p
        p = p - eta * (g * s_view) * (c / factor)

    return p, prev


@pytest.mark.parametrize("shape", [(32, 16), (16, 32), (24, 24)])
@pytest.mark.parametrize("limiter_norm", ["exact", "projected"])
@pytest.mark.parametrize("scale_type", ["channel", "tensor"])
def test_fused_step_matches_fp64_reference(shape, limiter_norm, scale_type):
    torch.manual_seed(0)
    lr, rank, scale = 1e-3, 4, 2.0
    betas = (0.9, 0.999)
    eps, wd, thr = 1e-8, 0.01, 1.01
    steps = 8

    p_init = torch.randn(*shape)
    grads = [torch.randn(*shape) for _ in range(steps)]

    probe = torch.nn.Parameter(p_init.clone())
    probe_opt = Hakuli([probe], lr=lr, rank=rank, update_proj_gap=10**6)
    probe.grad = grads[0].clone()
    probe_opt.step()
    proj = probe_opt.state[probe]["proj"].clone()

    p = torch.nn.Parameter(p_init.clone())
    opt = Hakuli(
        [p], lr=lr, rank=rank, scale=scale, betas=betas, eps=eps,
        weight_decay=wd, nl_threshold=thr, limiter_norm=limiter_norm,
        scale_type=scale_type, update_proj_gap=10**6,
    )
    opt.state[p]["proj"] = proj.clone()
    for g in grads:
        p.grad = g.clone()
        opt.step()

    ref_p, ref_prev = reference_run(
        p_init, grads, proj, lr, rank, scale, betas, eps, wd, thr,
        limiter_norm, scale_type,
    )

    assert torch.allclose(p.detach().double(), ref_p, atol=5e-6, rtol=5e-6)
    assert float(opt.state[p]["prev_norm"]) == pytest.approx(ref_prev, rel=1e-4)


def test_projected_limiter_trajectory_close_to_exact():
    torch.manual_seed(0)
    shape = (256, 192)
    rank = 64
    p_init = torch.randn(*shape)
    grads = [torch.randn(*shape) for _ in range(20)]

    def run(limiter_norm):
        p = torch.nn.Parameter(p_init.clone())
        opt = Hakuli(
            [p], lr=1e-3, rank=rank, update_proj_gap=10**6,
            limiter_norm=limiter_norm,
        )
        for g in grads:
            p.grad = g.clone()
            opt.step()
        return p.detach()

    pe = run("exact")
    pp = run("projected")
    rel = (pe - pp).norm() / pe.norm()
    assert rel < 0.05


def test_transport_moments_math_tall():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=2, transport_moments=True)

    g1 = torch.randn(32, 16)
    p.grad = g1
    opt.step()
    old_proj = opt.state[p]["proj"].clone()
    m1 = opt.state[p]["exp_avg"].clone()
    v1 = opt.state[p]["exp_avg_sq"].clone()

    g2 = torch.randn(32, 16)
    p.grad = g2
    opt.step()
    new_proj = opt.state[p]["proj"].clone()
    assert not torch.equal(new_proj, old_proj)

    a = (new_proj.float() @ old_proj.float().t()) / 4.0
    low = new_proj.float() @ g2
    expected_m = 0.9 * (a @ m1) + 0.1 * low
    expected_v = 0.999 * ((a * a) @ v1) + 0.001 * low * low

    assert torch.allclose(opt.state[p]["exp_avg"], expected_m, atol=1e-6, rtol=1e-5)
    assert torch.allclose(opt.state[p]["exp_avg_sq"], expected_v, atol=1e-6, rtol=1e-5)


def test_transport_moments_math_wide():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 32))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=2, transport_moments=True)

    p.grad = torch.randn(16, 32)
    opt.step()
    old_proj = opt.state[p]["proj"].clone()
    m1 = opt.state[p]["exp_avg"].clone()
    v1 = opt.state[p]["exp_avg_sq"].clone()

    g2 = torch.randn(16, 32)
    p.grad = g2
    opt.step()
    new_proj = opt.state[p]["proj"].clone()
    assert not torch.equal(new_proj, old_proj)

    a = (old_proj.float().t() @ new_proj.float()) / 4.0
    low = g2.float() @ new_proj.float()
    expected_m = 0.9 * (m1 @ a) + 0.1 * low
    expected_v = 0.999 * (v1 @ (a * a)) + 0.001 * low * low

    assert torch.allclose(opt.state[p]["exp_avg"], expected_m, atol=1e-6, rtol=1e-5)
    assert torch.allclose(opt.state[p]["exp_avg_sq"], expected_v, atol=1e-6, rtol=1e-5)


def test_transport_disabled_leaves_moments_in_old_basis():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Hakuli([p], lr=1e-3, rank=4, update_proj_gap=2, transport_moments=False)

    p.grad = torch.randn(32, 16)
    opt.step()
    m1 = opt.state[p]["exp_avg"].clone()
    v1 = opt.state[p]["exp_avg_sq"].clone()

    g2 = torch.randn(32, 16)
    p.grad = g2
    opt.step()
    new_proj = opt.state[p]["proj"].clone()
    low = new_proj.float() @ g2
    expected_m = 0.9 * m1 + 0.1 * low
    expected_v = 0.999 * v1 + 0.001 * low * low

    assert torch.allclose(opt.state[p]["exp_avg"], expected_m, atol=1e-6, rtol=1e-5)
    assert torch.allclose(opt.state[p]["exp_avg_sq"], expected_v, atol=1e-6, rtol=1e-5)


def test_transport_rejects_non_bool():
    p = torch.nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError):
        Hakuli([p], transport_moments="yes")


def test_foreach_state_matches_single_state():
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(16)) for _ in range(8)]
    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_a = Hakuli(params_a, lr=1e-2, foreach=False)
    opt_b = Hakuli(params_b, lr=1e-2, foreach=True)

    torch.manual_seed(7)
    for _ in range(4):
        for pa, pb in zip(params_a, params_b):
            g = torch.randn(16)
            pa.grad = g.clone()
            pb.grad = g.clone()
        opt_a.step()
        opt_b.step()

    for pa, pb in zip(params_a, params_b):
        sa, sb = opt_a.state[pa], opt_b.state[pb]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(sa[key], sb[key], atol=1e-7)
        assert float(sa["prev_norm"]) == pytest.approx(
            float(sb["prev_norm"]), rel=1e-6
        )
