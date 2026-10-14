import pytest
import torch
import torch.nn.functional as F
from doclens.late_interaction import maxsim


def reference(q, d, qm, dm):
    dense = F.normalize(q, dim=-1) @ F.normalize(d, dim=-1).T
    dense = dense.masked_fill(~dm[None, :], -torch.inf)
    return dense.max(dim=-1).values[qm].sum()


@pytest.mark.parametrize("chunk", [1, 3, 16, 1000])
def test_chunked_equal_to_dense_with_masks(chunk):
    torch.manual_seed(44)
    q = torch.randn(17, 9)
    d = torch.randn(101, 9)
    qm = torch.rand(17) > .25
    dm = torch.rand(101) > .25
    actual = maxsim(q, d, qm, dm, chunk_size=chunk)
    expected = reference(q, d, qm, dm)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_gradient_agrees_with_dense():
    torch.manual_seed(3)
    q = torch.randn(6, 5, requires_grad=True)
    d = torch.randn(9, 5, requires_grad=True)
    actual = maxsim(q, d, chunk_size=3)
    grad = torch.autograd.grad(actual, [q, d])
    expected = reference(q, d, torch.ones(6, dtype=torch.bool),
                         torch.ones(9, dtype=torch.bool))
    grad_ref = torch.autograd.grad(expected, [q, d])
    for a, b in zip(grad, grad_ref):
        assert torch.allclose(a, b, atol=1e-5)


def test_large_document_smoke_and_malformed_inputs():
    q = torch.randn(32, 16)
    d = torch.randn(65536, 16)
    result = maxsim(q, d, chunk_size=128)
    assert result.isfinite()
    with pytest.raises(ValueError, match="chunk_size"):
        maxsim(q, d[:1], chunk_size=0)
    with pytest.raises(ValueError, match="finite"):
        maxsim(q, torch.tensor([[float("nan")] + [0.] * 15]))
