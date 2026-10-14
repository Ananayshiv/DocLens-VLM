import pytest
import torch
from doclens.late_interaction import maxsim, score_pages, contrastive_loss


def test_mathematical_maxsim():
    q = torch.tensor([[1., 0.], [0., 1.]])
    good = torch.tensor([[1., 0.], [0., 1.]])
    bad = torch.tensor([[-1., 0.], [0., -1.]])
    assert maxsim(q, good).item() == pytest.approx(2.)
    assert maxsim(q, bad).item() == pytest.approx(0.)
    assert score_pages(q, [bad, good]).argmax().item() == 1


def test_mask_excludes_padding():
    q = torch.tensor([[1., 0.], [0., 1.]])
    d = torch.tensor([[1., 0.], [0., 1.]])
    assert maxsim(q, d, query_mask=torch.tensor([True, False])).item() == pytest.approx(1.)
    assert maxsim(q, d, document_mask=torch.tensor([True, False])).item() == pytest.approx(1.)


def test_unnormalized_vectors():
    q = torch.tensor([[3., 0.]])
    d = torch.tensor([[5., 0.]])
    assert maxsim(q, d).item() == pytest.approx(1.)
    assert maxsim(q, d, normalize=False).item() == pytest.approx(15.)


def test_requires_nonempty():
    with pytest.raises(ValueError): maxsim(torch.empty(0, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError): maxsim(torch.randn(2, 3), torch.randn(2, 5))
    with pytest.raises(ValueError): maxsim(torch.ones(1, 2), torch.ones(1, 2), document_mask=torch.tensor([False]))


def test_contrastive_loss_gradients():
    q = [torch.tensor([[1., 0.]], requires_grad=True), torch.tensor([[0., 1.]], requires_grad=True)]
    d = [torch.tensor([[1., 0.]], requires_grad=True), torch.tensor([[0., 1.]], requires_grad=True)]
    loss = contrastive_loss(q, d, temperature=.2)
    assert 0 < loss.item() < 1
    loss.backward()
    assert q[0].grad is not None
    with pytest.raises(ValueError): contrastive_loss([q[0]], [d[0]])


def test_adapter_optimizes_and_preserves_shape():
    from doclens.adapter import LowRankTokenAdapter, inbatch_adapter_step
    adapt = LowRankTokenAdapter(4, rank=2)
    opt = torch.optim.Adam(adapt.parameters(), lr=.02)
    q = [torch.tensor([[1., 0., 0., 0.]]), torch.tensor([[0., 1., 0., 0.]])]
    pages = [x.clone() for x in q]
    assert adapt(q[0]).shape == (1, 4)
    initial = adapt.up.weight.detach().clone()
    loss = inbatch_adapter_step(adapt, opt, q, pages)
    assert loss >= 0
    assert not torch.equal(initial, adapt.up.weight.detach())
