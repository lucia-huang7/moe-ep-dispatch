import torch
from moe_lab.reference import dense_reference


def test_dropped_routes_and_nonuniform_weights():
    x = torch.tensor([[2., 3.]])
    ids = torch.tensor([[1, -1, 0]])
    weights = torch.tensor([[.25, 99., .75]])
    matrices = torch.stack([torch.eye(2), 2*torch.eye(2)])
    biases = torch.tensor([[1., 2.], [3., 4.]])
    expected = .25*(2*x.double()+biases[1]) + .75*(x.double()+biases[0])
    torch.testing.assert_close(dense_reference(x, ids, weights, matrices, biases), expected)
