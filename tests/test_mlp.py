"""Check independently sized MLPs and their training behavior."""

import unittest

import torch
from torch import nn

from model import MLP


class MLPTests(unittest.TestCase):
    def test_different_networks_train_independently(self):
        networks = nn.ModuleDict({
            "small": MLP(8, 3, hidden_dims=[4]),
            "deep": MLP(8, 2, hidden_dims=[12, 6, 4]),
            "wide": MLP(8, 5, hidden_dims=[16, 10]),
            "linear": MLP(8, 1),
        })
        inputs = torch.randn(2, 3, 8, requires_grad=True)
        loss = 0
        for network in networks.values():
            outputs = network(inputs)
            self.assertEqual(outputs.shape, (2, 3, network.output_dim))
            loss = loss + outputs.square().mean()
        loss.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(all(p.grad is not None for p in networks.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in networks.parameters()))
        old_weights = networks["deep"].layers[0].weight.detach().clone()
        with torch.no_grad():
            networks["small"].layers[0].weight.zero_()
        torch.testing.assert_close(networks["deep"].layers[0].weight, old_weights)

    def test_hidden_activation_and_unrestricted_output(self):
        network = MLP(2, 1, hidden_dims=[2])
        with torch.no_grad():
            network.layers[0].weight.copy_(torch.eye(2))
            network.layers[0].bias.zero_()
            network.layers[-1].weight.fill_(-1)
            network.layers[-1].bias.zero_()
        torch.testing.assert_close(network(torch.tensor([-3., 2.])), torch.tensor([-2.]))

    def test_linear_network_and_state_dict_round_trip(self):
        network = MLP(2, 1)
        with torch.no_grad():
            network.layers[0].weight.copy_(torch.tensor([[2., -1.]]))
            network.layers[0].bias.fill_(1)
        clone = MLP(2, 1)
        clone.load_state_dict(network.state_dict())
        inputs = torch.tensor([[1., 4.], [2., -1.]])
        expected = torch.tensor([[-1.], [6.]])
        torch.testing.assert_close(network(inputs), expected)
        torch.testing.assert_close(clone(inputs), expected)

    def test_invalid_dimensions_are_rejected(self):
        for dimensions in ((0, 1, []), (2, -1, []), (2, 1, [0]),
                           (2, 1, [True]), (2, 1, [1.5]), (False, 1, [])):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                MLP(*dimensions)

    def test_invalid_input_shape_is_rejected(self):
        network = MLP(4, 2, hidden_dims=[3])
        for inputs in (torch.tensor(1.), torch.ones(2, 5)):
            with self.subTest(shape=inputs.shape), self.assertRaisesRegex(ValueError, "shape"):
                network(inputs)


if __name__ == "__main__":
    unittest.main()
