"""Shape, initialization, and optimization checks for the public adapter."""

import unittest

import torch

from eccvw2.temporal import DilatedDepthwiseTCN


class TemporalAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_paper_parameter_count_and_receptive_field(self):
        adapter = DilatedDepthwiseTCN(64)
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 18560)
        self.assertEqual(adapter.receptive_field, 31)
        self.assertEqual(adapter.dilations, (1, 2, 4, 8))
        self.assertFalse(adapter.causal)
        for block, dilation in zip(adapter.blocks, (1, 2, 4, 8)):
            self.assertEqual(block.depthwise_conv.groups, 64)
            self.assertEqual(block.depthwise_conv.padding_mode, "zeros")
            self.assertEqual(block.depthwise_conv.padding, (dilation,))

    def test_zero_projection_initializes_an_exact_identity(self):
        adapter = DilatedDepthwiseTCN(64)
        # Two examples, each with 3 x 8 independent spatial trajectories.
        signal = torch.randn(2 * 3 * 8, 64, 32)
        output = adapter(signal)
        self.assertEqual(output.shape, signal.shape)
        torch.testing.assert_close(output, signal, atol=0, rtol=0)

    def test_adapter_can_take_an_optimizer_step(self):
        adapter = DilatedDepthwiseTCN(64)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=9e-4)
        signal = torch.randn(24, 64, 32)
        target = torch.randn_like(signal)
        before = adapter.output_projection.weight.detach().clone()
        loss = (adapter(signal) - target).square().mean()
        loss.backward()
        for name, parameter in adapter.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, adapter.output_projection.weight))


if __name__ == "__main__":
    unittest.main()
