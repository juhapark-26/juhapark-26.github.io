"""Optional CPU end-to-end smoke test using the user's external egoPPG checkout.

Set BEAT_EGOPPG_ROOT to a local official egoPPG source checkout to enable this
test. It creates random inputs in memory and neither downloads nor saves weights.
"""

import os
import unittest
from unittest.mock import patch

import torch

from eccvw2.losses import (
    batch_global_waveform_mse,
    cardiac_phase_velocity_loss,
    official_filtfilt_matrix,
    physiological_band_spectral_js,
)


@unittest.skipUnless(os.environ.get("BEAT_EGOPPG_ROOT"), "Set BEAT_EGOPPG_ROOT for the external-backbone CPU smoke test")
class ExternalBackboneTests(unittest.TestCase):
    def test_paper_shape_and_one_training_step(self):
        from eccvw2.base import build_pulseformer

        torch.set_num_threads(2)
        torch.manual_seed(0)
        model_config = {
            "name": "pulseformer_bottleneck_dwtcn",
            "temporal_mixer": {
                "type": "dilated_depthwise_tcn",
                "insertion_point": "encoder_bottleneck",
                "apply_per_spatial_location": True,
                "channels": 64,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8],
                "normalization": "group_norm",
                "activation": "silu",
                "dropout": 0.1,
                "causal": False,
                "layer_scale": {"enabled": True, "init": 1e-3},
                "pointwise_glu": {"enabled": True, "expansion": 2},
                "residual_output": {"enabled": True, "zero_init": True},
            },
        }
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Network downloads are disabled in tests")):
            model = build_pulseformer(model_config, frames=128).cpu().train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        shapes = {}

        def record_shape(module, inputs, output):
            shapes["bottleneck"] = tuple(output.shape)

        hook = model.ConvBlock9.register_forward_hook(record_shape)
        video = torch.randn(1, 1, 128, 48, 128)
        imu = torch.randn(1, 128)
        time = torch.arange(128, dtype=torch.float32) / 30
        target = torch.sin(2 * torch.pi * 1.2 * time)[None]
        optimizer = torch.optim.Adam(model.parameters(), lr=9e-4)
        before = model.temporal_mixer.output_projection.weight.detach().clone()
        try:
            prediction = model(video, imu)
        finally:
            hook.remove()
        self.assertEqual(shapes["bottleneck"], (1, 64, 32, 3, 8))
        self.assertEqual(prediction.shape, (1, 128))
        self.assertTrue(torch.isfinite(prediction).all().item())
        matrix = official_filtfilt_matrix(128, sampling_rate_hz=30, band_hz=[0.7, 2.8], filter_order=4)
        waveform = batch_global_waveform_mse(prediction, target)
        spectral = physiological_band_spectral_js(prediction, target, sampling_rate_hz=30, band_hz=[0.7, 2.8], n_fft=128)
        mppl = cardiac_phase_velocity_loss(prediction, target, filter_matrix=matrix, edge_crop_samples=27, lags=[1, 2, 4])
        loss = waveform + 0.1 * spectral + 0.5 * mppl
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.temporal_mixer.output_projection.weight))


if __name__ == "__main__":
    unittest.main()
