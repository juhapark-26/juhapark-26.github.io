"""CPU-only numerical contracts for the paper waveform, spectral, and MPPL losses."""

import unittest

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert

from eccvw2.losses import (
    _fft_analytic_signal,
    batch_global_waveform_mse,
    cardiac_phase_velocity_loss,
    official_filtfilt_matrix,
    physiological_band_spectral_js,
)


class PaperLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.matrix = official_filtfilt_matrix(
            128, sampling_rate_hz=30, band_hz=[0.7, 2.8], filter_order=4
        )
        time = torch.arange(128, dtype=torch.float64) / 30
        pulse = torch.sin(2 * torch.pi * 1.2 * time)
        cls.target = torch.cat((torch.zeros(1), pulse[:-1] - pulse[1:]))[None]

    def mppl(self, prediction, target=None):
        return cardiac_phase_velocity_loss(
            prediction,
            self.target if target is None else target,
            filter_matrix=self.matrix,
            edge_crop_samples=27,
            lags=[1, 2, 4],
        )

    def test_waveform_uses_batch_global_sample_standard_deviation(self):
        target = torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.]], dtype=torch.float64)
        prediction = target * torch.tensor([[1.], [3.]], dtype=torch.float64)
        normalize = lambda x: (x - x.mean()) / x.std().clamp_min(1e-8)
        expected = (normalize(prediction) - normalize(target)).square().mean()
        actual = batch_global_waveform_mse(prediction, target)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual.item(), 0.01)
        self.assertEqual(batch_global_waveform_mse(target, target).item(), 0.0)

    def test_spectral_discrete_band_is_bins_3_through_11(self):
        frequency = torch.fft.rfftfreq(128, d=1 / 30)
        mask = (frequency >= 0.7) & (frequency <= 2.8)
        self.assertEqual(torch.nonzero(mask).flatten().tolist(), list(range(3, 12)))
        self.assertEqual(frequency[mask][0].item(), 0.703125)
        self.assertEqual(frequency[mask][-1].item(), 2.578125)

    def test_spectral_identical_signals_have_zero_loss(self):
        loss = physiological_band_spectral_js(
            self.target, self.target,
            sampling_rate_hz=30, band_hz=[0.7, 2.8], n_fft=128,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=12)

    def test_filtfilt_matrix_matches_scipy(self):
        generator = torch.Generator().manual_seed(42)
        signal = torch.randn(3, 128, dtype=torch.float64, generator=generator)
        b, a = butter(4, [0.7 / 15, 2.8 / 15], btype="bandpass")
        expected = filtfilt(b, a, signal.numpy(), axis=-1)
        np.testing.assert_allclose((signal @ self.matrix).numpy(), expected, atol=2e-8, rtol=2e-8)

    def test_hilbert_matches_scipy_without_extra_padding(self):
        generator = torch.Generator().manual_seed(42)
        for length in (127, 128):
            with self.subTest(length=length):
                signal = torch.randn(2, length, dtype=torch.float64, generator=generator)
                expected = hilbert(signal.numpy(), axis=-1)
                np.testing.assert_allclose(_fft_analytic_signal(signal).numpy(), expected, atol=1e-12)

    def test_mppl_perfect_prediction_is_zero(self):
        prediction = self.target.clone().requires_grad_()
        loss = self.mppl(prediction)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(prediction.grad).all().item())
        self.assertEqual(prediction.grad.norm().item(), 0.0)

    def test_mppl_constant_and_near_constant_predictions_are_finite(self):
        generator = torch.Generator().manual_seed(42)
        for scale in (0.0, 1e-8, 1e-5):
            with self.subTest(scale=scale):
                prediction = (torch.randn(self.target.shape, dtype=torch.float64, generator=generator) * scale).requires_grad_()
                loss = self.mppl(prediction)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(torch.isfinite(prediction.grad).all().item())
                self.assertLess(prediction.grad.norm().item(), 1e3)

    def test_mppl_zero_prediction_and_target_are_finite(self):
        prediction = torch.zeros_like(self.target, requires_grad=True)
        loss = self.mppl(prediction, torch.zeros_like(self.target))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_mppl_different_pulse_frequency_increases_error(self):
        time = torch.arange(128, dtype=torch.float64) / 30
        pulse = torch.sin(2 * torch.pi * 2.0 * time)
        prediction = torch.cat((torch.zeros(1), pulse[:-1] - pulse[1:]))[None]
        self.assertGreater(self.mppl(prediction).item(), self.mppl(self.target).item() + 1e-3)


if __name__ == "__main__":
    unittest.main()
