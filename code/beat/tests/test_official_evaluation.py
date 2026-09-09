"""Optional official post-processing checks on generated pulse signals only."""

import os
from pathlib import Path
import unittest

import numpy as np

from eccvw2.evaluation import _official_metric_function, evaluate_waveform_store


UPSTREAM_ROOT = os.environ.get("BEAT_EGOPPG_ROOT", "")
HAS_EVALUATOR = bool(UPSTREAM_ROOT) and (
    Path(UPSTREAM_ROOT) / "evaluation" / "post_process.py"
).is_file()


@unittest.skipUnless(HAS_EVALUATOR, "Set BEAT_EGOPPG_ROOT to a checkout containing the official evaluator")
class OfficialEvaluationTests(unittest.TestCase):
    @staticmethod
    def differential(pulse):
        return np.concatenate(([0.0], pulse[:-1] - pulse[1:]))

    def test_generated_pulse_has_expected_peak_interval_hr(self):
        metric = _official_metric_function(UPSTREAM_ROOT)
        time = np.arange(1800) / 30
        for frequency in (1.2, 1.5):
            with self.subTest(frequency=frequency):
                signal = self.differential(np.sin(2 * np.pi * frequency * time))
                target_hr, predicted_hr, _, _, _ = metric(
                    signal, signal, "Peak_Detection", True, 30
                )
                self.assertAlmostEqual(target_hr, frequency * 60, delta=1.0)
                self.assertEqual(predicted_hr, target_hr)

    def test_perfect_clip_store_uses_complete_60_second_windows(self):
        time = np.arange(1800) / 30
        pulse = np.concatenate((
            np.sin(2 * np.pi * 1.2 * time),
            np.sin(2 * np.pi * 1.5 * time),
            np.sin(2 * np.pi * 1.0 * np.arange(128) / 30),
        ))
        signal = self.differential(pulse)
        # Reverse insertion order to exercise sorting by original clip index.
        chunks = {
            index: signal[start:start + 128].copy()
            for index, start in reversed(list(enumerate(range(0, len(signal), 128))))
        }
        store = {
            "final": {"synthetic": chunks},
            "target": {"synthetic": {index: value.copy() for index, value in chunks.items()}},
        }
        result = evaluate_waveform_store(
            store, UPSTREAM_ROOT, fs=30, window_seconds=60, hr_method="Peak_Detection"
        )["final"]
        self.assertEqual(result["number_of_hr_windows"], 2)
        self.assertEqual(result["number_of_participants"], 1)
        for key in ("hr_mae_bpm", "hr_rmse_bpm", "hr_mape_percent", "waveform_mae_z", "waveform_rmse_z"):
            self.assertEqual(result[key], 0.0, key)
        self.assertAlmostEqual(result["hr_pearson"], 1.0)
        self.assertAlmostEqual(result["waveform_pearson"], 1.0)


if __name__ == "__main__":
    unittest.main()
