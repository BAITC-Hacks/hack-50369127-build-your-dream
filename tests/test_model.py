import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast.features import add_time_features
from wind_power_forecast.model import predict_power, train_power_model


def history(rows=120):
    wind = np.tile(np.linspace(0, 15, 24), (rows + 23) // 24)[:rows]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "wind_speed": wind,
            "temperature": 10.0,
            "normalized_power": np.clip((wind / 12) ** 3, 0, 1),
        }
    )


class PowerModelTest(unittest.TestCase):
    def test_shuffled_history_has_the_same_chronological_split_and_predictions(self):
        data = history()
        model, report = train_power_model(data)
        shuffled_model, shuffled_report = train_power_model(data.sample(frac=1, random_state=19))
        self.assertEqual(report.to_dict(), shuffled_report.to_dict())
        pd.testing.assert_series_equal(
            predict_power(model, data), predict_power(shuffled_model, data)
        )
        self.assertLess(
            pd.Timestamp(report.validation_training_ended_at),
            pd.Timestamp(report.validation_started_at),
        )
        self.assertLessEqual(report.validation_mae, report.baseline_mae)
        self.assertEqual(report.validation_kind, "chronological_observed_weather_holdout")

    def test_validation_excludes_tail_labels_but_final_refit_includes_them(self):
        data = history()
        data["wind_speed"] = 5.0
        data["normalized_power"] = 0.0
        data.loc[96:, "normalized_power"] = 1.0
        model, report = train_power_model(data)

        # Every selection model sees only zero labels. Perfect validation would
        # expose leakage; predicting zero after refit would omit recent history.
        self.assertAlmostEqual(report.validation_mae, 1.0)
        self.assertAlmostEqual(report.baseline_mae, 1.0)
        self.assertEqual(report.validation_training_rows, 96)
        self.assertEqual(report.training_rows, 120)
        self.assertTrue(report.refit_on_all_history)
        self.assertAlmostEqual(report.training_mean_power, 0.2)
        np.testing.assert_allclose(predict_power(model, data), 0.2)

    def test_duplicate_hours_are_rejected_before_splitting(self):
        data = history()
        with self.assertRaisesRegex(ValueError, "unique"):
            train_power_model(pd.concat([data, data.iloc[-1:]], ignore_index=True))

    def test_invalid_training_rows_are_counted_and_not_fitted(self):
        data = history()
        data.loc[0, "wind_speed"] = np.inf
        data.loc[1, "temperature"] = np.nan
        data.loc[2, "normalized_power"] = np.inf
        data.loc[3, "wind_speed"] = -1
        data.loc[4, "normalized_power"] = 1.1
        model, report = train_power_model(data)
        self.assertEqual(report.dropped_training_rows, 5)
        self.assertEqual(report.training_rows, 115)
        self.assertTrue(np.isfinite(predict_power(model, data.iloc[5:])).all())

    def test_nonfinite_prediction_inputs_and_outputs_fail_clearly(self):
        class InvalidModel:
            def predict(self, features):
                return np.repeat(np.nan, len(features))

        data = history(1)
        with self.assertRaisesRegex(ValueError, "finite normalized-power"):
            predict_power(InvalidModel(), data)
        data["wind_speed"] = np.inf
        with self.assertRaisesRegex(ValueError, "features must be finite"):
            predict_power(InvalidModel(), data)

    def test_prediction_preserves_row_order_and_index(self):
        data = history()
        model, _ = train_power_model(data)
        forecast = data.iloc[[5, 2, 10]]
        predicted = predict_power(model, forecast)
        self.assertEqual(predicted.index.tolist(), [5, 2, 10])
        np.testing.assert_allclose(predicted, predict_power(model, data).iloc[[5, 2, 10]])
        self.assertTrue(predicted.between(0, 1).all())

    def test_time_features_preserve_project_timezone(self):
        features = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2026-01-01T01:00:00+05:00")]}
        )
        result = add_time_features(features)
        self.assertAlmostEqual(result["hour_sin"].iloc[0], 0.2588190451)
        self.assertEqual(result["dayofyear_sin"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
