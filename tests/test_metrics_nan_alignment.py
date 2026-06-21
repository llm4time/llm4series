import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from llm4series import UniTimeSeries, MultiTimeSeries


def test_metrics_drop_nan_in_y_pred():
  y_true = UniTimeSeries([1.0, 2.0, 3.0, 4.0], name="v")
  y_pred = [1.1, np.nan, 2.9, 4.2]
  # NaN appears only in y_pred; the matching y_true position must be dropped too.
  expected_true, expected_pred = [1.0, 3.0, 4.0], [1.1, 2.9, 4.2]
  assert y_true.mae(y_pred) == round(mean_absolute_error(expected_true, expected_pred), 2)
  assert y_true.rmse(y_pred) == round(root_mean_squared_error(expected_true, expected_pred), 2)
  assert isinstance(y_true.smape(y_pred), float)


def test_metrics_drop_nan_in_y_true():
  y_true = UniTimeSeries([1.0, np.nan, 3.0, 4.0], name="v")
  y_pred = [1.1, 2.0, 2.9, 4.2]
  expected_true, expected_pred = [1.0, 3.0, 4.0], [1.1, 2.9, 4.2]
  assert y_true.mae(y_pred) == round(mean_absolute_error(expected_true, expected_pred), 2)
  assert y_true.rmse(y_pred) == round(root_mean_squared_error(expected_true, expected_pred), 2)


def test_metrics_match_sklearn_without_nan():
  y_true = UniTimeSeries([1.0, 2.0, 3.0, 4.0], name="v")
  y_pred = [1.1, 1.9, 3.2, 3.8]
  assert y_true.mae(y_pred) == round(mean_absolute_error([1.0, 2.0, 3.0, 4.0], y_pred), 2)
  assert y_true.rmse(y_pred) == round(root_mean_squared_error([1.0, 2.0, 3.0, 4.0], y_pred), 2)


def test_metrics_length_mismatch_raises():
  y_true = UniTimeSeries([1.0, 2.0, 3.0, 4.0], name="v")
  with pytest.raises(ValueError):
    y_true.mae([1.0, 2.0])


def test_multitimeseries_metrics_handle_nan():
  df = MultiTimeSeries(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]}))
  pred = pd.DataFrame({"a": [1.1, np.nan, 2.9, 4.2], "b": [5.0, 6.1, 7.2, 7.9]})
  result = df.mae(pred)
  assert set(result.index) == {"a", "b"}
  assert not result.isna().any()
