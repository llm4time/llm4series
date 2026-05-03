from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd
import numpy as np


class Metrics:
  def __init__(self, y_true: pd.DataFrame | pd.Series | list[float],
                     y_pred: pd.DataFrame | pd.Series | list[float]) -> None:
    self._validate_inputs(y_true, y_pred)
    self.y_true = y_true
    self.y_pred = y_pred
    self.name = (
      y_true.name if isinstance(y_true, pd.Series) and y_true.name
      else y_pred.name if isinstance(y_pred, pd.Series) and y_pred.name
      else "value"
    )

  def _validate_inputs(self, y_true, y_pred) -> None:
    valid_types = (pd.DataFrame, pd.Series, list)
    if not (isinstance(y_true, valid_types) and isinstance(y_pred, valid_types)) or \
      (isinstance(y_true, pd.DataFrame) != isinstance(y_pred, pd.DataFrame)):
      raise TypeError(f"Invalid input types: {type(y_true).__name__} and {type(y_pred).__name__}.")

  def _drop_nan(self, y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]

  def _apply(self, func, decimals: int):
    if isinstance(self.y_true, pd.DataFrame):
      return pd.Series({
        col: func(*self._drop_nan(self.y_true[col], self.y_pred[col]), decimals)
        for col in self.y_true.columns
      })
    return func(*self._drop_nan(self.y_true, self.y_pred), decimals)

  def smape(self, decimals: int = 2) -> float | pd.Series:
    def _smape(y_true: np.ndarray, y_pred: np.ndarray, decimals: int) -> float:
      numerator = np.abs(y_true - y_pred)
      denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
      return round(float(np.mean(numerator / (denominator + 1e-10)) * 100), decimals)
    return self._apply(_smape, decimals)

  def mae(self, decimals: int = 2) -> float | pd.Series:
    def _mae(y_true: np.ndarray, y_pred: np.ndarray, decimals: int) -> float:
      return round(float(mean_absolute_error(y_true, y_pred)), decimals)
    return self._apply(_mae, decimals)

  def rmse(self, decimals: int = 2) -> float | pd.Series:
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray, decimals: int) -> float:
      return round(float(root_mean_squared_error(y_true, y_pred)), decimals)
    return self._apply(_rmse, decimals)

  def metrics(self, decimals: int = 2) -> pd.DataFrame:
    data = {
      "smape": self.smape(decimals),
      "mae":   self.mae(decimals),
      "rmse":  self.rmse(decimals),
    }
    if isinstance(self.y_true, pd.DataFrame):
      return pd.DataFrame(data)
    return pd.DataFrame(data, index=[self.name]).T
