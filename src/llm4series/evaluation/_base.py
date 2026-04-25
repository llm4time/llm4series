from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd
import numpy as np


class Metrics:
  def __init__(self, y_true: pd.DataFrame | pd.Series | list[float],
                     y_pred: pd.DataFrame | pd.Series | list[float]) -> None:
    """Initialize the Metrics calculator with true and predicted values.
    
    Creates a Metrics object for computing performance metrics (SMAPE, MAE, RMSE)
    between true and predicted values. Supports univariate and multivariate inputs.
    
    Args:
        y_true (pd.DataFrame | pd.Series | list[float]): True values.
        y_pred (pd.DataFrame | pd.Series | list[float]): Predicted values.
    
    Raises:
        TypeError: If input types are invalid or mismatched (e.g., DataFrame vs Series).
    """
    self._validate_inputs(y_true, y_pred)
    self.y_true = y_true
    self.y_pred = y_pred
    self.name = (
      y_true.name if isinstance(y_true, pd.Series) and y_true.name
      else y_pred.name if isinstance(y_pred, pd.Series) and y_pred.name
      else "value"
    )

  def _validate_inputs(self, y_true, y_pred) -> None:
    """Validate that input types are compatible.
    
    Checks that both inputs are of valid types and that DataFrame inputs
    are paired with DataFrame inputs (not mixed with Series or lists).
    
    Args:
        y_true: True values to validate.
        y_pred: Predicted values to validate.
    
    Raises:
        TypeError: If inputs are not DataFrame/Series/list or types are incompatible.
    """
    valid_types = (pd.DataFrame, pd.Series, list)
    if not (isinstance(y_true, valid_types) and isinstance(y_pred, valid_types)) or \
      (isinstance(y_true, pd.DataFrame) != isinstance(y_pred, pd.DataFrame)):
      raise TypeError(f"Invalid input types: {type(y_true).__name__} and {type(y_pred).__name__}.")

  def _drop_nan(self, y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN values from both arrays.
    
    Converts inputs to float arrays and removes any rows containing NaN values.
    
    Args:
        y_true: True values (any array-like).
        y_pred: Predicted values (any array-like).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays with NaN values removed (aligned).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]

  def _apply(self, func, decimals: int):
    """Apply a metric function to the data.
    
    Applies the given function across all columns (if DataFrame) or to the
    entire series/list. Handles NaN removal and rounding.
    
    Args:
        func: Metric function to apply (takes y_true, y_pred, decimals).
        decimals (int): Number of decimal places for rounding results.
    
    Returns:
        float | pd.Series: Computed metric(s).
    """
    if isinstance(self.y_true, pd.DataFrame):
      return pd.Series({
        col: func(*self._drop_nan(self.y_true[col], self.y_pred[col]), decimals)
        for col in self.y_true.columns
      })
    return func(*self._drop_nan(self.y_true, self.y_pred), decimals)

  def smape(self, decimals: int = 2) -> float | pd.Series:
    """Calculate Symmetric Mean Absolute Percentage Error.
    
    Computes SMAPE metric which measures the relative error between true and
    predicted values. Returns percentage values (0-200, where 0 is perfect).
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 2.
    
    Returns:
        float | pd.Series: SMAPE value(s) as percentage(s).
    """
    def _smape(y_true: np.ndarray, y_pred: np.ndarray, decimals: int) -> float:
      numerator = np.abs(y_true - y_pred)
      denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
      return round(float(np.mean(numerator / (denominator + 1e-10)) * 100), decimals)
    return self._apply(_smape, decimals)

  def mae(self, decimals: int = 2) -> float | pd.Series:
    """Calculate Mean Absolute Error.
    
    Computes MAE, the average absolute difference between true and predicted values.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 2.
    
    Returns:
        float | pd.Series: MAE value(s).
    """
    def _mae(y_true: np.ndarray, y_pred: np.ndarray, decimals: int) -> float:
      return round(float(mean_absolute_error(y_true, y_pred)), decimals)
    return self._apply(_mae, decimals)

  def rmse(self, decimals: int = 2) -> float | pd.Series:
    """Calculate Root Mean Squared Error.
    
    Computes RMSE, the square root of mean squared differences between true and
    predicted values. Emphasizes larger errors more than MAE.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 2.
    
    Returns:
        float | pd.Series: RMSE value(s).
    """
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray, decimals: int) -> float:
      return round(float(root_mean_squared_error(y_true, y_pred)), decimals)
    return self._apply(_rmse, decimals)

  def metrics(self, decimals: int = 2) -> pd.DataFrame:
    """Compute all metrics (SMAPE, MAE, RMSE) in a single DataFrame.
    
    Calculates SMAPE, MAE, and RMSE metrics and returns them organized in a
    DataFrame for easy comparison.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 2.
    
    Returns:
        pd.DataFrame: DataFrame with metrics as columns (and series name as index for univariate data).
    """
    data = {
      "smape": self.smape(decimals),
      "mae":   self.mae(decimals),
      "rmse":  self.rmse(decimals),
    }
    if isinstance(self.y_true, pd.DataFrame):
      return pd.DataFrame(data)
    return pd.DataFrame(data, index=[self.name]).T
