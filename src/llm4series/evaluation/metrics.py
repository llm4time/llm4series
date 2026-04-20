from ._base import Metrics
from ..data import TimeSeries
from scipy.stats import sem as scipy_sem
import pandas as pd


def smape(y_true: TimeSeries | list[float], y_pred: TimeSeries | list[float], decimals: int = 2) -> float:
  return Metrics(y_true, y_pred).smape(decimals=decimals)

def mae(y_true: TimeSeries | list[float], y_pred: TimeSeries | list[float], decimals: int = 2) -> float:
  return Metrics(y_true, y_pred).mae(decimals=decimals)

def rmse(y_true: TimeSeries | list[float], y_pred: TimeSeries | list[float], decimals: int = 2) -> float:
  return Metrics(y_true, y_pred).rmse(decimals=decimals)

def sem(errors: list[float], decimals: int = 4) -> float:
  return round(scipy_sem(errors), decimals)

def metrics(y_true: TimeSeries | list[float], y_pred: TimeSeries | list[float], decimals: int = 2) -> pd.DataFrame:
  return Metrics(y_true, y_pred).metrics(decimals)
