import pandas as pd
from ._core import TimeSeries
from ._operations import _UniTimeSeriesImputation, _UniTimeSeriesStatistics, _UniTimeSeriesMetrics, _UniTimeSeriesPlot
from ._operations import _MultiTimeSeriesImputation, _MultiTimeSeriesStatistics, _MultiTimeSeriesMetrics, _MultiTimeSeriesPlot
from typing import Optional, Union, override, Literal


class UniTimeSeries(
    pd.Series,
    TimeSeries,
    _UniTimeSeriesPlot,
    _UniTimeSeriesImputation,
    _UniTimeSeriesStatistics,
    _UniTimeSeriesMetrics
):
  def __init__(self, data, *args, **kwargs):
    super().__init__(data, *args, **kwargs)

  @property
  def _constructor(self) -> 'UniTimeSeries':
    return UniTimeSeries

  @override
  def agg_duplicates(self, method: Literal['first', 'last', 'sum'], inplace: bool | None = False) -> Optional['UniTimeSeries']:
    ts = self if inplace else self.copy()
    match method:
      case "first":
        ts = ts[~ts.index.duplicated(keep="first")]
      case "last":
        ts = ts[~ts.index.duplicated(keep="last")]
      case "sum":
        ts = ts.groupby(ts.index).sum()
      case _:
        raise ValueError(f"Invalid method: {method}. Choose from 'first', 'last', 'sum'.")
    if inplace:
      self.__dict__.update(ts.__dict__)
    else:
      return ts

  @override
  def mean(self, decimals: int = 4, **kwargs) -> float:
    return round(super().mean(**kwargs), decimals)

  @override
  def median(self, decimals: int = 4, **kwargs) -> float:
    return round(super().median(**kwargs), decimals)

  @override
  def std(self, decimals: int = 4, **kwargs) -> float:
    return round(super().std(**kwargs), decimals)

  @override
  def min(self, decimals: int = 4, **kwargs) -> float:
    return round(super().min(**kwargs), decimals)

  @override
  def max(self, decimals: int = 4, **kwargs) -> float:
    return round(super().max(**kwargs), decimals)

  @override
  def quantile(self, q: float, decimals: int = 4, **kwargs) -> float:
    return round(super().quantile(q, **kwargs), decimals)

  @override
  def plot(self, kind: Literal["line", "chart", "bar", "stl", "autocorr"], title: str = None, xlabel: str = None, ylabel: str = None, **kwargs):
    match kind:
      case "line":
        return self.lineplot(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
      case "chart":
        return self.linechart(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
      case "bar":
        return self.barplot(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
      case "stl":
        return self.stlplot(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
      case "autocorr":
        return self.autocorrplot(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
      case _:
        raise ValueError(f"Invalid kind: {kind}. Choose from 'line', 'chart', 'bar', 'stl', 'autocorr'.")

  def trend(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> Union['UniTimeSeries', float]:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["t_strength"]
    return res["trend"]

  def seasonal(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> Union['UniTimeSeries', float]:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["s_strength"]
    return res["seasonal"]

  def residual(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> Union['UniTimeSeries', float]:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["r_strength"]
    return res["residual"]


class MultiTimeSeries(
    pd.DataFrame,
    TimeSeries,
    _MultiTimeSeriesPlot,
    _MultiTimeSeriesImputation,
    _MultiTimeSeriesStatistics,
    _MultiTimeSeriesMetrics
):
  def __init__(self, data, *args, **kwargs):
    super().__init__(data, *args, **kwargs)

  def __getitem__(self, key):
    obj = super().__getitem__(key)
    if isinstance(obj, pd.Series) and not isinstance(obj, UniTimeSeries):
      return UniTimeSeries(obj)
    elif isinstance(obj, pd.DataFrame) and not isinstance(obj, MultiTimeSeries):
      return MultiTimeSeries(obj)
    return obj

  @property
  def _constructor(self) -> 'MultiTimeSeries':
    return MultiTimeSeries

  @property
  def _constructor_sliced(self) -> UniTimeSeries:
    return UniTimeSeries

  @property
  def num_columns(self):
    return self.select_dtypes(include='number').columns

  @property
  def cat_columns(self):
    return self.select_dtypes(exclude='number').columns

  @override
  def agg_duplicates(self, method: Literal['first', 'last', 'sumf', 'suml'], inplace: bool | None = False) -> Optional['MultiTimeSeries']:
    ts = self if inplace else self.copy()
    match method:
      case "first":
        ts = ts[~ts.index.duplicated(keep="first")]
      case "last":
        ts = ts[~ts.index.duplicated(keep="last")]
      case "sumf":
        ts = pd.concat([
            ts[self.num_columns].groupby(ts.index).sum(),
            ts[self.cat_columns].groupby(ts.index).first()
        ], axis=1)
      case "suml":
        ts = pd.concat([
            ts[self.num_columns].groupby(ts.index).sum(),
            ts[self.cat_columns].groupby(ts.index).last()
        ], axis=1)
      case _:
        raise ValueError(f"Invalid method: {method}. Choose from 'first', 'last', 'sumf', 'suml'.")
    if inplace:
      self.__dict__.update(ts.__dict__)
    else:
      return ts

  @override
  def mean(self, decimals: int = 4, **kwargs) -> pd.Series:
    return round(super().mean(**kwargs), decimals)

  @override
  def median(self, decimals: int = 4, **kwargs) -> pd.Series:
    return round(super().median(**kwargs), decimals)

  @override
  def std(self, decimals: int = 4, **kwargs) -> pd.Series:
    return round(super().std(**kwargs), decimals)

  @override
  def min(self, decimals: int = 4, **kwargs) -> pd.Series:
    return round(super().min(**kwargs), decimals)

  @override
  def max(self, decimals: int = 4, **kwargs) -> pd.Series:
    return round(super().max(**kwargs), decimals)

  @override
  def quantile(self, q: float, decimals: int = 4, **kwargs) -> pd.Series:
    return round(super().quantile(q, **kwargs), decimals)

  @override
  def plot(self, kind: Literal["line", "chart", "bar", "stl"], title: str = None, xlabel: str = None, ylabel: str = None, lightness: float = 0.7, **kwargs):
    match kind:
      case "line":
        return self.lineplot(title=title, xlabel=xlabel, ylabel=ylabel, lightness=lightness, **kwargs)
      case "chart":
        return self.linechart(title=title, xlabel=xlabel, ylabel=ylabel, lightness=lightness, **kwargs)
      case "bar":
        return self.barplot(title=title, xlabel=xlabel, ylabel=ylabel, lightness=lightness, **kwargs)
      case "stl":
        return self.stlplot(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
      case _:
        raise ValueError(f"Invalid kind: {kind}. Choose from 'line', 'chart', 'bar', 'stl'.")

  def trend(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["t_strength"]
    return res["trend"]

  def seasonal(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["s_strength"]
    return res["seasonal"]

  def residual(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["r_strength"]
    return res["residual"]
