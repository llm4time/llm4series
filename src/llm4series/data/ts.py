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
    """Initialize a univariate time series.
    
    Creates a new UniTimeSeries by extending pd.Series with time series
    specific operations for imputation, statistics, metrics, and visualization.
    
    Args:
        data: Series data (array-like, dict, or scalar).
        *args: Additional positional arguments passed to pd.Series.
        **kwargs: Additional keyword arguments passed to pd.Series.
    """
    super().__init__(data, *args, **kwargs)

  @property
  def _constructor(self) -> 'UniTimeSeries':
    """Return the constructor for creating new UniTimeSeries objects.
    
    Used internally by pandas for constructing new instances from operations.
    
    Returns:
        type: UniTimeSeries class.
    """
    return UniTimeSeries

  @override
  def agg_duplicates(self, method: Literal['first', 'last', 'sum'], inplace: bool | None = False) -> Optional['UniTimeSeries']:
    """Aggregate duplicate index values using the specified method.
    
    Handles duplicate timestamps by keeping only one value per index using
    first occurrence, last occurrence, or sum aggregation.
    
    Args:
        method (Literal['first', 'last', 'sum']): Aggregation method for duplicates.
            - 'first': Keep first occurrence
            - 'last': Keep last occurrence
            - 'sum': Sum all values with duplicate index
        inplace (bool | None, optional): If True, modify the original series. Defaults to False.
    
    Returns:
        Optional[UniTimeSeries]: Deduplicated series if inplace=False, None if inplace=True.
    
    Raises:
        ValueError: If method is not 'first', 'last', or 'sum'.
    """
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
    """Initialize a multivariate time series.
    
    Creates a new MultiTimeSeries by extending pd.DataFrame with time series
    specific operations for imputation, statistics, metrics, and visualization.
    
    Args:
        data: DataFrame data (dict, array, or DataFrame).
        *args: Additional positional arguments passed to pd.DataFrame.
        **kwargs: Additional keyword arguments passed to pd.DataFrame.
    """
    super().__init__(data, *args, **kwargs)

  def __getitem__(self, key):
    """Get items from the time series, ensuring proper type returns.
    
    Overrides pandas __getitem__ to return UniTimeSeries for single columns
    and MultiTimeSeries for multiple columns.
    
    Args:
        key: Column label(s) or boolean mask.
    
    Returns:
        UniTimeSeries | MultiTimeSeries: Single column returns UniTimeSeries,
                                         multiple columns return MultiTimeSeries.
    """
    obj = super().__getitem__(key)
    if isinstance(obj, pd.Series) and not isinstance(obj, UniTimeSeries):
      return UniTimeSeries(obj)
    elif isinstance(obj, pd.DataFrame) and not isinstance(obj, MultiTimeSeries):
      return MultiTimeSeries(obj)
    return obj

  @property
  def _constructor(self) -> 'MultiTimeSeries':
    """Return the constructor for creating new MultiTimeSeries objects.
    
    Used internally by pandas for constructing new instances from operations.
    
    Returns:
        type: MultiTimeSeries class.
    """
    return MultiTimeSeries

  @property
  def _constructor_sliced(self) -> UniTimeSeries:
    """Return the constructor for creating UniTimeSeries from column selection.
    
    Used internally by pandas when selecting a single column from this DataFrame.
    
    Returns:
        type: UniTimeSeries class.
    """
    return UniTimeSeries

  @property
  def num_columns(self):
    """Get all numerical columns.
    
    Returns:
        Index: Column names with numeric data types.
    """
    return self.select_dtypes(include='number').columns

  @property
  def cat_columns(self):
    """Get all categorical (non-numerical) columns.
    
    Returns:
        Index: Column names with non-numeric data types.
    """
    return self.select_dtypes(exclude='number').columns

  @override
  def agg_duplicates(self, method: Literal['first', 'last', 'sumf', 'suml'], inplace: bool | None = False) -> Optional['MultiTimeSeries']:
    """Aggregate duplicate index values with separate handling for numeric and categorical columns.
    
    Handles duplicate timestamps by aggregating values based on the specified method.
    For numeric columns, supports sum aggregation. For categorical columns, keeps
    either first or last occurrence.
    
    Args:
        method (Literal['first', 'last', 'sumf', 'suml']): Aggregation method for duplicates.
            - 'first': Keep first occurrence
            - 'last': Keep last occurrence
            - 'sumf': Sum numeric columns, keep first categorical
            - 'suml': Sum numeric columns, keep last categorical
        inplace (bool | None, optional): If True, modify the original series. Defaults to False.
    
    Returns:
        Optional[MultiTimeSeries]: Deduplicated series if inplace=False, None if inplace=True.
    
    Raises:
        ValueError: If method is not one of the supported types.
    """
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
    """Calculate the mean of each numerical column.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
        **kwargs: Additional arguments passed to pandas mean().
    
    Returns:
        pd.Series: Rounded mean values for each column.
    """
    return round(super().mean(**kwargs), decimals)

  @override
  def median(self, decimals: int = 4, **kwargs) -> pd.Series:
    """Calculate the median of each numerical column.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
        **kwargs: Additional arguments passed to pandas median().
    
    Returns:
        pd.Series: Rounded median values for each column.
    """
    return round(super().median(**kwargs), decimals)

  @override
  def std(self, decimals: int = 4, **kwargs) -> pd.Series:
    """Calculate the standard deviation of each numerical column.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
        **kwargs: Additional arguments passed to pandas std().
    
    Returns:
        pd.Series: Rounded standard deviation values for each column.
    """
    return round(super().std(**kwargs), decimals)

  @override
  def min(self, decimals: int = 4, **kwargs) -> pd.Series:
    """Calculate the minimum value of each numerical column.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
        **kwargs: Additional arguments passed to pandas min().
    
    Returns:
        pd.Series: Rounded minimum values for each column.
    """
    return round(super().min(**kwargs), decimals)

  @override
  def max(self, decimals: int = 4, **kwargs) -> pd.Series:
    """Calculate the maximum value of each numerical column.
    
    Args:
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
        **kwargs: Additional arguments passed to pandas max().
    
    Returns:
        pd.Series: Rounded maximum values for each column.
    """
    return round(super().max(**kwargs), decimals)

  @override
  def quantile(self, q: float, decimals: int = 4, **kwargs) -> pd.Series:
    """Calculate a specific quantile for each numerical column.
    
    Args:
        q (float): Quantile value between 0 and 1 (e.g., 0.25 for 25th percentile).
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
        **kwargs: Additional arguments passed to pandas quantile().
    
    Returns:
        pd.Series: Rounded quantile values for each column.
    """
    return round(super().quantile(q, **kwargs), decimals)

  @override
  def plot(self, kind: Literal["line", "chart", "bar", "stl"], title: str = None, xlabel: str = None, ylabel: str = None, lightness: float = 0.7, **kwargs):
    """Create various types of plots for the multivariate time series.
    
    Generates interactive Plotly visualizations with separate traces for each column.
    
    Args:
        kind (Literal["line", "chart", "bar", "stl"]): Type of plot to generate.
            - 'line': Line plot with sequential index
            - 'chart': Line chart with datetime index
            - 'bar': Grouped bar plot of summary statistics
            - 'stl': STL decomposition subplots for all columns
        title (str, optional): Plot title. Defaults to None.
        xlabel (str, optional): X-axis label. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
        lightness (float, optional): Lightness adjustment for colors (0-1). Defaults to 0.7.
        **kwargs: Additional arguments passed to the plot function.
    
    Returns:
        go.Figure: Plotly Figure object.
    
    Raises:
        ValueError: If kind is not one of the supported types.
    """
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
    """Extract the trend component from STL decomposition for all columns.
    
    Performs STL decomposition and returns either the trend components or their strengths.
    
    Args:
        strength (bool, optional): If True, return trend strength metrics. If False, return trend values. Defaults to False.
        period (int, optional): Period for STL decomposition. If None, inferred from frequency. Defaults to None.
        freq (str, optional): Frequency of the time series. Defaults to None.
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
    
    Returns:
        pd.DataFrame: Trend values (DataFrame) if strength=False,
                     or trend strength metrics (Series) if strength=True.
    """
    res = self.stl(period, freq, decimals)
    if strength:
      return res["t_strength"]
    return res["trend"]

  def seasonal(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    """Extract the seasonal component from STL decomposition for all columns.
    
    Performs STL decomposition and returns either the seasonal components or their strengths.
    
    Args:
        strength (bool, optional): If True, return seasonal strength metrics. If False, return seasonal values. Defaults to False.
        period (int, optional): Period for STL decomposition. If None, inferred from frequency. Defaults to None.
        freq (str, optional): Frequency of the time series. Defaults to None.
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
    
    Returns:
        pd.DataFrame: Seasonal values (DataFrame) if strength=False,
                     or seasonal strength metrics (Series) if strength=True.
    """
    res = self.stl(period, freq, decimals)
    if strength:
      return res["s_strength"]
    return res["seasonal"]

  def residual(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    """Extract the residual component from STL decomposition for all columns.
    
    Performs STL decomposition and returns either the residual components or their strengths.
    
    Args:
        strength (bool, optional): If True, return residual strength metrics. If False, return residual values. Defaults to False.
        period (int, optional): Period for STL decomposition. If None, inferred from frequency. Defaults to None.
        freq (str, optional): Frequency of the time series. Defaults to None.
        decimals (int, optional): Number of decimal places for rounding. Defaults to 4.
    
    Returns:
        pd.DataFrame: Residual values (DataFrame) if strength=False,
                     or residual strength metrics (Series) if strength=True.
    """
    res = self.stl(period, freq, decimals)
    if strength:
      return res["r_strength"]
    return res["residual"]
