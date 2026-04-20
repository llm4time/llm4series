from typing import Self, Literal
from abc import ABC, abstractmethod
import pandas as pd
import random
import os


Sampling = Literal["frontend", "backend", "random", "uniform"]
TSFormat = Literal["array", "context", "csv", "custom", "json", "toon", "markdown", "plain", "symbol", "tsv"]
TSType = Literal["numeric", "textual"]


class TimeSeriesStatistics(ABC):

  @abstractmethod
  def stl(self: Self, period: int | None, freq: str | None, decimals: int = 4) -> dict:
    """
    Performs STL decomposition (Seasonal-Trend decomposition using LOESS) on the time series.

    Args:
        period (int, optional): Seasonality period.
        freq (str, optional): Time series frequency.
        decimals (int, optional): Number of decimal places for rounding.

    Returns:
        dict: A dictionary containing the time series components (trend, seasonality, residuals)
              and their relative strengths.
    """
    ...

  @abstractmethod
  def mean(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Computes the arithmetic mean of the time series.

    Args:
        decimals (int | None): Number of decimal places for rounding.
                               If None, no rounding is applied.
        **kwargs: Additional implementation-specific arguments.

    Returns:
        float: Mean value of the time series.
    """
    ...

  @abstractmethod
  def median(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Computes the median of the time series.

    Args:
        decimals (int | None): Number of decimal places for rounding.
                               If None, no rounding is applied.
        **kwargs: Additional implementation-specific arguments.

    Returns:
        float: Median value of the time series.
    """
    ...


class TimeSeriesImputation(ABC):

  @abstractmethod
  def impute_mean(self: Self, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Replaces missing values with the mean of the series.

    Computes the mean of the available values in the series and fills missing values
    with this value rounded to the specified number of decimal places.

    Args:
        decimals (int | None): Number of decimal places to round the mean.
        inplace (bool | None): If True, modifies the original series. If False, returns
                              a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.
    """
    ...

  @abstractmethod
  def impute_median(self: Self, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Replaces missing values with the median of the series.

    Computes the median of the available values in the series and fills missing values
    with this value rounded to the specified number of decimal places.

    Args:
        decimals (int | None): Number of decimal places to round the median.
        inplace (bool | None): If True, modifies the original series. If False, returns
                              a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.
    """
    ...

  @abstractmethod
  def impute_ffill(self: Self, inplace: bool | None) -> Self | None:
    """
    Imputes missing values using forward fill followed by backward fill.

    Fills missing values by propagating the last valid value forward and then
    propagating the next valid value backward if missing values remain.

    Args:
        inplace (bool | None): If True, modifies the original series. If False, returns
                              a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.
    """
    ...

  @abstractmethod
  def impute_bfill(self: Self, inplace: bool | None) -> Self | None:
    """
    Imputes missing values using backward fill followed by forward fill.

    Fills missing values by propagating the next valid value backward and then
    propagating the last valid value forward if missing values remain.

    Args:
        inplace (bool | None): If True, modifies the original series. If False, returns
                               a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.
    """
    ...

  @abstractmethod
  def impute_sma(self: Self, window: int, min_periods: int | None, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Imputes missing values using Simple Moving Average (SMA).

    Computes the simple moving average with the specified window and replaces
    missing values. After SMA, forward and backward fill are applied to fill
    remaining values.

    Args:
        window (int): Window size for moving average calculation.
        min_periods (int): Minimum number of non-null values required
                           to compute the mean.
        decimals (int | None): Number of decimal places for rounding.
        inplace (bool | None): If True, modifies the original series. If False, returns
                               a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.
    """
    ...

  @abstractmethod
  def impute_ema(self: Self, span: int, adjust: bool | None, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Imputes missing values using Exponential Moving Average (EMA).

    Computes the EMA with the specified span and fills missing values.
    After EMA, forward and backward fill are applied to fill remaining values.

    Args:
        span (int): Period for the exponential moving average.
        adjust (bool): If True, adjusts weights to account for the full series.
        decimals (int | None): Number of decimal places for rounding.
        inplace (bool | None): If True, modifies the original series. If False, returns
                               a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.
    """
    ...

  @abstractmethod
  def impute_interpolate(self: Self, method: str, order: int | None, inplace: bool | None) -> Self | None:
    """
    Imputes missing values using interpolation.

    Supports linear or spline interpolation. After interpolation,
    forward and backward fill are applied to fill remaining values.

    Args:
        method (str): Interpolation method. Accepted values: 'linear', 'spline'.
        order (int | None): Spline order, if method='spline'.
        inplace (bool | None): If True, modifies the original series. If False, returns
                               a new series.

    Returns:
        TimeSeries | None: Time series with missing values imputed,
                           or None if inplace=True.

    Raises:
        ValueError: If the provided method is not supported.
    """
    ...

  @abstractmethod
  def std(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Computes the standard deviation of the time series.

    Args:
        decimals (int | None): Number of decimal places for rounding.
                               If None, no rounding is applied.
        **kwargs: Additional implementation-specific arguments.

    Returns:
        float: Standard deviation of the time series.
    """
    ...

  @abstractmethod
  def min(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Returns the minimum value present in the time series.

    Args:
        decimals (int | None): Number of decimal places for rounding.
                               If None, no rounding is applied.
        **kwargs: Additional implementation-specific arguments.

    Returns:
        float: Minimum value of the time series.
    """
    ...

  @abstractmethod
  def max(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Returns the maximum value present in the time series.

    Args:
        decimals (int | None): Number of decimal places for rounding.
                               If None, no rounding is applied.
        **kwargs: Additional implementation-specific arguments.

    Returns:
        float: Maximum value of the time series.
    """
    ...

  @abstractmethod
  def quantile(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Computes the quantile of the time series for a specified probability value.

    Args:
        decimals (int | None): Number of decimal places for rounding.
                               If None, no rounding is applied.
        **kwargs: Additional arguments, including parameter 'q' for the desired quantile (0 <= q <= 1).

    Returns:
        float: Corresponding quantile value of the time series.
    """
    ...


class TimeSeriesMetrics(ABC):

  @abstractmethod
  def smape(self: Self, y_pred: list[float], decimals: int | None) -> float:
    """
    sMAPE — Symmetric Mean Absolute Percentage Error.

    Measures the average absolute percentage error between observed and predicted values,
    normalized by the average of the absolute observed and predicted values.

    Args:
        y_pred (list[float]): Predicted values.
        decimals (int, optional): Number of decimal places for rounding.

    Returns:
        float: sMAPE value.
    """
    ...

  @abstractmethod
  def mae(self: Self, y_pred: list[float], decimals: int | None) -> float:
    """
    MAE — Mean Absolute Error.

    Measures the average absolute errors between observed and predicted values,
    providing a direct measure of forecast accuracy.

    Args:
        y_pred (list[float]): Predicted values.
        decimals (int, optional): Number of decimal places for rounding.

    Returns:
        float: MAE value.
    """
    ...

  @abstractmethod
  def rmse(self: Self, y_pred: list[float], decimals: int | None) -> float:
    """
    RMSE — Root Mean Squared Error.

    Measures the average squared errors between observed and predicted values,
    penalizing larger errors.

    Args:
        y_pred (list[float]): Predicted values.
        decimals (int, optional): Number of decimal places for rounding.

    Returns:
        float: RMSE value.
    """
    ...


class TimeSeriesPlot(ABC):

  @abstractmethod
  def linechart(self: Self):
    """
    Creates a line chart of the time series.
    """
    ...

  @abstractmethod
  def lineplot(self: Self):
    """
    Creates a line plot for a time series indexed by periods.
    """
    ...

  @abstractmethod
  def barplot(self: Self):
    """
    Creates a bar chart of statistics.
    """
    ...

  @abstractmethod
  def stlplot(self: Self, period: int = None, freq: str = None, decimals: int = None):
    """
    Creates plots of the STL decomposition components (trend, seasonality, residuals).
    """
    ...


class TimeSeries(ABC):
  @abstractmethod
  def agg_duplicates(self: Self, method: str, inplace: bool | None) -> Self | None:
    """
    Removes or aggregates duplicate values in the time series index.

    Args:
        method (str): Method used to handle duplicates.
        inplace (bool | None): If True, modifies the current object and returns None.
            If False, returns a new copy with duplicates resolved.

    Returns:
        Self | None: Time series with duplicates resolved or None if `inplace=True`.
    """
    ...

  def normalize(self: Self, freq: str, start: str = None, end: str = None) -> Self:
    """
    Reindexes the time series to a specific frequency, filling gaps.

    Creates a continuous index between the start and end dates and reindexes the series,
    filling missing values with NaN.

    Args:
        freq (str): Time series frequency. If None, attempts to infer from the series.
        start (str | None): Start date. If None, uses the earliest date in the series.
        end (str | None): End date. If None, uses the latest date in the series.

    Returns:
        TimeSeries | None: Reindexed and normalized time series.

    Raises:
        ValueError: If the frequency cannot be inferred automatically.
    """
    start_date = pd.to_datetime(start) if start else self.index.min()
    end_date = pd.to_datetime(end) if end else self.index.max()

    if freq is None:
      freq = self.index.freq
      if freq is None:
        raise ValueError("Error trying to infer frequency automatically.")

    full_idx = pd.date_range(start=start_date, end=end_date, freq=freq)
    return self.reindex(full_idx, fill_value=pd.NA)

  def split(self: Self, start: str | pd.DatetimeIndex | None = None, end: str | pd.DatetimeIndex | None = None, test_size: float | None = None, periods: int | None = None) -> tuple[Self, Self]:
    """
    Splits the time series into two parts.

    Can split by date range or by proportion of data.

    Args:
        start (str | pd.DatetimeIndex | None): Start date of the training set.
        end (str | pd.DatetimeIndex | None): End date of the training set.
        periods (int | None): Number of values to include in the test set.
                              If None, includes all data after end.
        test_size (float | None): Proportion of data for the test set (0 to 1).
                                  If provided, ignores start, end and periods.

    Returns:
        tuple[TimeSeries, TimeSeries]: Pair of time series (training, test).
    """
    if test_size is not None:
        if not (0 < test_size < 1):
            raise ValueError("Error: `test_size` must be between 0 and 1.")
        idx = int(len(self) * (1 - test_size))
        train = self.iloc[:idx]
        test = self.iloc[idx:] if periods is None else self.iloc[idx:][:periods]
        return train, test

    train = self[(self.index >= str(start)) & (self.index <= str(end))]
    test = self[self.index > str(end)] if periods is None else self[self.index > str(end)][:periods]
    return train, test

  def slide(self: Self, method: Sampling, window: int, samples: int, step: int = None) -> list[tuple[Self, Self]]:
    """
    Generates sequential samples of the time series as window pairs (input, output).

    Creates consecutive subsets of the time series using different sampling strategies,
    where each sample consists of an input window immediately followed by an output window.
    The method defines how the starting windows are selected.

    Args:
        method (Sampling): Sampling strategy. Supported methods:
            - 'frontend': Generates sequential windows from the beginning of the series.
            - 'backend': Generates sequential windows from the end of the series.
            - 'random': Randomly selects starting points for windows.
            - 'uniform': Generates windows uniformly distributed across the series.
        window (int): Size of each window.
        samples (int): Total number of samples to generate.
        step (int): Interval between starting points of windows.

    Returns:
        list[tuple[TimeSeries, TimeSeries]]: List of (input, output) tuples, where each
            element represents a sample containing two consecutive windows of the series.

    Raises:
        ValueError: If the specified method is not supported:
                    'frontend', 'backend', 'random', or 'uniform'.
    """
    max_start = len(self) - 2 * window

    if method == "frontend":
      idxs = [i * 2 * window for i in range(samples)]

    elif method == "backend":
      total = len(self) // window - 1
      samples = min(samples, total)
      idxs = [len(self) - (samples - i) * 2 * window for i in range(samples)]

    elif method == "random":
      if max_start < 0:
        return []
      idxs = sorted(random.sample(range(max_start + 1), k=min(samples, max_start + 1)))

    elif method == "uniform":
      if max_start < 0 or samples <= 0:
        return []
      if step is None:
        step = max_start / (samples - 1) if samples > 1 else 0
        idxs = [int(i * step) for i in range(samples)]
      else:
        idxs = list(range(0, max_start + 1, step))[:samples]

    else:
      raise ValueError('Supported methods: frontend, backend, random, uniform.')

    windows = []
    for idx in idxs:
      end_out = idx + 2 * window
      if end_out > len(self):
        break
      windows.append((
          self._constructor(self.iloc[idx:idx + window].copy()),
          self._constructor(self.iloc[idx + window:end_out].copy())
      ))
    return windows

  def to_str(self: Self, format: TSFormat, type: TSType = "numeric") -> str:
    """
    Converts the time series to a string representation in various formats.

    Args:
        format (TSFormat): Desired output format. Supported formats:
            - TSFormat.ARRAY: Returns the series as an array.
            - TSFormat.CONTEXT: Returns the series in contextual format.
            - TSFormat.CSV: Returns the series as CSV.
            - TSFormat.CUSTOM: Custom format.
            - TSFormat.JSON: Returns the series as JSON.
            - TSFormat.MARKDOWN: Returns the series as a Markdown table.
            - TSFormat.PLAIN: Returns the series as plain text.
            - TSFormat.SYMBOL: Returns the series using symbolic notation.
            - TSFormat.TSV: Returns the series as TSV.
        type (TSType, optional): Desired representation type. Can be:
            - TSType.NUMERIC (default): Keeps numeric values.
            - TSType.TEXTUAL: Converts the series into an encoded textual form.

    Returns:
        str: String representation of the time series in the specified format and type.

    Raises:
        ValueError: If the provided `format` is not supported.
    """
    from .. import formatting as fmt

    ts = fmt.encode_textual(self) if type == "textual" else self

    formats_map = {
        "array": fmt._to_array,
        "context": fmt._to_context,
        "csv": fmt._to_csv,
        "custom": fmt._to_custom,
        "json": fmt._to_json,
        "markdown": fmt._to_markdown,
        "plain": fmt._to_plain,
        "symbol": fmt._to_symbol,
        "toon": fmt._to_toon,
        "tsv": fmt._to_tsv,
    }
    if format not in formats_map:
      raise ValueError(f"Unknown format: {format}.")
    try:
      return formats_map[format](ts)
    except Exception:
      raise ValueError(f"Failed to convert TimeSeries to format {format}.")

  def to_file(self: Self, path: str) -> None:
    """
    Saves the time series to a file in the format specified by the file extension.

    Exports the time series to disk in one of the supported formats:
    CSV, Excel (XLSX), JSON, or Parquet. The file extension is used to
    automatically determine the export format.

    Args:
        path (str): Full output file path, including name and extension
                    (e.g., 'data/series.csv').
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".csv":
      self.to_csv(path, index=True)
    elif ext == ".xlsx":
      self.to_excel(path, index=True)
    elif ext == ".json":
      self.to_json(path, orient="records", date_format="iso")
    elif ext == ".parquet":
      self.to_parquet(path, index=True)
    else:
      raise ValueError(f"Supported extensions: .csv, .xlsx, .json, .parquet")
