import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import override, Callable
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from ._core import TimeSeries, TimeSeriesImputation, TimeSeriesStatistics, TimeSeriesMetrics, TimeSeriesPlot


class _UniTimeSeriesImputation(TimeSeriesImputation):

  def _apply_imputation(self, func: Callable, inplace: bool = False):
    ts = self if inplace else self.copy()
    func(ts)
    return None if inplace else ts

  @override
  def impute_mean(self, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    return self.fillna(round(self.mean(), decimals), inplace=inplace)

  @override
  def impute_median(self, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    return self.fillna(round(self.median(), decimals), inplace=inplace)

  @override
  def impute_ffill(self, inplace: bool = False) -> TimeSeries | None:
    return self._apply_imputation(lambda ts: (ts.ffill(inplace=True), ts.bfill(inplace=True)), inplace)

  @override
  def impute_bfill(self, inplace: bool = False) -> TimeSeries | None:
    return self._apply_imputation(lambda ts: (ts.bfill(inplace=True), ts.ffill(inplace=True)), inplace)

  @override
  def impute_sma(self, window: int, min_periods: int = 1, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts.fillna(ts.rolling(window=window, min_periods=min_periods).mean().round(decimals), inplace=True)
    ts.impute_ffill(inplace=True)
    return None if inplace else ts

  @override
  def impute_ema(self, span: int, adjust: bool = False, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts.fillna(ts.ewm(span=span, adjust=adjust).mean().round(decimals), inplace=True)
    ts.impute_ffill(inplace=True)
    return None if inplace else ts

  @override
  def impute_interpolate(self, method: str = 'linear', order: int = 2, inplace: bool = False) -> TimeSeries | None:
    if method not in ('linear', 'spline'):
      raise ValueError("Supported methods: linear or spline.")
    ts = self if inplace else self.copy()
    try:
      ts.interpolate(method=method if method == 'linear' else 'spline', order=order if method == 'spline' else None, inplace=True)
    except (ValueError, TypeError):
      ts.interpolate(method='linear', inplace=True)
    ts.impute_ffill(inplace=True)
    return None if inplace else ts


class _MultiTimeSeriesImputation(TimeSeriesImputation):

  def _impute_cols(self, cols, func: Callable, inplace: bool = False):
    ts = self if inplace else self.copy()
    ts[cols] = func(ts[cols])
    return None if inplace else ts

  @override
  def impute_mean(self, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    return self._impute_cols(self.num_columns,
      lambda x: x.fillna(x.mean().round(decimals)), inplace)

  @override
  def impute_median(self, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    return self._impute_cols(self.num_columns,
      lambda x: x.fillna(x.median().round(decimals)), inplace)

  @override
  def impute_ffill(self, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.cat_columns] = ts[self.cat_columns].ffill().bfill()
    ts[self.num_columns] = ts[self.num_columns].ffill().bfill()
    return None if inplace else ts

  @override
  def impute_bfill(self, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.cat_columns] = ts[self.cat_columns].bfill().ffill()
    ts[self.num_columns] = ts[self.num_columns].bfill().ffill()
    return None if inplace else ts

  @override
  def impute_sma(self, window: int, min_periods: int = 1, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    ts = self._impute_cols(self.num_columns,
      lambda x: x.fillna(x.rolling(window=window, min_periods=min_periods).mean().round(decimals)), inplace)
    ts.impute_ffill(inplace=True) if ts is not None else self.impute_ffill(inplace=True)
    return ts

  @override
  def impute_ema(self, span: int, adjust: bool = False, decimals: int = 4, inplace: bool = False) -> TimeSeries | None:
    ts = self._impute_cols(self.num_columns,
      lambda x: x.fillna(x.ewm(span=span, adjust=adjust).mean().round(decimals)), inplace)
    ts.impute_ffill(inplace=True) if ts is not None else self.impute_ffill(inplace=True)
    return ts

  @override
  def impute_interpolate(self, method: str = 'linear', order: int = 2, inplace: bool = False) -> TimeSeries | None:
    if method not in ('linear', 'spline'):
      raise ValueError("Supported methods: linear or spline.")
    ts = self if inplace else self.copy()
    try:
      interp_kwargs = {'method': method if method == 'linear' else 'spline'}
      if method == 'spline':
        interp_kwargs['order'] = order
      ts[self.num_columns] = ts[self.num_columns].interpolate(**interp_kwargs)
    except (ValueError, TypeError):
      ts[self.num_columns] = ts[self.num_columns].interpolate(method='linear')
    ts.impute_ffill(inplace=True)
    return None if inplace else ts


class _UniTimeSeriesStatistics(TimeSeriesStatistics):

  @override
  def stl(self, period: int = None, freq: str = None, decimals: int = 4) -> pd.Series | None:
    from .._internal import logger
    if hasattr(self.index, 'dtype') and pd.api.types.is_integer_dtype(self.index):
      raise ValueError("Time series index must be datetime-like (e.g., pd.DatetimeIndex), not integer. "
                       "STL decomposition requires a proper time frequency.")
    ts = self.copy()
    freq = freq or ts.index.freqstr
    if period and (not isinstance(period, int) or period <= 0):
      raise ValueError("period must be a positive integer")
    if decimals and (not isinstance(decimals, int) or decimals < 0):
      raise ValueError("decimals must be a non-negative integer")

    if freq:
      freq_map = {"ms": 1000, "s": 60, "min": 60, "h": 24, "d": 7, "m": 12, "y": 10}
      period = period or freq_map.get(freq.lower())
      try:
        ts = ts.asfreq(freq)
      except (ValueError, TypeError) as e:
        logger.error(f"Failed to set series frequency to '{freq}': {e}")
        return pd.Series({"trend": self.__class__(dtype=float), "seasonal": self.__class__(dtype=float),
                         "residual": self.__class__(dtype=float), "t_strength": pd.NA, "s_strength": pd.NA, "r_strength": pd.NA})

    try:
      res = STL(ts.dropna(), period=period).fit()
      trend = res.trend.round(decimals)
      seasonal = res.seasonal.round(decimals)
      resid = res.resid.round(decimals)
      var_r, var_t, var_s = np.var(resid), np.var(trend), np.var(seasonal)
      total_var = var_t + var_s + var_r
      t_strength = round(var_t / total_var, decimals) if total_var > 0 else np.nan
      s_strength = round(var_s / total_var, decimals) if total_var > 0 else np.nan
      r_strength = round(var_r / total_var, decimals) if total_var > 0 else np.nan
      return pd.Series({"trend": self.__class__(trend), "seasonal": self.__class__(seasonal),
                       "residual": self.__class__(resid), "t_strength": t_strength,
                       "s_strength": s_strength, "r_strength": r_strength})
    except (ValueError, TypeError, RuntimeError) as e:
      logger.error(f"STL decomposition failed: {e}")
      return None

  def adfuller(self) -> pd.Series | None:
    from .._internal import logger
    try:
      result = adfuller(self.dropna())
      return pd.Series({"test_statistic": result[0], "p_value": result[1], "used_lag": result[2],
                       "n_obs": result[3], "icbest": result[5], "critical_value_1%": result[4]["1%"],
                       "critical_value_5%": result[4]["5%"], "critical_value_10%": result[4]["10%"]})
    except (ValueError, TypeError) as e:
      logger.error(f"ADF test failed: {e}")
      return None

  def acf(self, adjusted: bool = False, nlags: int = None, qstat: bool = False, fft: bool = True,
          alpha = None, bartlett_confint: bool = True, missing: str = "none"):
    from .._internal import logger
    try:
      return acf(self.dropna(), adjusted=adjusted, nlags=nlags, qstat=qstat, fft=fft,
                 alpha=alpha, bartlett_confint=bartlett_confint, missing=missing)
    except (ValueError, TypeError) as e:
      logger.error(f"ACF calculation failed: {e}")
      return None

  def pacf(self, nlags: int = None, method: str = "ywadjusted", alpha: float = None):
    from .._internal import logger
    try:
      return pacf(self.dropna(), nlags=nlags, method=method, alpha=alpha)
    except (ValueError, TypeError) as e:
      logger.error(f"PACF calculation failed: {e}")
      return None


class _MultiTimeSeriesStatistics(TimeSeriesStatistics):

  @override
  def stl(self, period: int = None, freq: str = None, decimals: int = 4) -> dict | None:
    from .._internal import logger
    trend_dict, seasonal_dict, resid_dict = {}, {}, {}
    t_strengths, s_strengths, r_strengths = {}, {}, {}
    for col in self.columns:
      ts = self[col].copy()
      if hasattr(ts, "stl"):
        res = ts.stl(period=period, freq=freq, decimals=decimals)
        if res is not None:
          trend_dict[col], seasonal_dict[col], resid_dict[col] = res["trend"], res["seasonal"], res["residual"]
          t_strengths[col], s_strengths[col], r_strengths[col] = res["t_strength"], res["s_strength"], res["r_strength"]
      else:
        logger.warning(f"Column '{col}' does not support STL decomposition.")
    return {"trend": pd.DataFrame(trend_dict), "seasonal": pd.DataFrame(seasonal_dict),
            "residual": pd.DataFrame(resid_dict), "t_strength": pd.Series(t_strengths),
            "s_strength": pd.Series(s_strengths), "r_strength": pd.Series(r_strengths)} if trend_dict else None


class _UniTimeSeriesMetrics(TimeSeriesMetrics):
  from sklearn.metrics import mean_absolute_error, root_mean_squared_error

  @override
  def smape(self, y_pred: list[float], decimals: int = 2) -> float:
    y_true = self[~np.isnan(self)]
    y_pred = np.array(y_pred)[~np.isnan(y_pred)]
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    epsilon = 1e-10
    smape = np.mean(numerator / (denominator + epsilon)) * 100
    return round(smape, decimals)

  @override
  def mae(self, y_pred: list[float], decimals: int = 2) -> float:
    from sklearn.metrics import mean_absolute_error
    y_true = self[~np.isnan(self)]
    y_pred = np.array(y_pred)[~np.isnan(y_pred)]
    mae = mean_absolute_error(y_true, y_pred)
    return round(mae, decimals)

  @override
  def rmse(self, y_pred: list[float], decimals: int = 2) -> float:
    from sklearn.metrics import root_mean_squared_error
    y_true = self[~np.isnan(self)]
    y_pred = np.array(y_pred)[~np.isnan(y_pred)]
    rmse = root_mean_squared_error(y_true, y_pred)
    return round(rmse, decimals)

  def metrics(self, y_pred: list[float], decimals: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "smape": self.smape(y_pred, decimals),
        "mae": self.mae(y_pred, decimals),
        "rmse": self.rmse(y_pred, decimals)
    }, index=[self.name]).T


class _MultiTimeSeriesMetrics:

  @override
  def smape(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.Series:
    return pd.Series({
        col: self[col].smape(y_pred[col], decimals)
        for col in self.columns
    })

  @override
  def mae(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.Series:
    return pd.Series({
        col: self[col].mae(y_pred[col], decimals)
        for col in self.columns
    })

  @override
  def rmse(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.Series:
    return pd.Series({
        col: self[col].rmse(y_pred[col], decimals)
        for col in self.columns
    })

  def metrics(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "smape": self.smape(y_pred, decimals),
        "mae": self.mae(y_pred, decimals),
        "rmse": self.rmse(y_pred, decimals)
    }).T


class _UniTimeSeriesPlot(TimeSeriesPlot):

  @override
  def linechart(self, title: str = None, xlabel: str = None, ylabel: str = None,
                showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=self.index, y=self, mode="lines", name=self.name,
                            line=dict(color=_get_color(0, lightness))))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=showlegend, **kwargs)
    return fig

  @override
  def lineplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
               showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(self.values))), y=self.values, mode="lines", name=self.name,
                            line=dict(color=_get_color(0, lightness))))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=showlegend, **kwargs)
    return fig

  @override
  def barplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
              x: list[str] = None, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color
    labels = ["mean", "std", "max", "min", "median"]
    y = [getattr(self, func)() for func in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x or labels, y=y, marker_color=_get_color(0, lightness)))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, barmode="group", **kwargs)
    return fig

  @override
  def stlplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
              x: list[str] = None, period: int = None, freq: str = None, decimals: int = 4,
              showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color, _adjust_lightness
    stl = self.stl(period=period, freq=freq, decimals=decimals)
    x = x or ["Trend", "Seasonal", "Residual"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=x)
    colors = {"trend": "#FFA500", "seasonal": "#008000", "residual": "#FF0000"}
    for i, component in enumerate(["trend", "seasonal", "residual"]):
      fig.add_trace(go.Scatter(x=stl[component].index, y=stl[component].values, mode="lines", name=x[i],
                              line=dict(color=_adjust_lightness(colors[component], lightness))), row=i+1, col=1)
    fig.update_layout(title=title, showlegend=showlegend, **kwargs)
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    fig.update_yaxes(title_text=ylabel, row=2, col=1)
    return fig

  def autocorrplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
                   x: list[str] = None, nlags: int = 40, alpha: float = 0.05, **kwargs) -> go.Figure:
    acf_res, pacf_res = self.acf(nlags=nlags, alpha=alpha), self.pacf(nlags=nlags, alpha=alpha, method="ywm")
    acf_vals, acf_ci, pacf_vals, pacf_ci = acf_res[0], acf_res[1], pacf_res[0], pacf_res[1]
    lags = list(range(len(acf_vals)))
    acf_lower, acf_upper = acf_ci[:, 0] - acf_vals, acf_ci[:, 1] - acf_vals
    pacf_lower, pacf_upper = pacf_ci[:, 0] - pacf_vals, pacf_ci[:, 1] - pacf_vals
    fig = make_subplots(rows=1, cols=2, subplot_titles=x or ["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"])
    def add_stems(fig, lags, vals, lower, upper, col):
      for lag, val in zip(lags, vals):
        fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines", line=dict(color="steelblue", width=1.5),
                                showlegend=False), row=1, col=col)
      fig.add_trace(go.Scatter(x=lags, y=vals, mode="markers", marker=dict(color="red", size=6),
                              showlegend=False), row=1, col=col)
      fig.add_trace(go.Scatter(x=lags, y=upper, mode="lines", line=dict(color="blue", dash="dash", width=1),
                              showlegend=False), row=1, col=col)
      fig.add_trace(go.Scatter(x=lags, y=lower, mode="lines", line=dict(color="blue", dash="dash", width=1),
                              fill="tonexty", fillcolor="rgba(0,0,255,0.08)", showlegend=False), row=1, col=col)
      fig.add_hline(y=0, line=dict(color="black", width=1), row=1, col=col)
    add_stems(fig, lags, acf_vals, acf_lower, acf_upper, 1)
    add_stems(fig, lags, pacf_vals, pacf_lower, pacf_upper, 2)
    fig.update_layout(title=title, **kwargs)
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    return fig


class _MultiTimeSeriesPlot(TimeSeriesPlot):

  @override
  def linechart(self, title: str = None, xlabel: str = None, ylabel: str = None,
                showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color
    fig = go.Figure()
    for i, col in enumerate(self.num_columns):
      fig.add_trace(go.Scatter(x=self.index, y=self[col], mode="lines", name=col,
                              line=dict(color=_get_color(i, lightness))))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=showlegend, **kwargs)
    return fig

  @override
  def lineplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
               showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color
    fig = go.Figure()
    x = list(range(len(self.values)))
    for i, col in enumerate(self.num_columns):
      fig.add_trace(go.Scatter(x=x, y=self[col], mode="lines", name=col,
                              line=dict(color=_get_color(i, lightness))))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=showlegend, **kwargs)
    return fig

  @override
  def barplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
              x: list[str] = None, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color
    fig = go.Figure()
    stats = ["mean", "std", "max", "min", "median"]
    for i, col in enumerate(self.num_columns):
      y = [getattr(self[col], func)() for func in stats]
      fig.add_trace(go.Bar(x=x or stats, y=y, name=col, marker_color=_get_color(i, lightness)))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, barmode="group", **kwargs)
    return fig

  @override
  def stlplot(self, title: str = None, xlabel: str = None, ylabel: str = None,
              x: list[str] = None, period: int = None, freq: str = None, decimals: int = 4,
              showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    from ..utils.visualization import _get_color, _adjust_lightness, _get_lightness_map
    stl = self.stl(period=period, freq=freq, decimals=decimals)
    x = x or ["Trend", "Seasonal", "Residual"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=x)
    colors = {"trend": "#FFA500", "seasonal": "#008000", "residual": "#FF0000"}
    for i, component in enumerate(["trend", "seasonal", "residual"]):
      df = stl[component]
      lightness_values = _get_lightness_map(len(df.columns), lightness)
      for j, col_name in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col_name], mode="lines", name=col_name,
                                line=dict(color=_adjust_lightness(colors[component], lightness_values[j]))), row=i+1, col=1)
    fig.update_layout(title=title, showlegend=showlegend, **kwargs)
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    fig.update_yaxes(title_text=ylabel, row=2, col=1)
    return fig
