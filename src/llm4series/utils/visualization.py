from ..data import TimeSeries, UniTimeSeries, MultiTimeSeries
import plotly.graph_objects as go
from typing import Sequence, Literal
import plotly.express as px
import pandas as pd
import colorsys
import math


def _adjust_lightness(hex_color: str, lightness: float) -> str:
  """Adjust the lightness of a hexadecimal color.

  Args:
    hex_color: Hexadecimal color code (e.g., '#RRGGBB' or 'RRGGBB').
    lightness: Desired lightness value between 0 and 1.

  Returns:
    Hexadecimal color code with adjusted lightness.
  """
  hex_color = hex_color.lstrip("#")
  r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
  h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
  l = min(max(lightness, 0), 1)
  r, g, b = colorsys.hls_to_rgb(h, l, s)
  return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def _get_lightness_map(n: int, lightness: float) -> list[float]:
  """Generate a list of lightness values distributed from lightness to 0.3.

  Args:
    n: Number of lightness values to generate.
    lightness: Starting lightness value between 0 and 1.

  Returns:
    List of n lightness values distributed between lightness and 0.3, capped at 1.0.
  """
  if n <= 0:
    return []
  elif n == 1:
    return [lightness]
  step = (lightness - 0.3) / (n - 1)
  values = [lightness - i * step for i in range(n)]
  return [min(1.0, math.ceil(v * 100) / 100) for v in values]


def _get_color(i: int, lightness: float = 0.7) -> str:
  """Get a color from the Plotly qualitative palette with adjusted lightness.

  Args:
    i: Index to select color from the palette (wraps around using modulo).
    lightness: Desired lightness value between 0 and 1. Defaults to 0.7.

  Returns:
    Hexadecimal color code with adjusted lightness.
  """
  color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
  return _adjust_lightness(color, lightness)


def _get_groups(
    series: Sequence[TimeSeries],
    groups: list[str] = None
) -> list[str | None]:
  """Validate and process groups parameter for multiple series.

  Args:
    series: Sequence of time series objects.
    groups: List of group names. If None, returns None for each series.
            If list with 1 element, repeats it for each series.
            If list matches series length, returns as-is.

  Returns:
    List of group names matching the length of series.

  Raises:
    ValueError: If groups length doesn't match series length (and is not 1).
  """
  if groups is None:
    return [None] * len(series)
  elif len(groups) == 1:
    return groups * len(series)
  elif len(groups) != len(series):
    raise ValueError("The number of groups must match the number of series provided.")
  return groups


def linechart(
    *series: Sequence[TimeSeries],
    groups: list[str] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    showlegend: bool = True,
    lightness: float = 0.7,
    **kwargs
) -> go.Figure:
  """Create an interactive line chart with time series data.

  Args:
    *series: Variable number of TimeSeries, UniTimeSeries, MultiTimeSeries, pd.Series, or pd.DataFrame objects.
    groups: List of group names to label each series. Can be a single name to apply to all.
    title: Chart title.
    xlabel: X-axis label.
    ylabel: Y-axis label.
    showlegend: Whether to display the legend. Defaults to True.
    lightness: Lightness adjustment for colors (0-1). Defaults to 0.7.
    **kwargs: Additional arguments passed to fig.update_layout().

  Returns:
    plotly.graph_objects.Figure: Interactive line chart figure.

  Raises:
    TypeError: If series type is not supported.
    ValueError: If groups length doesn't match series length.
  """
  groups = _get_groups(series, groups)
  fig = go.Figure()
  for i, s in enumerate(series):
    if isinstance(s, UniTimeSeries) or isinstance(s, pd.Series):
      name = f"{groups[i]} - {s.name}" if groups[i] and s.name else groups[i] or s.name
      color = _get_color(i, lightness)
      fig.add_trace(go.Scatter(
          x=s.index, y=s.values, mode="lines", name=name, line=dict(color=color)))
    elif isinstance(s, MultiTimeSeries) or isinstance(s, pd.DataFrame):
      lightness_values = _get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = _get_color(i if groups[i] else j, lightness_values[j])
        fig.add_trace(go.Scatter(
            x=s.index, y=s[c], mode="lines", name=name, line=dict(color=color)))
    else:
      raise TypeError(f"Type not supported: {type(s).__name__}.")
  fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=showlegend, **kwargs)
  return fig


def lineplot(
    *series: Sequence[TimeSeries],
    groups: list[str] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    showlegend: bool = True,
    lightness: float = 0.7,
    **kwargs
) -> go.Figure:
  """Create an interactive line plot with index-based positioning.

  Args:
    *series: Variable number of TimeSeries, UniTimeSeries, MultiTimeSeries, pd.Series, or pd.DataFrame objects.
    groups: List of group names to label each series. Can be a single name to apply to all.
    title: Plot title.
    xlabel: X-axis label.
    ylabel: Y-axis label.
    showlegend: Whether to display the legend. Defaults to True.
    lightness: Lightness adjustment for colors (0-1). Defaults to 0.7.
    **kwargs: Additional arguments passed to fig.update_layout().

  Returns:
    plotly.graph_objects.Figure: Interactive line plot figure.

  Raises:
    TypeError: If series type is not supported.
    ValueError: If groups length doesn't match series length.
  """
  groups = _get_groups(series, groups)
  fig = go.Figure()
  for i, s in enumerate(series):
    if isinstance(s, UniTimeSeries) or isinstance(s, pd.Series):
      name = f"{groups[i]} - {s.name}" if groups[i] and s.name else groups[i] or s.name
      color = _get_color(i, lightness)
      fig.add_trace(go.Scatter(
          x=list(range(len(s))), y=s, mode="lines", name=name, line=dict(color=color)))
    elif isinstance(s, MultiTimeSeries) or isinstance(s, pd.DataFrame):
      lightness_values = _get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = _get_color(i if groups[i] else j, lightness_values[j])
        fig.add_trace(go.Scatter(
            x=list(range(len(s))), y=s[c], mode="lines", name=name, line=dict(color=color)))
    else:
      raise TypeError(f"Type not supported: {type(s).__name__}.")
  fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=showlegend, **kwargs)
  return fig


def barplot(
    *series: Sequence[TimeSeries],
    x: list[str] = None,
    groups: list[str] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    lightness: float = 0.7,
    **kwargs
) -> go.Figure:
  """Create an interactive bar plot showing statistical aggregates or values.

  For UniTimeSeries/pd.Series: Shows mean, std, max, min, median statistics.
  For MultiTimeSeries/pd.DataFrame: Shows statistics per column.

  Args:
    *series: Variable number of TimeSeries, UniTimeSeries, MultiTimeSeries, pd.Series, or pd.DataFrame objects.
    x: Custom x-axis labels. If not provided, uses 'mean', 'std', 'max', 'min', 'median' or index labels.
    groups: List of group names to label each series. Can be a single name to apply to all.
    title: Plot title.
    xlabel: X-axis label.
    ylabel: Y-axis label.
    lightness: Lightness adjustment for colors (0-1). Defaults to 0.7.
    **kwargs: Additional arguments passed to fig.update_layout().

  Returns:
    plotly.graph_objects.Figure: Interactive bar plot figure.

  Raises:
    TypeError: If series type is not supported.
    ValueError: If groups length doesn't match series length.
  """
  groups = _get_groups(series, groups)
  fig = go.Figure()
  for i, s in enumerate(series):
    if isinstance(s, MultiTimeSeries):
      lightness_values = _get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        stats = ["mean", "std", "max", "min", "median"]
        y = [getattr(s[c], func)() for func in stats]
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = _get_color(i, lightness_values[j]) if groups[i] else _get_color(
            j, lightness_values[j])
        fig.add_trace(go.Bar(x=x or stats, y=y, name=name, marker_color=color))
    elif isinstance(s, UniTimeSeries):
      stats = ["mean", "std", "max", "min", "median"]
      y = [getattr(s, func)() for func in stats]
      name = groups[i] or s.name
      color = _get_color(i, lightness)
      fig.add_trace(go.Bar(x=x or stats, y=y, name=name, marker_color=color))
    elif isinstance(s, pd.Series):
      name = groups[i] or s.name
      color = _get_color(i, lightness)
      fig.add_trace(go.Bar(x=x or s.index.astype(str),
                           y=s.values, name=name, marker_color=color))
    elif isinstance(s, pd.DataFrame):
      lightness_values = _get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = _get_color(i if groups[i] else j, lightness_values[j])
        fig.add_trace(go.Bar(x=x or s.index.astype(str),
                             y=s[c].values, name=name, marker_color=color))
    else:
      raise TypeError(f"Type not supported: {type(s).__name__}.")
  fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, barmode="group", **kwargs)
  return fig


def plot(
    *series: Sequence[TimeSeries],
    kind: Literal["line", "chart", "bar"],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    groups: list[str] = None,
    **kwargs
) -> go.Figure:
  """Create an interactive plot of the specified kind.

  Args:
    *series: Variable number of TimeSeries, UniTimeSeries, MultiTimeSeries, pd.Series, or pd.DataFrame objects.
    kind: Type of plot. Options: 'line' (line plot), 'chart' (line chart with datetime), 'bar' (bar plot).
    title: Plot title.
    xlabel: X-axis label.
    ylabel: Y-axis label.
    groups: List of group names to label each series. Can be a single name to apply to all.
    **kwargs: Additional arguments passed to the specific plotting function.

  Returns:
    plotly.graph_objects.Figure: Interactive plot figure.

  Raises:
    ValueError: If kind is not one of 'line', 'chart', or 'bar'.
    TypeError: If series type is not supported.
  """
  match kind:
    case "line":
      return lineplot(*series, title=title, xlabel=xlabel, ylabel=ylabel, groups=groups, **kwargs)
    case "chart":
      return linechart(*series, title=title, xlabel=xlabel, ylabel=ylabel, groups=groups, **kwargs)
    case "bar":
      return barplot(*series, title=title, xlabel=xlabel, ylabel=ylabel, groups=groups, **kwargs)
    case _:
      raise ValueError(f"Invalid kind: {kind}. Choose from 'line', 'chart', 'bar'.")
