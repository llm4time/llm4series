from ..data import TimeSeries, UniTimeSeries, MultiTimeSeries
import plotly.graph_objects as go
from typing import Sequence, Literal
import plotly.express as px
import pandas as pd
import colorsys
import math


def _adjust_lightness(hex_color: str, lightness: float) -> str:
  hex_color = hex_color.lstrip("#")
  r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
  h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
  l = min(max(lightness, 0), 1)
  r, g, b = colorsys.hls_to_rgb(h, l, s)
  return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def _get_lightness_map(n: int, lightness: float) -> list[float]:
  if n <= 0:
    return []
  elif n == 1:
    return [lightness]
  step = (lightness - 0.3) / (n - 1)
  values = [lightness - i * step for i in range(n)]
  return [min(1.0, math.ceil(v * 100) / 100) for v in values]


def _get_color(i: int, lightness: float = 0.7) -> str:
  color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
  return _adjust_lightness(color, lightness)


def _get_groups(
    series: Sequence[TimeSeries],
    groups: list[str] = None
) -> list[str | None]:
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
  match kind:
    case "line":
      return lineplot(*series, title=title, xlabel=xlabel, ylabel=ylabel, groups=groups, **kwargs)
    case "chart":
      return linechart(*series, title=title, xlabel=xlabel, ylabel=ylabel, groups=groups, **kwargs)
    case "bar":
      return barplot(*series, title=title, xlabel=xlabel, ylabel=ylabel, groups=groups, **kwargs)
    case _:
      raise ValueError(f"Invalid kind: {kind}. Choose from 'line', 'chart', 'bar'.")
