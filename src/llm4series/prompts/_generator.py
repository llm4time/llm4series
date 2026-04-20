import llm4series.data as ls
import pandas as pd
from typing import Literal
from ._templates import *


PromptType = Literal["zero_shot", "few_shot", "cot", "cot_few", "custom"]


def prompt(
    type: PromptType,
    ts: ls.TimeSeries,
    tsformat: ls.TSFormat,
    tstype: ls.TSType,
    forecast_horizon: int,
    examples: int = 0,
    sampling: ls.Sampling = None,
    template: str = None,
    stl: dict = None,
    **kwargs
) -> str:
  if template is None and type == "custom":
    raise ValueError("Template must be set for custom prompt.")
  if examples == 0 and type in ["few_shot", "cot_few"]:
    raise ValueError("Must contain at least 1 example.")

  base_kwargs = {
      "input_len": len(ts),
      "output_example": ts[:forecast_horizon].to_str(tsformat, tstype),
      "forecast_horizon": forecast_horizon,
  }
  base_kwargs.update(kwargs)

  def _statistics(series, stl_col=None):
    lines = [
        f"- Mean: {series.mean()}",
        f"- Median: {series.median()}",
        f"- Standard Deviation: {series.std()}",
        f"- Minimum Value: {series.min()}",
        f"- Maximum Value: {series.max()}",
        f"- First Quartile (Q1): {series.quantile(0.25)}",
        f"- Third Quartile (Q3): {series.quantile(0.75)}"
    ]
    if stl_col is not None:
      trend = stl_col.get("t_strength")
      if isinstance(trend, pd.Series):
        trend = trend.iloc[series.name]
      if trend is not None:
        lines.append(f"- Trend Strength (STL): {trend}")
      seasonal = stl_col.get("s_strength")
      if isinstance(seasonal, pd.Series):
        seasonal = seasonal.iloc[series.name]
      if seasonal is not None:
        lines.append(f"- Seasonality Strength (STL): {seasonal}")
    return "\n".join(lines)

  if isinstance(ts, ls.UniTimeSeries):
    base_kwargs["statistics"] = _statistics(ts, stl)
  elif isinstance(ts, ls.MultiTimeSeries):
    rows = []
    for col in ts.num_columns:
      stl_col = {k: v.get(col) for k, v in stl.items()} if stl is not None else None
      header = f"Column: {col}\n" if len(ts.num_columns) > 1 else ""
      rows.append(header + _statistics(ts[col], stl_col))
    base_kwargs["statistics"] = "\n".join(rows)
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  min_periods = forecast_horizon * 2 * examples
  if len(ts) < min_periods:
    raise ValueError(
        f"For {examples} examples there must be {min_periods} periods in the time series.")

  try:
    sampling = sampling or "backend"
    if sampling not in ["frontend", "backend", "random", "uniform"]:
      raise ValueError()
  except ValueError:
    raise ValueError("Supported samplings: frontend, backend, random, uniform.")

  if "forecast_examples" not in kwargs:
    forecast_examples = "\n".join([
        f"- Example {i}:\n"
        f"Input (history):\n{input.to_str(tsformat, tstype)}\n\n"
        f"Output (forecast):\n<out>\n{output.to_str(tsformat, tstype)}\n</out>"
        f"{'' if i == examples else '\n'}"
        for i, (input, output) in enumerate(
            ts.slide(method=sampling, window=forecast_horizon, samples=examples),
            start=1)
    ])
    base_kwargs.update({"forecast_examples": forecast_examples})

  prompt_map = {
      "zero_shot": ZERO_SHOT,
      "few_shot": FEW_SHOT,
      "cot_few": COT_FEW,
      "cot": COT,
      "custom": template
  }
  if type not in prompt_map:
    raise ValueError("Supported prompts: zero_shot, few_shot, cot, cot_few, custom.")

  try:
    return prompt_map[type].format(**base_kwargs)
  except KeyError as e:
    raise ValueError(f"Key {e} not defined.")
