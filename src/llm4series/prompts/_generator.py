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
  """Generate a prompt for time series forecasting based on the specified template type.

  Args:
    type: Prompt template type. Options: 'zero_shot', 'few_shot', 'cot', 'cot_few', 'custom'.
    ts: Time series data (UniTimeSeries or MultiTimeSeries).
    tsformat: Format for representing the time series (e.g., array, csv, markdown).
    tstype: Type of time series representation (e.g., values, datetime).
    forecast_horizon: Number of future periods to forecast.
    examples: Number of examples to include in few-shot prompts. Must be > 0 for few_shot/cot_few types.
    sampling: Sampling method for selecting examples ('frontend', 'backend', 'random', 'uniform'). Defaults to 'backend'.
    template: Custom prompt template string. Required only for 'custom' type.
    stl: Optional dictionary containing STL decomposition components (trend strength, seasonality strength).
    **kwargs: Additional format arguments to be passed to the template.

  Returns:
    str: Formatted prompt ready for use with language models.

  Raises:
    ValueError: If template is missing for 'custom' type, examples is 0 for few-shot types,
                insufficient data for the requested examples, invalid sampling method,
                or invalid prompt type.
    TypeError: If ts is not a TimeSeries type.
    KeyError: If a required key is missing from the prompt template.
  """
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
    """Calculate and format statistical summary of a time series.

    Args:
      series: Time series data to compute statistics for.
      stl_col: Optional dictionary containing STL decomposition components for the series.
               Can include 't_strength' (trend strength) and 's_strength' (seasonality strength).

    Returns:
      str: Formatted string containing statistical measures (mean, median, std, min, max, quartiles,
           and optional STL components) separated by newlines.
    """
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
