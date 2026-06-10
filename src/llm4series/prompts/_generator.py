from ._templates import *
from dataclasses import dataclass
from typing import get_args, Literal, Type
from pydantic import BaseModel, Field, create_model, conlist
from datetime import datetime
from .._internal import logger
import llm4series.data as ls
import pandas as pd


PromptType = Literal["zero_shot", "few_shot", "cot", "cot_few", "custom"]

def make_response_format(columns: list[str], forecast_horizon: int) -> Type[BaseModel]:
    D = conlist(datetime, min_length=forecast_horizon, max_length=forecast_horizon)
    V = conlist(float, min_length=forecast_horizon, max_length=forecast_horizon)
    return create_model("ForecastRow",
      date=(D, Field(..., description=f"List of {forecast_horizon} dates corresponding to the forecast horizon")),
      **{col: (V, Field(..., description=f"List of {forecast_horizon} forecasted values for column '{col}'"))
        for col in columns})

@dataclass(kw_only=True)
class PromptConfig:
  system: str
  text: str
  data: str
  response_format: Type[BaseModel]


def prompt(
    type: PromptType,
    ts: ls.TimeSeries,
    forecast_horizon: int,
    tsformat: ls.TSFormat = "csv",
    tstype: ls.TSType = "numeric",
    sampling: ls.Sampling = "uniform",
    examples: int = 0,
    template: str = None,
    stl: dict = None,
    decimals: int = 3,
    **kwargs
) -> str:
  prompt_map = {
      "zero_shot": ZERO_SHOT,
      "few_shot": FEW_SHOT,
      "cot_few": COT_FEW,
      "cot": COT,
      "custom": template
  }
  if type not in prompt_map:
    logger.error(f"Invalid prompt type: {type}.")
    raise ValueError("Supported prompts: zero_shot, few_shot, cot, cot_few, custom.")

  if forecast_horizon <= 0:
    logger.error(f"Invalid forecast horizon: {forecast_horizon}. Must be a positive integer.")
    raise ValueError("Forecast horizon must be a positive integer.")

  if tsformat not in get_args(ls.TSFormat):
    logger.error(f"Invalid time series format: {tsformat}.")
    raise ValueError(f"Supported time series formats: {', '.join(get_args(ls.TSFormat))}.")

  if tstype not in get_args(ls.TSType):
    logger.error(f"Invalid time series type: {tstype}.")
    raise ValueError(f"Supported time series types: {', '.join(get_args(ls.TSType))}.")

  if sampling not in get_args(ls.Sampling):
    logger.error(f"Invalid sampling method: {sampling}.")
    raise ValueError(f"Supported samplings: {', '.join(get_args(ls.Sampling))}.")

  if template is None and type == "custom":
    logger.error("Template must be provided for custom prompt type.")
    raise ValueError("Template must be set for custom prompt.")

  if examples == 0 and type in ["few_shot", "cot_few"]:
    logger.error("Must contain at least 1 example.")
    raise ValueError("Must contain at least 1 example.")

  min_periods = forecast_horizon * 2 * examples
  if len(ts) < min_periods:
    logger.error(f"Not enough data points: {len(ts)}. Required at least {min_periods} for {examples} examples and forecast horizon of {forecast_horizon}.")
    raise ValueError(f"For {examples} examples there must be {min_periods} periods in the time series.")

  if decimals <= 0:
    logger.error(f"Invalid decimals: {decimals}. Must be a non-negative integer.")
    raise ValueError("Decimals must be a non-negative integer.")

  ts = ts.round(decimals)

  base_kwargs = {
      "input_len": len(ts),
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
    logger.error(f"Unsupported time series type: {type(ts).__name__}.")
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  if "forecast_examples" not in kwargs:
    forecast_examples = "\n".join([
        f"- Example {i}:\n"
        f"Input (history):\n{input.to_str(tsformat, tstype)}\n\n"
        f"Output (forecast):\n{output.to_str(tsformat, tstype)}"
        f"{'' if i == examples else '\n'}"
        for i, (input, output) in enumerate(
            ts.slide(method=sampling, window=forecast_horizon, samples=examples),
            start=1)
    ])
    base_kwargs.update({"forecast_examples": forecast_examples})

  try:
    system = SYSTEM.format(**base_kwargs)
    text = prompt_map[type].format(**base_kwargs)
    data = ts.to_str(tsformat, tstype)
    columns = [ts.name or "value"] if isinstance(ts, ls.UniTimeSeries) else ts.num_columns
    response_format = make_response_format(columns, forecast_horizon)
    return PromptConfig(system=system, text=text, data=data, response_format=response_format)
  except KeyError as e:
    logger.error(f"Missing key in template: {e}.")
    raise ValueError(f"Key {e} not defined.")
