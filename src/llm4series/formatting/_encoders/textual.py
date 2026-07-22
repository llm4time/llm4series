from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries
from ..._internal import logger
import pandas as pd
import re


def _encode_textual(ts: TimeSeries) -> TimeSeries:
  ts = ts.copy()
  def encode(v):
    if pd.isna(v):
      return v
    return ' '.join(str(v))
  if isinstance(ts, UniTimeSeries):
    values = pd.Series(ts.astype(object)).map(encode)
    ts[:] = values.to_numpy()
  elif isinstance(ts, MultiTimeSeries):
    for col in ts.num_columns:
      ts[col] = pd.Series(ts[col].astype(object)).map(encode)
  else:
    logger.error(f"Unsupported time series type: {type(ts).__name__}.")
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  return ts


def _decode_textual(ts: TimeSeries) -> TimeSeries:
  ts = ts.copy()
  def decode(v):
    s = str(v).strip()
    if re.fullmatch(r"[-\d\s.]+", s):
      return float(s.replace(" ", ""))
    return v
  if isinstance(ts, UniTimeSeries):
    values = pd.Series(ts).map(decode)
    ts[:] = values.to_numpy()
  elif isinstance(ts, MultiTimeSeries):
    for col in ts.columns:
      ts[col] = pd.Series(ts[col]).map(decode)
  else:
    logger.error(f"Unsupported time series type: {type(ts).__name__}.")
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  return ts
