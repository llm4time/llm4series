from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries
import pandas as pd
import re


def _encode_textual(ts: TimeSeries) -> TimeSeries:
  ts = ts.copy()
  def encode(v):
    if pd.isna(v):
      return v
    return ' '.join(str(v))
  if isinstance(ts, UniTimeSeries):
    ts = ts.astype(object)
    ts[:] = ts.apply(encode)
  elif isinstance(ts, MultiTimeSeries):
    for col in ts.num_columns:
      ts[col] = ts[col].astype(object)
      ts[col] = ts[col].apply(encode)
  else:
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
    ts = ts.apply(decode)
  elif isinstance(ts, MultiTimeSeries):
    for col in ts.columns:
      ts[col] = ts[col].apply(decode)
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  return ts
