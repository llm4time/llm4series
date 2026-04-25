from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries
import pandas as pd
import re


def _encode_textual(ts: TimeSeries) -> TimeSeries:
  """Encode TimeSeries values to textual format with spaced characters.
  
  Converts numeric values to strings with spaces between each character,
  useful for LLM processing. Preserves NaN values. Works with both univariate
  and multivariate time series.
  
  Args:
      ts (TimeSeries): UniTimeSeries or MultiTimeSeries to encode.
  
  Returns:
      TimeSeries: Time series with values converted to spaced text format.
  
  Raises:
      TypeError: If input is not a TimeSeries, UniTimeSeries, or MultiTimeSeries.
  """
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
  """Decode textual TimeSeries back to numeric format when possible.
  
  Converts spaced text values back to numeric format using regex pattern matching.
  Matches numeric patterns (digits, dots, hyphens, spaces) and converts them to floats.
  Non-numeric values are preserved as-is. Works with both univariate and multivariate
  time series.
  
  Args:
      ts (TimeSeries): UniTimeSeries or MultiTimeSeries to decode.
  
  Returns:
      TimeSeries: Time series with textual values converted back to numeric format where possible.
  
  Raises:
      TypeError: If input is not a TimeSeries, UniTimeSeries, or MultiTimeSeries.
  """
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
