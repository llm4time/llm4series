import ast
import pandas as pd
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries


def _to_array(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries) or isinstance(ts, MultiTimeSeries):
    return str(ts.values.tolist() if not ts.empty else [])
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

def _from_array(string: str) -> TimeSeries:
  try:
    data = ast.literal_eval(string) or []
    if not isinstance(data, list):
      raise ValueError("Input must be a list or array")
    df = pd.DataFrame(data)
    if df.empty:
      raise ValueError("Array cannot be empty")
    return read_file(df)
  except (ValueError, SyntaxError) as e:
    raise ValueError(f"Invalid array format: {e}")
  except Exception as e:
    raise ValueError(f"Error parsing array to DataFrame: {e}")
