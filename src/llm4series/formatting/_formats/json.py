import json
import pandas as pd
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries
import json


def _to_json(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    columns = [ts.name]
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    columns = ts.columns
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  data = [
      {(ts.index.name or "index"): idx, **{col: val for col, val in zip(columns, row)}}
      for idx, row in zip([str(idx) for idx in ts.index], values)
  ]
  return json.dumps(data)


def _from_json(string: str) -> TimeSeries:
  try:
    data = json.loads(string)
    if not isinstance(data, (dict, list)):
      raise ValueError("JSON must represent a dict or list")
    df = pd.DataFrame(data)
    if df.empty:
      raise ValueError("DataFrame cannot be empty")
    return read_file(df, index_col=df.columns[0])
  except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON format: {e}")
  except (KeyError, ValueError, TypeError) as e:
    raise ValueError(f"Error parsing JSON to DataFrame: {e}")
