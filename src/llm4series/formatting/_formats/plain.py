import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries


def _to_plain(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    values = [[v] for v in ts.to_list()]
    columns = [ts.name]
  elif isinstance(ts, MultiTimeSeries):
    values = ts.to_numpy().tolist()
    columns = ts.columns
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  lines = [
      f"{(ts.index.name or "index")}: {idx}, " +
      ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
      for idx, row in zip(ts.index, values)
  ]
  return "\n".join(lines)


def _from_plain(string: str) -> TimeSeries:
  try:
    if not string or not string.strip():
      raise ValueError("Input string cannot be empty")
    data = [{k.strip(): v.strip() for k, v in (p.split(":", 1) for p in line.split(","))}
            for line in string.strip().splitlines()]
    if not data:
      raise ValueError("No valid data parsed from input")
    df = pd.read_csv(StringIO(pd.DataFrame(data).to_csv(index=False)))
    if df.empty:
      raise ValueError("DataFrame cannot be empty")
    return read_file(df, index_col=df.columns[0])
  except (ValueError, KeyError, IndexError) as e:
    raise ValueError(f"Error parsing plain format: {e}")
  except Exception as e:
    raise ValueError(f"Unexpected error parsing plain format: {e}")
