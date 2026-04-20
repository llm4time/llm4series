import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries


def _to_csv(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"{(ts.index.name or "index")},{ts.name}"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = f"{(ts.index.name or "index")}," + ",".join(ts.columns)
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  lines = [f"{idx}," + ",".join(str(v) for v in row)
           for idx, row in zip(ts.index, values)]
  return header + "\n" + "\n".join(lines)


def _from_csv(string: str) -> TimeSeries:
  try:
    if not string or not string.strip():
      raise ValueError("Input string cannot be empty")
    df = pd.read_csv(StringIO(string))
    if df.empty:
      raise ValueError("DataFrame cannot be empty")
    return read_file(df, index_col=df.columns[0])
  except pd.errors.ParserError as e:
    raise ValueError(f"Error parsing CSV format: {e}")
  except Exception as e:
    raise ValueError(f"Unexpected error parsing CSV format: {e}")
