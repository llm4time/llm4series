import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries
import re


def _to_context(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"{(ts.index.name or "index")},{ts.name}"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = f"{(ts.index.name or "index")},{",".join(ts.columns)}"
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  lines = [f"{idx}," + ",".join(f"[{v}]" for v in row)
           for idx, row in zip(ts.index, values)]
  return header + "\n" + "\n".join(lines)


def _from_context(string: str) -> TimeSeries:
  string = re.sub(r'\[([^\]]+)\]', r'\1', string)
  df = pd.read_csv(StringIO(string))
  return read_file(df, index_col=df.columns[0])
