import pandas as pd
from toon import decode
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries
import toon


def _to_toon(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    data = [
        {(ts.index.name or "index"): str(idx), ts.name: val}
        for idx, val in zip(ts.index, ts.to_list())
    ]
  elif isinstance(ts, MultiTimeSeries):
    data = [
        {(ts.index.name or "index"): str(idx), **{col: val for col, val in zip(ts.columns, row)}}
        for idx, row in zip(ts.index, ts.to_numpy().tolist())
    ]
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  return toon.encode(data)


def _from_toon(string: str) -> TimeSeries:
  data = decode(string)
  df = pd.DataFrame(data)
  return read_file(df, index_col=df.columns[0])
