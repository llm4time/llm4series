import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries


def _to_custom(ts: TimeSeries, sep: str = "|") -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"{(ts.index.name or "index")}{sep}{ts.name}"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = f"{(ts.index.name or "index")}{sep}" + sep.join(ts.columns)
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  lines = [f"{idx}{sep}" + sep.join(str(v) for v in row)
           for idx, row in zip(ts.index, values)]
  return header + "\n" + "\n".join(lines)


def _from_custom(string: str) -> TimeSeries:
  df = pd.read_csv(StringIO(string), sep="|")
  return read_file(df, index_col=df.columns[0])
