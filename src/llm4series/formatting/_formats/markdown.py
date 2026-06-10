import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries
from ..._internal import logger


def _to_markdown(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"|{(ts.index.name or "index")}|{ts.name}|"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = "|" + (ts.index.name or "index") + "|" + "|".join(ts.columns) + "|"
    values = ts.to_numpy().tolist()
  else:
    logger.error(f"Unsupported time series type: {type(ts).__name__}.")
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
  sep = "|" + "|".join("---" for _ in header.split("|") if _ != "") + "|"
  rows = [f"|{idx}|" + "|".join(str(v) for v in row) +
          "|" for idx, row in zip(ts.index, values)]
  return header + "\n" + sep + "\n" + "\n".join(rows)


def _from_markdown(string: str) -> TimeSeries:
  try:
    if not string or not string.strip():
      logger.error("Input string cannot be empty.")
      raise ValueError("Input string cannot be empty.")
    lines = string.strip().splitlines()
    if len(lines) < 3:
      logger.error("Markdown table must have at least 3 lines (header, separator, data).")
      raise ValueError("Markdown table must have at least 3 lines (header, separator, data).")
    data = "\n".join([line.strip().strip("|") for line in [lines[0]] + lines[2:]])
    df = pd.read_csv(StringIO(data), sep="|", engine="python", skipinitialspace=True)
    if df.empty:
      logger.error("DataFrame cannot be empty.")
      raise ValueError("DataFrame cannot be empty.")
    return read_file(df, index_col=df.columns[0])
  except (ValueError, IndexError) as e:
    logger.error(f"Error parsing markdown format: {e}.")
    raise ValueError(f"Error parsing markdown format: {e}.")
  except Exception as e:
    logger.error(f"Unexpected error parsing markdown format: {e}.")
    raise ValueError(f"Unexpected error parsing markdown format: {e}.")
