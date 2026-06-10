import json
import pandas as pd
from ...data import TimeSeries, read_file
from ...data import UniTimeSeries, MultiTimeSeries
from ..._internal import logger
import json


def _to_json(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    columns = [ts.name]
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    columns = ts.columns
    values = ts.to_numpy().tolist()
  else:
    logger.error(f"Unsupported time series type: {type(ts).__name__}.")
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
      logger.error("JSON must represent a dict or list.")
      raise ValueError("JSON must represent a dict or list.")
    df = pd.DataFrame(data)
    if df.empty:
      logger.error("DataFrame cannot be empty.")
      raise ValueError("DataFrame cannot be empty.")
    return read_file(df, index_col=df.columns[0])
  except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON format: {e}.")
    raise ValueError(f"Invalid JSON format: {e}.")
  except (KeyError, ValueError, TypeError) as e:
    logger.error(f"Error parsing JSON to DataFrame: {e}.")
    raise ValueError(f"Error parsing JSON to DataFrame: {e}.")
