import os
import pandas as pd
from llm4series._internal import logger
from .ts import UniTimeSeries, MultiTimeSeries


def _parse_date_column(col: pd.Series) -> pd.Series:
  sample = col.dropna().astype(str).head(10)
  def looks_iso(val: str) -> bool:
    return len(val) >= 10 and val[4] in ("-", "/") and val[7] in ("-", "/")
  if all(looks_iso(v) for v in sample):
    return pd.to_datetime(col, errors="coerce")
  def first_component(val: str) -> int | None:
    for sep in ("-", "/", "."):
      if sep in val:
        try:
          return int(val.split(sep)[0])
        except ValueError:
          return None
    return None
  first_components = [first_component(v) for v in sample]
  first_components = [c for c in first_components if c is not None]
  dayfirst = any(c > 12 for c in first_components)
  return pd.to_datetime(col, dayfirst=dayfirst, errors="coerce")


def read_file(path_or_df: str | pd.DataFrame, index_col: str = None) -> MultiTimeSeries | UniTimeSeries:
  try:
    if isinstance(path_or_df, pd.DataFrame):
      df = path_or_df.copy()
    elif isinstance(path_or_df, str):
      _, ext = os.path.splitext(path_or_df)
      ext = ext.lower()
      readers_map = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
      }
      if ext not in readers_map:
        raise ValueError("Supported extensions: .csv, .xlsx, .json, .parquet")
      df = readers_map[ext](path_or_df)
    else:
      raise ValueError("Input must be a file path or a pandas DataFrame.")

    if index_col is None:
      if "date" not in df.columns:
        raise ValueError("No `index_col` specified and no 'date' column found.")
      index_col = "date"
    elif index_col not in df.columns:
      raise ValueError(f"The specified `index_col` '{index_col}' does not exist in the file.")

    used_index = False

    if index_col in df.columns:
      col = df[index_col]
      if pd.api.types.is_numeric_dtype(col):
        df = df.set_index(index_col)
        used_index = True
      elif pd.api.types.is_datetime64_any_dtype(col):
        df = df.set_index(index_col)
        used_index = True
      else:
        df[index_col] = _parse_date_column(col)
        df = df.set_index(index_col)
        used_index = True
      if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    try:
      if used_index and isinstance(df.index, pd.DatetimeIndex):
        freq = pd.infer_freq(df.index)
        if freq:
          df.index.freq = freq
    except Exception:
      pass
    if used_index and df.shape[1] == 1:
      ts = UniTimeSeries(df[df.columns[0]])
    else:
      ts = MultiTimeSeries(df)
    return ts

  except FileNotFoundError:
    logger.error("File not found.")
    raise FileNotFoundError("File not found.")

  except Exception:
    logger.exception("Error reading file.")
    raise Exception("An unexpected error occurred.")
