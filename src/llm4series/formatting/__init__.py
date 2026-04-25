from ._formats import (
  _to_array,
  _from_array,
  _to_csv,
  _from_csv,
  _to_tsv,
  _from_tsv,
  _to_json,
  _from_json,
  _to_markdown,
  _from_markdown,
  _to_plain,
  _from_plain,
  _to_context,
  _from_context,
  _to_custom,
  _from_custom,
  _to_symbol,
  _from_symbol,
  _to_toon,
  _from_toon,
)
from ._encoders import _decode_textual, _encode_textual
from ..data import TimeSeries, TSFormat


def encode_textual(ts: TimeSeries) -> TimeSeries:
  """Encode a TimeSeries to textual format.
  
  Converts a TimeSeries object to a textual representation suitable for
  LLM processing.
  
  Args:
      ts (TimeSeries): The time series to encode.
  
  Returns:
      TimeSeries: Encoded time series in textual format.
  """
  return _encode_textual(ts)


def from_str(string: str, format: TSFormat) -> TimeSeries:
  """Parse a TimeSeries from a string in the specified format.
  
  Converts a string representation to a TimeSeries object. Supports multiple
  formats including CSV, JSON, Markdown, plain text, and domain-specific formats.
  
  Args:
      string (str): String representation of the time series.
      format (TSFormat): Format of the input string ('array', 'csv', 'json', 'markdown',
                         'plain', 'context', 'custom', 'symbol', 'toon', 'tsv').
  
  Returns:
      TimeSeries: Parsed time series object (UniTimeSeries or MultiTimeSeries).
  
  Raises:
      ValueError: If format is unknown or string cannot be parsed in that format.
  """
  formats_map = {
    "array": _from_array,
    "context": _from_context,
    "csv": _from_csv,
    "custom": _from_custom,
    "json": _from_json,
    "markdown": _from_markdown,
    "plain": _from_plain,
    "symbol": _from_symbol,
    "toon": _from_toon,
    "tsv": _from_tsv,
  }
  if format not in formats_map:
    raise ValueError(f"Unknown format: {format}.")
  try:
    return _decode_textual(formats_map[format](string))
  except Exception:
    raise ValueError(f"Failed to parse TimeSeries from format {format}.")

__all__ = ['from_str', 'encode_textual']
