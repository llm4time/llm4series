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
  return _encode_textual(ts)


def from_str(string: str, format: TSFormat) -> TimeSeries:
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
