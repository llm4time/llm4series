import ast
import pandas as pd
from ...data import TimeSeries
from ...data import UniTimeSeries, MultiTimeSeries
from ..._internal import logger


class _SpecialFloatTransformer(ast.NodeTransformer):
  _values = {
      "nan": float("nan"),
      "inf": float("inf"),
      "infinity": float("inf"),
  }

  def visit_Name(self, node):
    value = self._values.get(node.id.lower())
    if value is not None:
      return ast.copy_location(ast.Constant(value=value), node)
    return node


def _parse_array_literal(string: str):
  tree = ast.parse(string, mode="eval")
  tree = _SpecialFloatTransformer().visit(tree)
  ast.fix_missing_locations(tree)
  return ast.literal_eval(tree)


def _to_array(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries) or isinstance(ts, MultiTimeSeries):
    return str(ts.values.tolist() if not ts.empty else [])
  else:
    logger.error(f"Unsupported time series type: {type(ts).__name__}")
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

def _from_array(string: str) -> TimeSeries:
  try:
    data = _parse_array_literal(string) or []
    if not isinstance(data, list):
      logger.error(f"Input must be a list or array, got {type(data).__name__}.")
      raise ValueError("Input must be a list or array.")
    if not data:
      logger.error("Array cannot be empty")
      raise ValueError("Array cannot be empty")
    index = pd.RangeIndex(start=0, stop=len(data), step=1)
    first_item = data[0]
    if isinstance(first_item, (list, tuple)):
      if not all(isinstance(item, (list, tuple)) for item in data):
        logger.error("Array rows must all be lists or tuples.")
        raise ValueError("Array rows must all be lists or tuples.")
      row_lengths = {len(item) for item in data}
      if len(row_lengths) != 1:
        logger.error("Array rows must all have the same length.")
        raise ValueError("Array rows must all have the same length.")
      return MultiTimeSeries(pd.DataFrame(data, index=index))
    if any(isinstance(item, (list, tuple)) for item in data):
      logger.error("Array values must all be scalars or all be row sequences.")
      raise ValueError("Array values must all be scalars or all be row sequences.")
    return UniTimeSeries(data, index=index, name="value")
  except (ValueError, SyntaxError) as e:
    logger.error(f"Invalid array format: {e}.")
    raise ValueError(f"Invalid array format: {e}.")
  except Exception as e:
    logger.error(f"Error parsing array to DataFrame: {e}.")
    raise ValueError(f"Error parsing array to DataFrame: {e}.")
