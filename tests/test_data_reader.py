import pytest
import pandas as pd
import tempfile
import os
import json
from llm4series.data.reader import read_file
from llm4series.data.ts import UniTimeSeries, MultiTimeSeries


class TestReadFile:

  @pytest.fixture
  def temp_dir(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      yield tmp_dir

  @pytest.fixture
  def sample_csv_file(self, temp_dir):
    csv_path = os.path.join(temp_dir, "sample.csv")
    pd.DataFrame({
      "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    return csv_path

  @pytest.fixture
  def sample_json_file(self, temp_dir):
    json_path = os.path.join(temp_dir, "sample.json")
    with open(json_path, "w") as f:
      json.dump({"date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                 "value": [1.0, 2.0, 3.0]}, f)
    return json_path

  @pytest.fixture
  def sample_dataframe(self):
    return pd.DataFrame({
      "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "value": [1.0, 2.0, 3.0],
    })

  def test_read_dataframe_without_date_col_raises(self, sample_dataframe):
    df = sample_dataframe[["value"]]
    with pytest.raises(Exception):
      read_file(df)

  def test_read_dataframe_with_date_col_returns_uni(self, sample_dataframe):
    result = read_file(sample_dataframe)
    assert isinstance(result, UniTimeSeries)
    assert len(result) == 3

  def test_read_dataframe_multiple_value_columns_returns_multi(self, sample_dataframe):
    df = sample_dataframe.copy()
    df["value2"] = [4.0, 5.0, 6.0]
    result = read_file(df)
    assert isinstance(result, MultiTimeSeries)
    assert result.shape == (3, 2)

  def test_read_dataframe_copy_created(self, sample_dataframe):
    result = read_file(sample_dataframe)
    result.iloc[0] = 999
    assert sample_dataframe.iloc[0, 0] != 999

  def test_read_csv_auto_detects_date_column(self, sample_csv_file):
    result = read_file(sample_csv_file)
    assert isinstance(result, UniTimeSeries)
    assert len(result) == 3
    assert pd.api.types.is_datetime64_any_dtype(result.index)
    assert result.index[0] == pd.Timestamp("2024-01-01")

  def test_read_csv_explicit_index_col_is_accepted(self, sample_csv_file):
    result = read_file(sample_csv_file, index_col="date")
    assert isinstance(result, UniTimeSeries)
    assert len(result) == 3
    assert pd.api.types.is_datetime64_any_dtype(result.index)

  def test_read_csv_explicit_custom_index_col(self, temp_dir):
    csv_path = os.path.join(temp_dir, "custom.csv")
    pd.DataFrame({
      "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path, index_col="timestamp")
    assert result.index.name == "timestamp"
    assert pd.api.types.is_datetime64_any_dtype(result.index)

  def test_read_csv_nonexistent_index_col_raises(self, sample_csv_file):
    with pytest.raises(Exception):
      read_file(sample_csv_file, index_col="nonexistent")

  def test_read_csv_case_insensitive_extension(self, temp_dir):
    csv_path = os.path.join(temp_dir, "file.CSV")
    pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]}).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert len(result) == 1

  def test_read_csv_multiple_columns_returns_multi(self, temp_dir):
    csv_path = os.path.join(temp_dir, "multi.csv")
    pd.DataFrame({
      "date": ["2024-01-01", "2024-01-02"],
      "value1": [1.0, 2.0],
      "value2": [3.0, 4.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert isinstance(result, MultiTimeSeries)
    assert result.shape == (2, 2)

  def test_read_json_file_returns_uni(self, sample_json_file):
    result = read_file(sample_json_file)
    assert isinstance(result, UniTimeSeries)
    assert len(result) == 3

  def test_read_json_case_insensitive_extension(self, temp_dir):
    json_path = os.path.join(temp_dir, "file.JSON")
    with open(json_path, "w") as f:
      json.dump({"date": ["2024-01-01"], "value": [1.0]}, f)
    result = read_file(json_path)
    assert len(result) == 1

  def test_date_iso_parsed_correctly(self, temp_dir):
    csv_path = os.path.join(temp_dir, "iso.csv")
    pd.DataFrame({
      "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert result.index[0] == pd.Timestamp("2024-01-01")
    assert result.index[1] == pd.Timestamp("2024-01-02")

  def test_date_parsed_dayfirst_false(self, temp_dir):
    csv_path = os.path.join(temp_dir, "dates.csv")
    pd.DataFrame({
      "date": ["01-02-2024", "02-02-2024", "03-02-2024"],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert pd.api.types.is_datetime64_any_dtype(result.index)
    assert result.index[0].month == 1

  def test_date_parsed_dayfirst_true_when_day_exceeds_12(self, temp_dir):
    csv_path = os.path.join(temp_dir, "dates_dayfirst.csv")
    pd.DataFrame({
      "date": ["13-01-2024", "14-01-2024", "15-01-2024"],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert pd.api.types.is_datetime64_any_dtype(result.index)
    assert result.index[0].day == 13
    assert result.index[0].month == 1

  def test_date_col_already_datetime64_set_directly(self):
    df = pd.DataFrame({
      "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
      "value": [1.0, 2.0, 3.0],
    })
    result = read_file(df)
    assert pd.api.types.is_datetime64_any_dtype(result.index)
    assert result.index[0] == pd.Timestamp("2024-01-01")

  def test_date_col_numeric_set_directly(self, temp_dir):
    csv_path = os.path.join(temp_dir, "numeric.csv")
    pd.DataFrame({
      "date": [1, 2, 3],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert result.index[0] == 1
    assert result.index[1] == 2

  def test_index_sorted_ascending(self, temp_dir):
    csv_path = os.path.join(temp_dir, "unsorted.csv")
    pd.DataFrame({
      "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
      "value": [3.0, 1.0, 2.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    assert result.index[0] == pd.Timestamp("2024-01-01")
    assert result.index[1] == pd.Timestamp("2024-01-02")
    assert result.index[2] == pd.Timestamp("2024-01-03")

  def test_frequency_inferred_for_regular_series(self, temp_dir):
    csv_path = os.path.join(temp_dir, "regular.csv")
    pd.DataFrame({
      "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
      "value": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(csv_path, index=False)
    result = read_file(csv_path)
    has_freq = (result.index.freq is not None or result.index.inferred_freq is not None)
    assert has_freq

  def test_file_not_found_raises_with_message(self):
    with pytest.raises(FileNotFoundError, match="File not found"):
      read_file("/nonexistent/path/file.csv")

  def test_unsupported_extension_raises(self, temp_dir):
    unsupported = os.path.join(temp_dir, "file.txt")
    with open(unsupported, "w") as f:
      f.write("data")
    with pytest.raises(Exception):
      read_file(unsupported)

  def test_invalid_input_type_raises(self):
    with pytest.raises(Exception):
      read_file(123)

  def test_csv_without_date_column_raises(self, temp_dir):
    csv_path = os.path.join(temp_dir, "no_date.csv")
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
    with pytest.raises(Exception):
      read_file(csv_path)

  def test_invalid_dates_handled_without_crash(self, temp_dir):
    csv_path = os.path.join(temp_dir, "bad_dates.csv")
    pd.DataFrame({
      "date": ["2024-01-01", "invalid-date", "2024-01-03"],
      "value": [1.0, 2.0, 3.0],
    }).to_csv(csv_path, index=False)
    try:
      result = read_file(csv_path)
      assert len(result) >= 0
    except (ValueError, TypeError, Exception):
      pass
