import pytest
import pandas as pd
from llm4series.data.ts import UniTimeSeries, MultiTimeSeries


class TestUniTimeSeries:

  @pytest.fixture
  def simple_series(self):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    return UniTimeSeries(data, index=dates)

  @pytest.fixture
  def series_with_duplicates(self):
    dates = pd.DatetimeIndex(
      [
        "2024-01-01",
        "2024-01-02",
        "2024-01-02",
        "2024-01-03",
        "2024-01-03",
      ]
    )
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    return UniTimeSeries(data, index=dates)

  def test_constructor_basic(self):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = [1, 2, 3, 4, 5]
    ts = UniTimeSeries(data, index=dates)
    assert len(ts) == 5
    assert isinstance(ts, UniTimeSeries)
    assert ts.iloc[0] == 1
    assert ts.iloc[-1] == 5

  def test_constructor_property(self, simple_series):
    assert simple_series._constructor == UniTimeSeries

  def test_agg_duplicates_first(self, series_with_duplicates):
    result = series_with_duplicates.agg_duplicates(method="first", inplace=False)
    assert len(result) == 3
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 2.0
    assert result.iloc[2] == 4.0

  def test_agg_duplicates_last(self, series_with_duplicates):
    result = series_with_duplicates.agg_duplicates(method="last", inplace=False)
    assert len(result) == 3
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 3.0
    assert result.iloc[2] == 5.0

  def test_agg_duplicates_sum(self, series_with_duplicates):
    result = series_with_duplicates.agg_duplicates(method="sum", inplace=False)
    assert len(result) == 3
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 5.0
    assert result.iloc[2] == 9.0

  def test_agg_duplicates_invalid_method(self, series_with_duplicates):
    with pytest.raises(ValueError, match="Invalid method"):
      series_with_duplicates.agg_duplicates(method="invalid")

  def test_agg_duplicates_inplace(self, series_with_duplicates):
    original_id = id(series_with_duplicates)
    result = series_with_duplicates.agg_duplicates(method="first", inplace=True)
    assert result is None
    assert len(series_with_duplicates) == 3

  def test_mean_rounding(self, simple_series):
    result = simple_series.mean(decimals=4)
    assert result == 5.5
    assert isinstance(result, float)

  def test_mean_default_decimals(self, simple_series):
    result = simple_series.mean()
    assert result == 5.5

  def test_median_rounding(self, simple_series):
    result = simple_series.median(decimals=4)
    assert result == 5.5

  def test_std_rounding(self, simple_series):
    result = simple_series.std(decimals=1)
    assert 3.0 <= result <= 3.1

  def test_min_rounding(self, simple_series):
    result = simple_series.min(decimals=4)
    assert result == 1.0

  def test_max_rounding(self, simple_series):
    result = simple_series.max(decimals=4)
    assert result == 10.0

  def test_quantile_rounding(self, simple_series):
    result = simple_series.quantile(q=0.25, decimals=4)
    assert 3.0 <= result <= 3.5

  def test_plot_line(self, simple_series, mocker):
    mock_lineplot = mocker.patch.object(simple_series, "lineplot")
    simple_series.plot(kind="line", title="Test")
    mock_lineplot.assert_called_once()

  def test_plot_chart(self, simple_series, mocker):
    mock_linechart = mocker.patch.object(simple_series, "linechart")
    simple_series.plot(kind="chart")
    mock_linechart.assert_called_once()

  def test_plot_bar(self, simple_series, mocker):
    mock_barplot = mocker.patch.object(simple_series, "barplot")
    simple_series.plot(kind="bar")
    mock_barplot.assert_called_once()

  def test_plot_invalid_kind(self, simple_series):
    with pytest.raises(ValueError, match="Invalid kind"):
      simple_series.plot(kind="invalid")

  def test_trend_default(self, simple_series, mocker):
    mock_stl = mocker.patch.object(
        simple_series,
        "stl",
        return_value={"trend": simple_series, "t_strength": 0.9},
    )
    result = simple_series.trend(strength=False)
    mock_stl.assert_called_once()
    assert result is simple_series

  def test_trend_with_strength(self, simple_series, mocker):
    mock_stl = mocker.patch.object(
        simple_series, "stl", return_value={"trend": simple_series, "t_strength": 0.95}
    )
    result = simple_series.trend(strength=True)
    assert result == 0.95


class TestMultiTimeSeries:
  @pytest.fixture
  def simple_dataframe(self):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = {
        "col_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "col_b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }
    return MultiTimeSeries(data, index=dates)

  @pytest.fixture
  def mixed_dataframe(self):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = {
        "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "category_col": ["A", "B", "A", "B", "A"],
    }
    return MultiTimeSeries(data, index=dates)

  @pytest.fixture
  def dataframe_with_duplicates(self):
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    data = {"col_a": [1.0, 2.0, 3.0, 4.0], "col_b": [10.0, 20.0, 30.0, 40.0]}
    return MultiTimeSeries(data, index=dates)

  def test_constructor_basic(self):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = {"col_a": [1, 2, 3, 4, 5], "col_b": [5, 4, 3, 2, 1]}
    ts = MultiTimeSeries(data, index=dates)
    assert ts.shape == (5, 2)
    assert isinstance(ts, MultiTimeSeries)

  def test_constructor_property(self, simple_dataframe):
    assert simple_dataframe._constructor == MultiTimeSeries
    assert simple_dataframe._constructor_sliced == UniTimeSeries

  def test_getitem_single_column_returns_unitimeseries(self, simple_dataframe):
    result = simple_dataframe["col_a"]
    assert isinstance(result, UniTimeSeries)
    assert len(result) == 10

  def test_getitem_multiple_columns_returns_multitimeseries(self, simple_dataframe):
    result = simple_dataframe[["col_a", "col_b"]]
    assert isinstance(result, MultiTimeSeries)
    assert result.shape == (10, 2)

  def test_num_columns_property(self, mixed_dataframe):
    result = mixed_dataframe.num_columns
    assert len(result) == 1
    assert "numeric_col" in result

  def test_cat_columns_property(self, mixed_dataframe):
    result = mixed_dataframe.cat_columns
    assert len(result) == 1
    assert "category_col" in result

  def test_agg_duplicates_first(self, dataframe_with_duplicates):
    result = dataframe_with_duplicates.agg_duplicates(method="first", inplace=False)
    assert len(result) == 3
    assert result.loc["2024-01-02", "col_a"] == 2.0

  def test_agg_duplicates_last(self, dataframe_with_duplicates):
    result = dataframe_with_duplicates.agg_duplicates(method="last", inplace=False)
    assert len(result) == 3
    assert result.loc["2024-01-02", "col_a"] == 3.0

  def test_agg_duplicates_invalid_method(self, dataframe_with_duplicates):
    with pytest.raises(ValueError, match="Invalid method"):
      dataframe_with_duplicates.agg_duplicates(method="invalid")

  def test_mean_rounding(self, simple_dataframe):
    result = simple_dataframe.mean(decimals=4)
    assert isinstance(result, pd.Series)
    assert result["col_a"] == 5.5
    assert result["col_b"] == 5.5

  def test_median_rounding(self, simple_dataframe):
    result = simple_dataframe.median(decimals=4)
    assert isinstance(result, pd.Series)
    assert result["col_a"] == 5.5

  def test_std_rounding(self, simple_dataframe):
    result = simple_dataframe.std(decimals=1)
    assert isinstance(result, pd.Series)
    assert len(result) == 2

  def test_min_rounding(self, simple_dataframe):
    result = simple_dataframe.min(decimals=4)
    assert result["col_a"] == 1.0
    assert result["col_b"] == 1.0

  def test_max_rounding(self, simple_dataframe):
    result = simple_dataframe.max(decimals=4)
    assert result["col_a"] == 10.0
    assert result["col_b"] == 10.0

  def test_quantile_rounding(self, simple_dataframe):
    result = simple_dataframe.quantile(q=0.5, decimals=4)
    assert isinstance(result, pd.Series)
    assert result["col_a"] == 5.5

  def test_plot_line(self, simple_dataframe, mocker):
    mock_lineplot = mocker.patch.object(simple_dataframe, "lineplot")
    simple_dataframe.plot(kind="line")
    mock_lineplot.assert_called_once()

  def test_plot_invalid_kind(self, simple_dataframe):
    with pytest.raises(ValueError, match="Invalid kind"):
      simple_dataframe.plot(kind="invalid")

  def test_trend_default(self, simple_dataframe, mocker):
    mock_stl = mocker.patch.object(
        simple_dataframe,
        "stl",
        return_value={"trend": simple_dataframe, "t_strength": simple_dataframe.mean()},
    )
    result = simple_dataframe.trend(strength=False)
    mock_stl.assert_called_once()

  def test_trend_with_strength(self, simple_dataframe, mocker):
    strength_result = pd.Series({"col_a": 0.9, "col_b": 0.85})
    mock_stl = mocker.patch.object(
        simple_dataframe,
        "stl",
        return_value={"trend": simple_dataframe, "t_strength": strength_result},
    )
    result = simple_dataframe.trend(strength=True)
    assert isinstance(result, pd.Series)
