import pytest
import pandas as pd
from llm4series.data.ts import UniTimeSeries


class TestTimeSeriesNormalize:

    @pytest.fixture
    def time_series_with_gaps(self):
        dates = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-05"])
        data = [1.0, 2.0, 3.0]
        return UniTimeSeries(data, index=dates)

    @pytest.fixture
    def regular_time_series(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        return UniTimeSeries(data, index=dates)

    def test_normalize_with_explicit_freq(self, time_series_with_gaps):
        result = time_series_with_gaps.normalize(freq="D")
        assert len(result) == 5
        assert result.iloc[0] == 1.0
        assert result.iloc[2] == 2.0
        assert result.iloc[4] == 3.0

    def test_normalize_with_inferred_freq(self, regular_time_series):
        result = regular_time_series.normalize(freq=None)
        assert len(result) == 5

    def test_normalize_custom_start_end(self, time_series_with_gaps):
        result = time_series_with_gaps.normalize(
            freq="D",
            start="2024-01-01",
            end="2024-01-07"
        )
        assert result.index[0] == pd.Timestamp("2024-01-01")
        assert result.index[-1] == pd.Timestamp("2024-01-07")

    def test_normalize_raises_error_without_freq(self, time_series_with_gaps):
        with pytest.raises(ValueError, match="infer frequency"):
            time_series_with_gaps.normalize(freq=None)


class TestTimeSeriesSplit:

  @pytest.fixture
  def time_series_for_split(self):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = list(range(1, 11))
    return UniTimeSeries(data, index=dates)

  def test_split_basic(self, time_series_for_split):
    train, _ = time_series_for_split.split(
      start="2024-01-01",
      end="2024-01-06",
    )
    assert len(train) == 6
    assert train.iloc[0] == 1.0
    assert train.iloc[-1] == 6.0

  def test_split_test_size(self, time_series_for_split):
    _, test = time_series_for_split.split(test_size=0.3)
    assert len(test) == 3

  def test_split_test_size_with_periods(self, time_series_for_split):
    _, test = time_series_for_split.split(test_size=0.3, periods=2)
    assert len(test) == 2

  def test_split_periods(self, time_series_for_split):
    _, test = time_series_for_split.split(
      start="2024-01-01",
      end="2024-01-06",
      periods=3,
    )
    assert len(test) == 3

  def test_split_periods_none(self, time_series_for_split):
    _, test = time_series_for_split.split(
      start="2024-01-01",
      end="2024-01-06",
    )
    assert len(test) == 4

  def test_split_with_datetime_index(self, time_series_for_split):
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-05")
    train, _ = time_series_for_split.split(
      start=start_date,
      end=end_date,
    )
    assert len(train) == 5

  def test_split_invalid_test_size(self, time_series_for_split):
    with pytest.raises(ValueError):
      time_series_for_split.split(test_size=1.5)

  def test_split_preserves_order(self, time_series_for_split):
    train, test = time_series_for_split.split(test_size=0.3)
    assert train.iloc[-1] < test.iloc[0]


class TestTimeSeriesSlide:

  @pytest.fixture
  def longer_series(self):
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    data = list(range(1, 21))
    return UniTimeSeries(data, index=dates)

  def test_slide_frontend(self, longer_series):
    windows = longer_series.slide(
        method="frontend",
        window=3,
        samples=2
    )
    assert len(windows) == 2
    assert len(windows[0][0]) == 3
    assert len(windows[0][1]) == 3

  def test_slide_backend(self, longer_series):
    windows = longer_series.slide(
        method="backend",
        window=3,
        samples=2
    )
    assert len(windows) <= 2

  def test_slide_random(self, longer_series):
    windows = longer_series.slide(
        method="random",
        window=3,
        samples=3
    )
    assert len(windows) <= 3

  def test_slide_uniform(self, longer_series):
    windows = longer_series.slide(
        method="uniform",
        window=3,
        samples=3
    )
    assert len(windows) <= 3

  def test_slide_invalid_method(self, longer_series):
    with pytest.raises(ValueError, match="Supported methods"):
        longer_series.slide(
            method="invalid",
            window=3,
            samples=2
        )

  def test_slide_insufficient_data(self):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = [1, 2, 3, 4, 5]
    series = UniTimeSeries(data, index=dates)
    windows = series.slide(
        method="frontend",
        window=10,
        samples=2
    )
    assert len(windows) == 0

  def test_slide_with_step(self, longer_series):
    windows = longer_series.slide(
        method="uniform",
        window=2,
        samples=5,
        step=2
    )
    assert len(windows) > 0


class TestTimeSeriesConvert:

  @pytest.fixture
  def simple_series(self):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = [1.0, 2.0, 3.0]
    return UniTimeSeries(data, index=dates)

  def test_to_str_method_exists(self, simple_series):
    assert hasattr(simple_series, 'to_str')
    assert callable(simple_series.to_str)

  def test_to_str_textual_plain_format(self, simple_series):
    result = simple_series.to_str("plain", "textual")
    assert isinstance(result, str)
    assert "1.0" not in result
    assert "1 0" in result
