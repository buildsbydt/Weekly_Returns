"""
Unit tests for weekly_returns.py
=================================
Run with:  python3 -m pytest test_weekly_returns.py -v
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from weekly_returns import (
    get_weekly_stock_data,
    calculate_weekly_returns,
    add_week_start_dates,
    build_weekly_comparison_table,
    build_single_ticker_table,
    get_week_summary,
    _exclude_incomplete_current_week,
    _normalise_ticker,
    _format_pct,
)


# ===================================================================
# Helpers: build realistic test DataFrames
# ===================================================================

def _make_weekly_df(prices, start_date="2026-01-05", freq="7D"):
    """Create a DataFrame that mimics yfinance weekly output."""
    dates = pd.date_range(start=start_date, periods=len(prices), freq=freq)
    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"
    return df


def _make_full_pipeline_df(prices, start_date="2026-01-05"):
    """Create a DataFrame that has been through calculate + add_week_start."""
    df = _make_weekly_df(prices, start_date)
    df = calculate_weekly_returns(df)
    df = add_week_start_dates(df)
    return df


# ===================================================================
# Step 1 - get_weekly_stock_data validation
# ===================================================================

class TestGetWeeklyStockDataValidation:
    def test_ticker_not_string(self):
        assert get_weekly_stock_data(123, 20) == "Ticker must be a string."

    def test_ticker_empty(self):
        assert get_weekly_stock_data("", 20) == "Ticker cannot be empty."

    def test_ticker_whitespace_only(self):
        assert get_weekly_stock_data("   ", 20) == "Ticker cannot be empty."

    def test_weeks_not_integer(self):
        assert get_weekly_stock_data("AAPL", 20.5) == "Weeks must be an integer."

    def test_weeks_not_integer_string(self):
        assert get_weekly_stock_data("AAPL", "20") == "Weeks must be an integer."

    def test_weeks_too_low(self):
        assert get_weekly_stock_data("AAPL", 0) == "Weeks must be between 1 and 520."

    def test_weeks_too_high(self):
        assert get_weekly_stock_data("AAPL", 521) == "Weeks must be between 1 and 520."

    def test_weeks_boundary_low(self):
        # weeks=1 should NOT return a validation error (it's valid)
        result = get_weekly_stock_data("AAPL", 1)
        assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got: {result}"

    def test_weeks_boundary_high(self):
        # weeks=520 should NOT return a validation error
        result = get_weekly_stock_data("AAPL", 520)
        assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got: {result}"


# ===================================================================
# Step 2 - calculate_weekly_returns
# ===================================================================

class TestCalculateWeeklyReturns:
    def test_not_dataframe(self):
        assert calculate_weekly_returns("not a df") == "Weekly data must be a DataFrame."

    def test_missing_close_column(self):
        df = pd.DataFrame({"Open": [100, 110]}, index=pd.date_range("2026-01-05", periods=2, freq="7D"))
        assert calculate_weekly_returns(df) == "Close column not found in data."

    def test_too_few_rows(self):
        df = _make_weekly_df([100])
        assert calculate_weekly_returns(df) == "Not enough data to calculate returns."

    def test_return_calculation_positive(self):
        df = _make_weekly_df([100, 110])
        result = calculate_weekly_returns(df)
        assert isinstance(result, pd.DataFrame)
        assert "weekly_return" in result.columns
        # 110/100 - 1 = 0.10
        assert result["weekly_return"].iloc[1] == pytest.approx(0.10)

    def test_return_calculation_negative(self):
        df = _make_weekly_df([100, 90])
        result = calculate_weekly_returns(df)
        # 90/100 - 1 = -0.10
        assert result["weekly_return"].iloc[1] == pytest.approx(-0.10)

    def test_return_calculation_zero(self):
        df = _make_weekly_df([100, 100])
        result = calculate_weekly_returns(df)
        assert result["weekly_return"].iloc[1] == pytest.approx(0.0)

    def test_first_row_is_nan(self):
        df = _make_weekly_df([100, 110, 120])
        result = calculate_weekly_returns(df)
        assert pd.isna(result["weekly_return"].iloc[0])

    def test_does_not_mutate_input(self):
        df = _make_weekly_df([100, 110])
        original_cols = list(df.columns)
        calculate_weekly_returns(df)
        assert list(df.columns) == original_cols

    def test_multi_week_sequence(self):
        # 100 -> 105 -> 110 -> 100
        df = _make_weekly_df([100, 105, 110, 100])
        result = calculate_weekly_returns(df)
        assert result["weekly_return"].iloc[1] == pytest.approx(0.05)
        assert result["weekly_return"].iloc[2] == pytest.approx(5 / 105)
        assert result["weekly_return"].iloc[3] == pytest.approx(-10 / 110)


# ===================================================================
# Step 3 - add_week_start_dates
# ===================================================================

class TestAddWeekStartDates:
    def test_not_datetime_index(self):
        df = pd.DataFrame({"Close": [100]}, index=[0])
        assert add_week_start_dates(df) == "Weekly data must have a datetime index."

    def test_adds_week_start_column(self):
        df = _make_weekly_df([100, 110])
        result = add_week_start_dates(df)
        assert isinstance(result, pd.DataFrame)
        assert "week_start" in result.columns

    def test_week_start_is_monday(self):
        # 2026-01-07 is a Wednesday -> Monday is 2026-01-05
        df = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2026-01-07"]))
        result = add_week_start_dates(df)
        assert result["week_start"].iloc[0] == date(2026, 1, 5)

    def test_already_monday(self):
        # 2026-01-05 is a Monday -> week_start should be the same day
        df = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2026-01-05"]))
        result = add_week_start_dates(df)
        assert result["week_start"].iloc[0] == date(2026, 1, 5)

    def test_friday_maps_to_monday(self):
        # 2026-01-09 is a Friday -> Monday is 2026-01-05
        df = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2026-01-09"]))
        result = add_week_start_dates(df)
        assert result["week_start"].iloc[0] == date(2026, 1, 5)

    def test_does_not_mutate_input(self):
        df = _make_weekly_df([100, 110])
        original_cols = list(df.columns)
        add_week_start_dates(df)
        assert list(df.columns) == original_cols


# ===================================================================
# Step 4 - build_weekly_comparison_table
# ===================================================================

class TestBuildWeeklyComparisonTable:
    def test_not_dataframe_first(self):
        df = _make_full_pipeline_df([100, 110, 120])
        assert build_weekly_comparison_table("bad", df, 20, "a", "b") == "Stock data must be a DataFrame."

    def test_not_dataframe_second(self):
        df = _make_full_pipeline_df([100, 110, 120])
        assert build_weekly_comparison_table(df, "bad", 20, "a", "b") == "Stock data must be a DataFrame."

    def test_weeks_not_integer(self):
        df1 = _make_full_pipeline_df([100, 110, 120])
        df2 = _make_full_pipeline_df([200, 210, 220])
        assert build_weekly_comparison_table(df1, df2, "20", "a", "b") == "Weeks must be an integer."

    def test_no_overlap(self):
        df1 = _make_full_pipeline_df([100, 110], start_date="2025-01-01")
        df2 = _make_full_pipeline_df([200, 210], start_date="2026-06-01")
        result = build_weekly_comparison_table(df1, df2, 20, "a", "b")
        assert result == "No overlapping weeks found."

    def test_correct_columns(self):
        df1 = _make_full_pipeline_df([100, 110, 120])
        df2 = _make_full_pipeline_df([200, 210, 220])
        result = build_weekly_comparison_table(df1, df2, 20, "aapl", "msft")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["week_start", "aapl_return", "msft_return"]

    def test_oldest_to_newest_order(self):
        df1 = _make_full_pipeline_df([100, 110, 120, 130])
        df2 = _make_full_pipeline_df([200, 210, 220, 230])
        result = build_weekly_comparison_table(df1, df2, 20, "a", "b")
        dates = list(result["week_start"])
        assert dates == sorted(dates)

    def test_tail_trims_to_weeks(self):
        prices = [100 + i for i in range(10)]
        df1 = _make_full_pipeline_df(prices)
        df2 = _make_full_pipeline_df(prices)
        result = build_weekly_comparison_table(df1, df2, 3, "a", "b")
        assert len(result) == 3


# ===================================================================
# Step 5 - get_week_summary
# ===================================================================

class TestGetWeekSummary:
    def _sample_table(self):
        df = pd.DataFrame({
            "week_start": [date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19)],
            "aapl_return": [0.05, -0.02, 0.03],
            "msft_return": [-0.01, 0.04, 0.01],
        })
        return df

    def test_invalid_mode(self):
        assert get_week_summary(self._sample_table(), "bad", 0) == "Mode must be 'weeks_ago' or 'week_start'."

    def test_weeks_ago_not_integer(self):
        assert get_week_summary(self._sample_table(), "weeks_ago", "0") == "Weeks ago must be an integer."

    def test_weeks_ago_out_of_range_negative(self):
        assert get_week_summary(self._sample_table(), "weeks_ago", -1) == "Weeks ago is out of range."

    def test_weeks_ago_out_of_range_too_high(self):
        assert get_week_summary(self._sample_table(), "weeks_ago", 5) == "Weeks ago is out of range."

    def test_weeks_ago_zero_is_most_recent(self):
        result = get_week_summary(self._sample_table(), "weeks_ago", 0)
        assert result["week_start"] == date(2026, 1, 19)
        assert result["aapl_return"] == pytest.approx(0.03)

    def test_weeks_ago_one(self):
        result = get_week_summary(self._sample_table(), "weeks_ago", 1)
        assert result["week_start"] == date(2026, 1, 12)

    def test_weeks_ago_boundary_max(self):
        result = get_week_summary(self._sample_table(), "weeks_ago", 2)
        assert result["week_start"] == date(2026, 1, 5)

    def test_week_start_invalid_date(self):
        assert get_week_summary(self._sample_table(), "week_start", "not-a-date") == \
            "Week start must be a valid date (YYYY-MM-DD)."

    def test_week_start_not_found(self):
        assert get_week_summary(self._sample_table(), "week_start", "2025-06-01") == \
            "Week start not found in table."

    def test_week_start_found(self):
        result = get_week_summary(self._sample_table(), "week_start", "2026-01-12")
        assert result["week_start"] == date(2026, 1, 12)
        assert result["msft_return"] == pytest.approx(0.04)


# ===================================================================
# Helpers
# ===================================================================

class TestExcludeIncompleteCurrentWeek:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["week_start", "Close"])
        result = _exclude_incomplete_current_week(df)
        assert result.empty

    def test_drops_current_week(self):
        # Build a df where the last row is THIS week's Monday
        today = datetime.today().date()
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(weeks=1)
        df = pd.DataFrame({
            "week_start": [last_monday, this_monday],
            "Close": [100, 110],
        })
        result = _exclude_incomplete_current_week(df)
        # Should only have 1 row if today is Mon-Fri of current week
        if today.weekday() <= 4:  # Mon-Fri
            assert len(result) == 1
            assert result["week_start"].iloc[0] == last_monday

    def test_keeps_completed_week(self):
        # A week from 2 weeks ago should be kept
        today = datetime.today().date()
        old_monday = today - timedelta(weeks=2, days=today.weekday())
        df = pd.DataFrame({
            "week_start": [old_monday],
            "Close": [100],
        })
        result = _exclude_incomplete_current_week(df)
        assert len(result) == 1


class TestNormaliseTicker:
    def test_uppercase(self):
        assert _normalise_ticker("aapl") == "AAPL"

    def test_strip_whitespace(self):
        assert _normalise_ticker("  msft  ") == "MSFT"

    def test_uk_hint(self, capsys):
        result = _normalise_ticker("vod")
        assert result == "VOD.L"
        captured = capsys.readouterr()
        assert "UK ticker" in captured.out

    def test_already_suffixed(self):
        result = _normalise_ticker("VOD.L")
        assert result == "VOD.L"


class TestFormatPct:
    def test_positive(self):
        assert _format_pct(0.0123) == "+1.23%"

    def test_negative(self):
        assert _format_pct(-0.0045) == "-0.45%"

    def test_zero(self):
        assert _format_pct(0.0) == "+0.00%"

    def test_nan(self):
        assert _format_pct(float("nan")) == "N/A"

    def test_large_positive(self):
        assert _format_pct(1.0) == "+100.00%"


class TestBuildSingleTickerTable:
    def test_not_dataframe(self):
        assert build_single_ticker_table("bad", 20, "aapl") == "Stock data must be a DataFrame."

    def test_weeks_not_integer(self):
        df = _make_full_pipeline_df([100, 110, 120])
        assert build_single_ticker_table(df, "20", "aapl") == "Weeks must be an integer."

    def test_correct_column_name(self):
        df = _make_full_pipeline_df([100, 110, 120])
        result = build_single_ticker_table(df, 20, "aapl")
        assert "aapl_return" in result.columns

    def test_trims_to_weeks(self):
        prices = [100 + i for i in range(10)]
        df = _make_full_pipeline_df(prices)
        result = build_single_ticker_table(df, 3, "aapl")
        assert len(result) == 3
