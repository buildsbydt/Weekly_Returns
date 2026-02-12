"""
Weekly Stock Return Reporter
=============================
Fetches weekly close-to-close returns for any ticker yfinance supports,
prints a comparison table and a single-week summary.

Usage examples:
    python weekly_returns.py                        # defaults: AAPL vs MSFT
    python weekly_returns.py AAPL                   # single-ticker mode
    python weekly_returns.py AAPL MSFT              # two-ticker comparison
    python weekly_returns.py VOD.L BP.L --weeks 10  # UK stocks, 10 weeks
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Step 1 - get_weekly_stock_data
# ---------------------------------------------------------------------------

def get_weekly_stock_data(ticker: str, weeks: int) -> pd.DataFrame | str:
    """
    Fetch weekly OHLCV data for *ticker* from yfinance.

    Returns a DataFrame on success, or an error-message string on failure.
    """
    # --- validation ---
    if not isinstance(ticker, str):
        return "Ticker must be a string."
    if ticker.strip() == "":
        return "Ticker cannot be empty."
    if not isinstance(weeks, int):
        return "Weeks must be an integer."
    if weeks < 1 or weeks > 520:
        return "Weeks must be between 1 and 520."

    # --- fetch ---
    # Buffer: fetch weeks + 4 to guarantee enough rows after partial-week trim
    fetch_weeks = weeks + 4
    start_date = datetime.today() - timedelta(weeks=fetch_weeks)
    end_date = datetime.today() + timedelta(days=1)

    data = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1wk",
        progress=False,
        auto_adjust=False,
    )

    if data is None or data.empty:
        return "No data found for ticker."

    # yfinance may return MultiIndex columns for single ticker; flatten
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


# ---------------------------------------------------------------------------
# Step 2 - calculate_weekly_returns
# ---------------------------------------------------------------------------

def calculate_weekly_returns(weekly_df: pd.DataFrame) -> pd.DataFrame | str:
    """
    Add a ``weekly_return`` column (close-to-close pct change) to *weekly_df*.

    Returns the augmented DataFrame, or an error string.
    """
    if not isinstance(weekly_df, pd.DataFrame):
        return "Weekly data must be a DataFrame."
    if "Close" not in weekly_df.columns:
        return "Close column not found in data."
    if len(weekly_df) < 2:
        return "Not enough data to calculate returns."

    weekly_df = weekly_df.copy()
    weekly_df["weekly_return"] = weekly_df["Close"] / weekly_df["Close"].shift(1) - 1
    return weekly_df


# ---------------------------------------------------------------------------
# Step 3 - add_week_start_dates
# ---------------------------------------------------------------------------

def add_week_start_dates(weekly_df: pd.DataFrame) -> pd.DataFrame | str:
    """
    Add a ``week_start`` column containing the Monday of each row's week.

    Returns the augmented DataFrame, or an error string.
    """
    if not isinstance(weekly_df.index, pd.DatetimeIndex):
        return "Weekly data must have a datetime index."

    weekly_df = weekly_df.copy()
    # Monday = weekday 0.  Subtract the weekday number of days to land on Monday.
    weekly_df["week_start"] = weekly_df.index.to_series().apply(
        lambda dt: dt - timedelta(days=dt.weekday())
    )
    weekly_df["week_start"] = pd.to_datetime(weekly_df["week_start"]).dt.date
    return weekly_df


# ---------------------------------------------------------------------------
# Step 4 - build_weekly_comparison_table
# ---------------------------------------------------------------------------

def build_weekly_comparison_table(
    ticker1_df: pd.DataFrame,
    ticker2_df: pd.DataFrame,
    weeks: int,
    ticker1_label: str,
    ticker2_label: str,
) -> pd.DataFrame | str:
    """
    Merge two weekly-return DataFrames into a clean comparison table.

    Returns a DataFrame with columns:
        week_start, <ticker1_label>_return, <ticker2_label>_return
    ordered oldest -> newest, trimmed to the most recent *weeks* rows.
    """
    if not isinstance(ticker1_df, pd.DataFrame) or not isinstance(ticker2_df, pd.DataFrame):
        return "Stock data must be a DataFrame."
    if not isinstance(weeks, int):
        return "Weeks must be an integer."

    col1 = f"{ticker1_label}_return"
    col2 = f"{ticker2_label}_return"

    t1 = ticker1_df[["week_start", "weekly_return"]].rename(columns={"weekly_return": col1})
    t2 = ticker2_df[["week_start", "weekly_return"]].rename(columns={"weekly_return": col2})

    # Inner join: only keep weeks where BOTH tickers have data.
    # This matters when comparing stocks on different exchanges (e.g. US vs UK)
    # because exchange holidays can mean one ticker has a week the other doesn't.
    # For two stocks on the same exchange this almost always keeps every week.
    merged = pd.merge(t1, t2, on="week_start", how="inner")

    if merged.empty:
        return "No overlapping weeks found."

    merged = merged.sort_values("week_start").tail(weeks).reset_index(drop=True)
    return merged[["week_start", col1, col2]]


# ---------------------------------------------------------------------------
# Helper: build single-ticker table (stretch goal #2)
# ---------------------------------------------------------------------------

def build_single_ticker_table(
    ticker_df: pd.DataFrame,
    weeks: int,
    ticker_label: str,
) -> pd.DataFrame | str:
    """Return a single-ticker weekly return table."""
    if not isinstance(ticker_df, pd.DataFrame):
        return "Stock data must be a DataFrame."
    if not isinstance(weeks, int):
        return "Weeks must be an integer."

    col = f"{ticker_label}_return"
    table = ticker_df[["week_start", "weekly_return"]].rename(columns={"weekly_return": col}).copy()
    table = table.dropna(subset=[col])
    table = table.sort_values("week_start").tail(weeks).reset_index(drop=True)

    if table.empty:
        return "No data available for the requested period."

    return table


# ---------------------------------------------------------------------------
# Step 5 - get_week_summary
# ---------------------------------------------------------------------------

def get_week_summary(
    weekly_table: pd.DataFrame,
    mode: str,
    value,
) -> pd.Series | str:
    """
    Return a single-row summary from *weekly_table*.

    Modes
    -----
    * ``"weeks_ago"`` - *value* is an int; 0 = most recent week.
    * ``"week_start"`` - *value* is a date string ``"YYYY-MM-DD"``.
    """
    if mode not in ("weeks_ago", "week_start"):
        return "Mode must be 'weeks_ago' or 'week_start'."

    if mode == "weeks_ago":
        if not isinstance(value, int):
            return "Weeks ago must be an integer."
        max_idx = len(weekly_table) - 1
        if value < 0 or value > max_idx:
            return "Weeks ago is out of range."
        row_idx = max_idx - value
        return weekly_table.iloc[row_idx]

    # mode == "week_start"
    try:
        target = datetime.strptime(str(value), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return "Week start must be a valid date (YYYY-MM-DD)."

    matches = weekly_table[weekly_table["week_start"] == target]
    if matches.empty:
        return "Week start not found in table."

    return matches.iloc[0]


# ---------------------------------------------------------------------------
# Stretch goal helpers
# ---------------------------------------------------------------------------

def _exclude_incomplete_current_week(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the last row if its week hasn't ended yet (today < Friday)."""
    if df.empty:
        return df
    today = datetime.today().date()
    last_week_monday = df["week_start"].iloc[-1]
    # A week is "complete" if we're past its Friday (Monday + 4 days)
    if isinstance(last_week_monday, str):
        last_week_monday = datetime.strptime(last_week_monday, "%Y-%m-%d").date()
    week_friday = last_week_monday + timedelta(days=4)
    if today <= week_friday:
        return df.iloc[:-1].copy()
    return df


def _best_worst(table: pd.DataFrame, return_cols: list[str]) -> dict:
    """Return best and worst week info for each return column."""
    results: dict[str, dict] = {}
    for col in return_cols:
        best_idx = table[col].idxmax()
        worst_idx = table[col].idxmin()
        results[col] = {
            "best_week": table.loc[best_idx, "week_start"],
            "best_return": table.loc[best_idx, col],
            "worst_week": table.loc[worst_idx, "week_start"],
            "worst_return": table.loc[worst_idx, col],
        }
    return results


def _normalise_ticker(raw: str) -> str:
    """Auto-uppercase, strip whitespace; suggest .L suffix for UK names."""
    cleaned = raw.strip().upper()
    # Heuristic: well-known UK tickers without a suffix get a hint
    UK_TICKERS = {"VOD", "BP", "SHEL", "BARC", "HSBA", "LLOY", "RIO", "GSK", "AZN", "ULVR"}
    if cleaned in UK_TICKERS:
        print(f"  [info] '{cleaned}' looks like a UK ticker - using '{cleaned}.L' instead.")
        cleaned = f"{cleaned}.L"
    return cleaned


def _format_pct(val) -> str:
    """Format a float as +1.23% / -0.45%."""
    if pd.isna(val):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.2f}%"


# ---------------------------------------------------------------------------
# Run-up detector: find periods where cumulative return >= threshold
# ---------------------------------------------------------------------------

def find_runups(weekly_table: pd.DataFrame, return_col: str, threshold: float = 0.15) -> list[dict] | str:
    """
    Find consecutive-week run-ups where compounded return >= threshold.

    How it works:
    - Walk through weeks in order, compounding returns from a start point.
    - If cumulative return from start drops below 0% (all gains given back),
      the run-up is over -- check if its peak hit the threshold, then reset.
    - A small negative week in the middle does NOT break the run-up, only
      losing all accumulated gains does.

    Returns a list of dicts:
        [{"start_date", "end_date", "weeks", "total_return"}, ...]
    """
    if not isinstance(weekly_table, pd.DataFrame):
        return "Weekly table must be a DataFrame."
    if return_col not in weekly_table.columns:
        return "Return column not found in table."
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        return "Threshold must be a positive number."

    dates = weekly_table["week_start"].tolist()
    returns = weekly_table[return_col].tolist()
    n = len(returns)

    results = []
    start_idx = 0
    cumulative = 1.0       # wealth multiplier from start (1.0 = breakeven)
    peak_cum = 1.0         # highest cumulative seen in this run
    peak_idx = 0

    for i in range(n):
        r = returns[i]
        if pd.isna(r):
            continue

        cumulative *= (1 + r)

        if cumulative > peak_cum:
            peak_cum = cumulative
            peak_idx = i

        # If cumulative drops below 1.0, all gains from this run are gone.
        # Close the run, check if it was significant, then reset.
        if cumulative < 1.0:
            total_return = peak_cum - 1.0
            if total_return >= threshold:
                results.append({
                    "start_date": dates[start_idx],
                    "end_date": dates[peak_idx],
                    "weeks": peak_idx - start_idx + 1,
                    "total_return": total_return,
                })
            # Reset for the next potential run-up
            start_idx = i + 1
            cumulative = 1.0
            peak_cum = 1.0
            peak_idx = i + 1

    # Check the final open run (data ended before cumulative went negative)
    total_return = peak_cum - 1.0
    if total_return >= threshold and start_idx < n:
        results.append({
            "start_date": dates[start_idx],
            "end_date": dates[peak_idx],
            "weeks": peak_idx - start_idx + 1,
            "total_return": total_return,
        })

    return results


# ---------------------------------------------------------------------------
# Pipeline helper: fetch -> calculate -> add dates (with incomplete-week trim)
# ---------------------------------------------------------------------------

def _prepare_ticker(ticker: str, weeks: int):
    """Run the full pipeline for one ticker.

    Returns (df, label, raw_weeks) or (error_str, None, None).
    raw_weeks = number of weekly rows yfinance returned before any trimming.
    """
    label = ticker.lower().replace(".", "_")

    result = get_weekly_stock_data(ticker, weeks)
    if isinstance(result, str):
        return result, None, None

    raw_weeks = len(result)

    result = calculate_weekly_returns(result)
    if isinstance(result, str):
        return result, None, None

    result = add_week_start_dates(result)
    if isinstance(result, str):
        return result, None, None

    # Stretch goal 1: drop incomplete current week
    result = _exclude_incomplete_current_week(result)

    return result, label, raw_weeks


# ---------------------------------------------------------------------------
# Interactive prompt - runs when user launches with no arguments
# ---------------------------------------------------------------------------

def _interactive_prompt() -> dict:
    """Ask the user for inputs interactively and return a settings dict."""
    print()
    print("  Weekly Stock Return Reporter")
    print("  " + "-" * 40)
    print()

    # --- ticker 1 ---
    t1 = input("  Enter first ticker symbol (e.g. AAPL): ").strip()
    if not t1:
        print("  [error] Ticker cannot be empty.")
        sys.exit(1)
    t1 = _normalise_ticker(t1)

    # --- ticker 2 (optional) ---
    t2_raw = input("  Enter second ticker to compare (or press Enter to skip): ").strip()
    t2 = _normalise_ticker(t2_raw) if t2_raw else None

    # --- table size (weeks of data to display) ---
    weeks_raw = input("  How many weeks of data to show in the table? (default 20, max 520): ").strip()
    if weeks_raw == "":
        weeks = 20
    else:
        try:
            weeks = int(weeks_raw)
        except ValueError:
            print("  [error] Must be a number.")
            sys.exit(1)

    # Summary defaults to most recent week — no need to ask
    summary_mode = "weeks_ago"
    summary_value = "0"

    # --- run-up detector (optional) ---
    print()
    runup_raw = input("  Find run-ups? Enter min % threshold (e.g. 15) or press Enter to skip: ").strip()
    runup_threshold = None
    if runup_raw:
        try:
            runup_threshold = float(runup_raw) / 100.0  # convert 15 -> 0.15
            if runup_threshold <= 0:
                print("  [error] Threshold must be positive.")
                sys.exit(1)
        except ValueError:
            print("  [error] Must be a number.")
            sys.exit(1)

    # --- export ---
    print()
    print("  Save output to file?")
    print("    1) No (default)")
    print("    2) CSV")
    print("    3) TXT")
    print("    4) Both CSV and TXT")
    export_choice = input("  Choose 1/2/3/4 (default 1): ").strip()

    save_csv = export_choice in ("2", "4")
    save_txt = export_choice in ("3", "4")

    tickers = [t1, t2] if t2 else [t1]

    return {
        "tickers": tickers,
        "weeks": weeks,
        "summary_mode": summary_mode,
        "summary_value": summary_value,
        "csv": save_csv,
        "txt": save_txt,
        "runup_threshold": runup_threshold,
    }


# ---------------------------------------------------------------------------
# Step 6 - main script
# ---------------------------------------------------------------------------

def _run(tickers, weeks, summary_mode, summary_value, save_csv, save_txt=False,
         runup_threshold=None):
    """Core logic shared by interactive and CLI modes."""
    single_mode = len(tickers) == 1

    print()
    if single_mode:
        print(f"  Weekly Returns -- {tickers[0]}")
    else:
        print(f"  Weekly Returns -- {tickers[0]} vs {tickers[1]}")
    print(f"  Window: {weeks} weeks  |  Price: Close")
    print("  " + "-" * 52)
    print()

    # --- prepare ticker(s) ---
    df1, label1, raw1 = _prepare_ticker(tickers[0], weeks)
    if isinstance(df1, str):
        print(f"  [error] {tickers[0]}: {df1}")
        sys.exit(1)

    if single_mode:
        table = build_single_ticker_table(df1, weeks, label1)
        if isinstance(table, str):
            print(f"  [error] {table}")
            sys.exit(1)
        return_cols = [f"{label1}_return"]
        # Data info
        print(f"  [data] {tickers[0]}: yfinance returned {raw1} weekly rows, showing {len(table)} weeks")
    else:
        df2, label2, raw2 = _prepare_ticker(tickers[1], weeks)
        if isinstance(df2, str):
            print(f"  [error] {tickers[1]}: {df2}")
            sys.exit(1)
        table = build_weekly_comparison_table(df1, df2, weeks, label1, label2)
        if isinstance(table, str):
            print(f"  [error] {table}")
            sys.exit(1)
        return_cols = [f"{label1}_return", f"{label2}_return"]
        # Data info
        print(f"  [data] {tickers[0]}: yfinance returned {raw1} weekly rows")
        print(f"  [data] {tickers[1]}: yfinance returned {raw2} weekly rows")
        print(f"  [data] Showing {len(table)} weeks where both tickers had data")

    print()

    # --- print table ---
    print(f"  {'Week Start':<14}", end="")
    for col in return_cols:
        print(f"  {col:>18}", end="")
    print()
    print("  " + "-" * (14 + 20 * len(return_cols)))

    for _, row in table.iterrows():
        print(f"  {str(row['week_start']):<14}", end="")
        for col in return_cols:
            print(f"  {_format_pct(row[col]):>18}", end="")
        print()

    print()

    # --- best / worst (stretch goal 4) ---
    bw = _best_worst(table, return_cols)
    for col, info in bw.items():
        ticker_name = col.replace("_return", "").upper()
        print(f"  [best]  {ticker_name} best week:  {info['best_week']}  {_format_pct(info['best_return'])}")
        print(f"  [worst]  {ticker_name} worst week: {info['worst_week']}  {_format_pct(info['worst_return'])}")
    print()

    # --- single-week summary ---
    if summary_mode == "weeks_ago":
        try:
            value: int | str = int(summary_value)
        except ValueError:
            print("  [error] Summary value must be an integer for weeks_ago mode.")
            sys.exit(1)
    else:
        value = summary_value

    summary = get_week_summary(table, summary_mode, value)
    if isinstance(summary, str):
        print(f"  [error] {summary}")
        sys.exit(1)

    print("  [summary] Single-week summary")
    print(f"      Week start: {summary['week_start']}")
    for col in return_cols:
        print(f"      {col}: {_format_pct(summary[col])}")
    print()

    # --- run-up analysis ---
    if runup_threshold is not None:
        print(f"  [run-ups] Scanning for cumulative run-ups >= {runup_threshold * 100:.0f}%")
        print()
        for col in return_cols:
            ticker_name = col.replace("_return", "").upper()
            runups = find_runups(table, col, runup_threshold)
            if isinstance(runups, str):
                print(f"  [error] {runups}")
                continue
            if not runups:
                print(f"  {ticker_name}: No run-ups found meeting the {runup_threshold * 100:.0f}% threshold.")
            else:
                print(f"  {ticker_name}: Found {len(runups)} run-up(s):")
                print()
                print(f"    {'#':>3}  {'Start Date':<14}  {'End Date':<14}  {'Weeks':>5}  {'Total Return':>13}")
                print(f"    {'---':>3}  {'-' * 14}  {'-' * 14}  {'-----':>5}  {'-' * 13}")
                for idx, ru in enumerate(runups, 1):
                    start_str = str(ru["start_date"])[:10]
                    end_str = str(ru["end_date"])[:10]
                    print(f"    {idx:>3}  {start_str:<14}  {end_str:<14}  {ru['weeks']:>5}  {_format_pct(ru['total_return']):>13}")
                print()
            print()

    # --- CSV export (stretch goal 3) ---
    if save_csv:
        filename = "_vs_".join(t.replace(".", "_") for t in tickers) + "_weekly.csv"
        table.to_csv(filename, index=False)
        print(f"  [saved] Table saved to {filename}")

    # --- TXT export ---
    if save_txt:
        filename = "_vs_".join(t.replace(".", "_") for t in tickers) + "_weekly.txt"
        lines = []
        if single_mode:
            lines.append(f"Weekly Returns -- {tickers[0]}")
        else:
            lines.append(f"Weekly Returns -- {tickers[0]} vs {tickers[1]}")
        lines.append(f"Window: {weeks} weeks  |  Price: Close")
        lines.append("-" * (14 + 20 * len(return_cols)))

        header = f"{'Week Start':<14}"
        for col in return_cols:
            header += f"  {col:>18}"
        lines.append(header)
        lines.append("-" * (14 + 20 * len(return_cols)))

        for _, row in table.iterrows():
            line = f"{str(row['week_start']):<14}"
            for col in return_cols:
                line += f"  {_format_pct(row[col]):>18}"
            lines.append(line)

        lines.append("")
        for col, info in bw.items():
            ticker_name = col.replace("_return", "").upper()
            lines.append(f"Best week:  {ticker_name}  {info['best_week']}  {_format_pct(info['best_return'])}")
            lines.append(f"Worst week: {ticker_name}  {info['worst_week']}  {_format_pct(info['worst_return'])}")

        lines.append("")
        lines.append("Single-week summary")
        lines.append(f"  Week start: {summary['week_start']}")
        for col in return_cols:
            lines.append(f"  {col}: {_format_pct(summary[col])}")

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [saved] Table saved to {filename}")

    print()


def main() -> None:
    # If no CLI arguments provided, launch interactive mode
    if len(sys.argv) == 1:
        settings = _interactive_prompt()
        _run(
            tickers=settings["tickers"],
            weeks=settings["weeks"],
            summary_mode=settings["summary_mode"],
            summary_value=settings["summary_value"],
            save_csv=settings["csv"],
            save_txt=settings["txt"],
            runup_threshold=settings.get("runup_threshold"),
        )
        return

    # Otherwise use CLI argument parsing
    parser = argparse.ArgumentParser(
        description="Weekly Stock Return Reporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 weekly_returns.py                  # interactive mode\n"
            "  python3 weekly_returns.py TSLA             # single ticker\n"
            "  python3 weekly_returns.py VOD.L BP.L       # UK stocks\n"
            "  python3 weekly_returns.py AAPL MSFT -w 10  # 10-week window\n"
            "  python3 weekly_returns.py AAPL --csv       # save to CSV\n"
            "  python3 weekly_returns.py AAPL --txt       # save to TXT\n"
            "  python3 weekly_returns.py TLT -w 260 --runups 15  # find 15%+ run-ups\n"
        ),
    )
    parser.add_argument("tickers", nargs="+",
                        help="One or two ticker symbols")
    parser.add_argument("-w", "--weeks", type=int, default=20,
                        help="Number of weeks to display (default: 20)")
    parser.add_argument("--csv", action="store_true",
                        help="Save the weekly table to a CSV file")
    parser.add_argument("--txt", action="store_true",
                        help="Save the weekly table to a TXT file")
    parser.add_argument("--summary-mode", type=str, default="weeks_ago",
                        choices=["weeks_ago", "week_start"],
                        help="Summary lookup mode (default: weeks_ago)")
    parser.add_argument("--summary-value", type=str, default="0",
                        help="Value for the summary lookup (default: 0 = most recent week)")
    parser.add_argument("--runups", type=float, default=None, metavar="PCT",
                        help="Find run-ups with cumulative return >= PCT%% (e.g. --runups 15)")

    args = parser.parse_args()

    tickers = [_normalise_ticker(t) for t in args.tickers[:2]]
    runup_threshold = args.runups / 100.0 if args.runups is not None else None

    _run(
        tickers=tickers,
        weeks=args.weeks,
        summary_mode=args.summary_mode,
        summary_value=args.summary_value,
        save_csv=args.csv,
        save_txt=args.txt,
        runup_threshold=runup_threshold,
    )


if __name__ == "__main__":
    main()
