# Weekly Stock Return Reporter

A command-line tool that fetches weekly close-to-close returns for any ticker [yfinance](https://github.com/ranaroussi/yfinance) supports. Compare two stocks side-by-side, find run-up streaks, and export results to CSV or TXT.

## Features

- **Single or dual ticker mode** — analyze one stock or compare two head-to-head
- **Weekly close-to-close returns** — percentage change between weekly closing prices
- **Best / worst week detection** — highlights the strongest and weakest weeks in the window
- **Run-up detector** — finds consecutive-week streaks where compounded return meets a threshold
- **Single-week summary** — drill into any specific week by date or offset
- **Incomplete week filtering** — automatically drops the current week if it hasn't ended yet
- **UK ticker auto-detection** — recognizes common UK tickers and appends `.L` suffix
- **Export** — save tables to CSV, TXT, or both
- **Interactive & CLI modes** — run with no arguments for a guided prompt, or pass flags for scripting

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- `yfinance >= 0.2.31`
- `pandas >= 2.0.0`

## Usage

### Interactive mode (no arguments)

```bash
python weekly_returns.py
```

You'll be prompted for ticker(s), number of weeks, run-up threshold, and export options.

### CLI mode

```bash
# Single ticker — 20 weeks (default)
python weekly_returns.py TSLA

# Two-ticker comparison
python weekly_returns.py AAPL MSFT

# Custom window size
python weekly_returns.py VOD.L BP.L --weeks 10

# Save to CSV
python weekly_returns.py AAPL --csv

# Save to TXT
python weekly_returns.py AAPL --txt

# Find run-ups with cumulative return >= 15%
python weekly_returns.py TLT -w 260 --runups 15

# Specific week summary by date
python weekly_returns.py AAPL --summary-mode week_start --summary-value 2026-01-05
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `tickers` | One or two ticker symbols (positional) | — |
| `-w`, `--weeks` | Number of weeks to display | `20` |
| `--csv` | Save the weekly table to a CSV file | off |
| `--txt` | Save the weekly table to a TXT file | off |
| `--summary-mode` | `weeks_ago` or `week_start` | `weeks_ago` |
| `--summary-value` | Offset (int) or date (`YYYY-MM-DD`) | `0` (most recent) |
| `--runups PCT` | Find run-ups ≥ PCT% cumulative return | off |

## Example Output

```
  Weekly Returns -- COIN vs RDDT
  Window: 20 weeks  |  Price: Close
  ----------------------------------------------------

  Week Start      coin_return        rddt_return
  --------------------------------------------------
  2025-09-22          +2.45%             +1.12%
  2025-09-29          -1.30%             +3.88%
  ...

  [best]  COIN best week:  2025-11-10  +18.42%
  [worst] COIN worst week: 2025-10-07  -9.15%

  [summary] Single-week summary
      Week start: 2026-02-03
      coin_return: +3.21%
      rddt_return: -0.87%
```

## Running Tests

```bash
python -m pytest test_weekly_returns.py -v
```

## License

MIT
