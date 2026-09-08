
# Local PPO Trading-Agent Streamlit Demo

This is a **local-only** demo app for your saved S&P 500 PPO trading agent.

It does **not** use:

- Kaggle paths
- W&B / wandb
- kaggle_secrets
- pandas-ta
- numba / llvmlite

## Folder structure

```text
trading_agent/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── SP500.csv
│   └── sp500_2025_20260.csv
└── artifacts/
    ├── best_model.zip
    └── vec_normalize_stats.pkl
```

You can also use `artifacts/ppo_quant_bot_final.zip` instead of `best_model.zip`.

## What goes in `data/`

Put the **raw CSV files** there:

```text
data/SP500.csv
data/sp500_2025_20260.csv
```

Do not put processed feature data there. The app rebuilds indicators from the raw CSV files.

## What goes in `artifacts/`

Copy your trained PPO outputs there:

```text
artifacts/best_model.zip
artifacts/vec_normalize_stats.pkl
```

The `.zip` model and `.pkl` scaler should come from the same training run.

## Install

Using uv:

```bash
uv pip install -r requirements.txt
```

Using normal pip:

```bash
pip install -r requirements.txt
```

## Run

Recommended:

```bash
streamlit run app.py
```

Alternative:

```bash
python -m streamlit run app.py
```

Windows shortcut:

```text
Double-click run_app.bat
```

Or:

```bash
python run_local.py
```

Do **not** run this with `python app.py`. Streamlit apps need the Streamlit runtime.
If you run `python app.py`, the app will now exit cleanly and tell you the correct command.

## Features

The Streamlit app lets you:

- Pick start date and end date
- Input starting money
- Run the saved PPO agent locally
- Show final money, final gain/loss, percentage return, drawdown, annualized return, win rate, and buy-and-hold comparison
- Visualize the full backtest
- Replay the trading process step by step with a slider
- See daily action logs and completed Long/Short segments
- Download the daily backtest log as CSV

## Important

The app recreates the same environment logic and feature engineering used in your training notebook. If you change the indicators, window size, or environment after training, the saved PPO model and `vec_normalize_stats.pkl` may no longer match.


## v3 fix

This version adds explicit unique Streamlit keys to every `st.plotly_chart()`
call. This fixes:

```text
StreamlitDuplicateElementId: There are multiple plotly_chart elements with the same auto-generated ID
```


## v4 update

The starting-capital input is now dynamic. After the user selects a start and end
date, the app finds the lowest S&P 500 close inside that selected period and
uses it as the minimum allowed starting capital.


## v5 update

Added more analysis plots:

- Portfolio drawdown / underwater curve
- Daily return comparison between PPO agent and S&P 500
- Daily return distribution
- Rolling 20-trading-day return comparison
- Cumulative PPO reward vs raw action strength
- Time spent in Long / Short / Flat
- Trade P&L by completed segment
- Agent daily return vs market daily return scatter plot
- Replay view now also includes reward/action behavior up to the selected step


## v6 action-result labels on chart

This version adds action-result annotations to the price chart so that each
state change can display:

- Profit or Loss when a trade is closed
- Open LONG / SHORT with exposure percentage
- Capital at the switch point

The behavior is inspired by the sample action-result chart. You can turn the
labels on or off with checkboxes in the **Full backtest charts** and
**Replay / process view** tabs.


## v7 update

Removed the caption line from the Streamlit UI.


## v8 file detection fix

The app now searches for required files in:
1. the folder containing `app.py`
2. the terminal working directory
3. parent folders of both locations

The sidebar also shows the exact checked paths.
