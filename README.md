
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
