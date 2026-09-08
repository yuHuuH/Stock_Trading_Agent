
# Build the Streamlit PPO Trading Demo with PyInstaller

This package adds a PyInstaller setup for the local Streamlit app.

## Why not one single `.exe`?

A true `--onefile` build is not recommended here because the app depends on:

- Streamlit
- Plotly
- PyTorch
- Stable-Baselines3
- Gym/Gymnasium
- your PPO model `.zip`
- your `vec_normalize_stats.pkl`
- the raw S&P 500 CSV files

A one-file build would be very large, slow to start, and more likely to break.
This setup builds a safer Windows folder app:

```text
dist/
└── TradingAgentDemo/
    ├── TradingAgentDemo.exe
    ├── data/
    │   ├── SP500.csv
    │   └── sp500_2025_20260.csv
    ├── artifacts/
    │   ├── best_model.zip
    │   └── vec_normalize_stats.pkl
    └── ...
```

You distribute the whole `dist/TradingAgentDemo` folder.

## Step 1 — Put files in the project before building

Before running PyInstaller, make sure you have:

```text
data/SP500.csv
data/sp500_2025_20260.csv
artifacts/best_model.zip
artifacts/vec_normalize_stats.pkl
```

You can also use:

```text
artifacts/ppo_quant_bot_final.zip
```

instead of `best_model.zip`.

## Step 2 — Build on Windows

Double-click:

```text
build_exe.bat
```

or run:

```bash
python build_exe.py
```

## Step 3 — Run the built app

After build completes:

```text
dist/TradingAgentDemo/TradingAgentDemo.exe
```

The EXE starts a local Streamlit server and opens your browser automatically.

## Important

Do not distribute only `TradingAgentDemo.exe`. Distribute the whole folder:

```text
dist/TradingAgentDemo/
```

The app needs its `_internal` runtime files plus the external `data/` and
`artifacts/` folders.

## Making it compact for sharing

After building, zip this folder:

```text
dist/TradingAgentDemo/
```

That ZIP is the compact distributable version.


## uv environment note

If your `.venv` says `No module named pip`, use the v7 scripts. They install
dependencies with `uv pip install --python ...` and do not require pip inside
the virtual environment.

Recommended:

```bash
build_exe_uv.bat
```


## v8 Streamlit PyInstaller fix

This version fixes the bundled EXE error:

```text
RuntimeError: server.port does not work when global.developmentMode is true.
```

The launcher now forces Streamlit production mode with:

```text
--global.developmentMode=false
```

It also includes `.streamlit/config.toml` with `developmentMode = false`.
