
"""
Local Streamlit PPO Trading-Agent Demo
======================================

This is a local-only demo. It does not use Kaggle paths and does not use W&B.

Expected files:
    data/SP500.csv
    data/sp500_2025_20260.csv
    artifacts/best_model.zip
    artifacts/vec_normalize_stats.pkl

Run:
    streamlit run app.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from gymnasium import spaces
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore")

# Streamlit apps must run through `streamlit run app.py`.
# If the user accidentally runs `python app.py`, Streamlit has no session
# context and st.session_state will not work. Exit cleanly with instructions.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    get_script_run_ctx = None

if get_script_run_ctx is None or get_script_run_ctx() is None:
    import sys

    print("\nThis is a Streamlit app, not a normal Python script.")
    print("Start it with one of these commands:\n")
    print("  streamlit run app.py")
    print("  python -m streamlit run app.py\n")
    print("On Windows, you can also double-click: run_app.bat")
    sys.exit(0)


# =============================================================================
# 1. LOCAL PROJECT PATHS ONLY
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent


def candidate_base_dirs() -> list[Path]:
    """
    Search common locations so the app still works if Streamlit is launched
    from a different working directory.
    """
    bases: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in bases:
            bases.append(resolved)

    add(BASE_DIR)
    add(Path.cwd())

    for parent in BASE_DIR.parents:
        add(parent)

    for parent in Path.cwd().parents:
        add(parent)

    return bases


def find_existing_file(relative_path: str) -> Path:
    for base in candidate_base_dirs():
        candidate = base / relative_path
        if candidate.exists():
            return candidate
    return BASE_DIR / relative_path


def checked_paths(relative_path: str) -> list[Path]:
    return [base / relative_path for base in candidate_base_dirs()]


HISTORICAL_CSV = find_existing_file("data/SP500.csv")
RECENT_CSV = find_existing_file("data/sp500_2025_20260.csv")

BEST_MODEL = find_existing_file("artifacts/best_model.zip")
FINAL_MODEL = find_existing_file("artifacts/ppo_quant_bot_final.zip")
SCALER_PATH = find_existing_file("artifacts/vec_normalize_stats.pkl")

WINDOW_SIZE = 20
DEFAULT_CAPITAL = 10_000.0
MINIMUM_BACKTEST_DAYS = 30


# =============================================================================
# 2. PURE-PANDAS INDICATORS — NO pandas-ta, NO numba, NO llvmlite
# =============================================================================
def parse_mixed_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def add_rsi(df: pd.DataFrame, length: int = 15) -> pd.DataFrame:
    close = df["Close"]
    diff = close.diff()

    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{length}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    close = df["Close"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    df[f"MACD_{fast}_{slow}_{signal}"] = macd
    df[f"MACDh_{fast}_{slow}_{signal}"] = macd_hist
    df[f"MACDs_{fast}_{slow}_{signal}"] = macd_signal
    return df


def add_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std_mult: float = 2.0,
) -> pd.DataFrame:
    close = df["Close"]
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=1)

    lower = mid - std_mult * std
    upper = mid + std_mult * std

    # Matches the names used in your original drop list.
    suffix = f"{length}_{std_mult:.1f}_{std_mult:.1f}"

    df[f"BBL_{suffix}"] = lower
    df[f"BBM_{suffix}"] = mid
    df[f"BBU_{suffix}"] = upper
    df[f"BBB_{suffix}"] = ((upper - lower) / mid) * 100
    df[f"BBP_{suffix}"] = (close - lower) / (upper - lower)
    return df


@st.cache_data(show_spinner="Loading raw CSV files and rebuilding indicators...")
def load_and_prepare_market_data() -> pd.DataFrame:
    historical = pd.read_csv(HISTORICAL_CSV)
    recent = pd.read_csv(RECENT_CSV)

    if "Adj Close" in recent.columns:
        recent = recent.drop(columns=["Adj Close"])

    if "Price" in recent.columns and "Date" not in recent.columns:
        recent = recent.rename(columns={"Price": "Date"})

    # Make the recent CSV match the original historical CSV schema.
    recent = recent.reindex(columns=historical.columns)

    historical["Date"] = parse_mixed_dates(historical["Date"])
    recent["Date"] = parse_mixed_dates(recent["Date"])

    df = pd.concat([historical, recent], ignore_index=True)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    df = df.set_index("Date").sort_index()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()

    # Same feature-building order as the training notebook.
    df = add_rsi(df, length=15)
    df = add_macd(df, fast=12, slow=26, signal=9)
    df = add_bbands(df, length=20, std_mult=2.0)

    df["Returns"] = df["Close"].pct_change()
    df["Z_Score"] = (
        (df["Close"] - df["Close"].rolling(20).mean())
        / df["Close"].rolling(20).std()
    )

    df["MA200"] = df["Close"].rolling(window=200).mean()
    df["Trend_MA200"] = (df["Close"] - df["MA200"]) / df["MA200"]

    columns_to_drop = [
        "BBL_20_2.0_2.0",
        "BBM_20_2.0_2.0",
        "BBU_20_2.0_2.0",
        "MA200",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df = df.dropna().copy()

    return df


# =============================================================================
# 3. SAME CUSTOM ENVIRONMENT AS YOUR NOTEBOOK
# =============================================================================
class Env_Quant_PPO(StocksEnv):
    def __init__(self, df, window_size, frame_bound, initial_capital=10000.0):
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)

        self.initial_capital = float(initial_capital)
        self.df = df
        self.trade_fee = 0.0005

        self.max_exposure = 0.50
        self.deadzone = 0.20

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        features_count = self.signal_features.shape[1] + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, features_count),
            dtype=np.float32,
        )

    def _process_data(self):
        prices = self.df.loc[:, "Close"].to_numpy()

        feature_columns = [
            col
            for col in self.df.columns
            if col not in ["Open", "High", "Low", "Close", "Volume"]
        ]

        signal_features = self.df.loc[:, feature_columns].to_numpy()
        return prices, signal_features

    def reset(self, **kwargs):
        self.days_held = 0
        self._position = 1
        self.current_capital = self.initial_capital

        obs, info = super().reset(**kwargs)

        self._position = 1
        self._last_trade_tick = self._current_tick
        self.history = {
            "step": [],
            "reward": [],
            "total_reward": [],
            "action": [],
            "capital": [],
            "position_size": [],
        }
        return self._get_observation(), info

    def _get_observation(self):
        obs = super()._get_observation()

        current_pos = (
            self._position.value
            if hasattr(self._position, "value")
            else self._position
        )

        unrealized_pnl = 0.0
        if current_pos != 1 and self._last_trade_tick is not None:
            entry_price = self.prices[self._last_trade_tick]
            current_price = self.prices[self._current_tick]

            if current_pos == 2:
                unrealized_pnl = (current_price - entry_price) / entry_price
            elif current_pos == 0:
                unrealized_pnl = (entry_price - current_price) / entry_price

        pos_arr = np.full((self.window_size, 1), float(current_pos))
        days_arr = np.full((self.window_size, 1), float(self.days_held) / 10.0)
        pnl_arr = np.full((self.window_size, 1), unrealized_pnl * 10.0)

        return np.concatenate((obs, pos_arr, days_arr, pnl_arr), axis=1)

    def _calculate_reward(
        self,
        logical_position,
        position_size,
        actual_profit_pct,
    ):
        step_reward = actual_profit_pct

        current_ma_trend = self.df["Trend_MA200"].iloc[self._current_tick]
        if current_ma_trend < 0:
            if logical_position == 2:
                step_reward -= 0.005 * position_size
            elif logical_position == 1:
                step_reward += 0.001
        elif current_ma_trend > 0:
            if logical_position == 0:
                step_reward -= 0.005 * position_size

        if (
            self._current_tick > self._start_tick
            and logical_position != self._position
        ):
            step_reward -= 0.002

        if logical_position == self._position and logical_position != 1:
            max_hold_days = 20

            if self.days_held <= max_hold_days:
                patience_bonus = min(self.days_held * 0.0001, 0.005)
                step_reward += patience_bonus
            else:
                overdue_days = self.days_held - max_hold_days
                time_penalty = overdue_days * 0.0005
                step_reward -= time_penalty

        return np.clip(step_reward * 10.0, -1.0, 1.0)

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        raw_action = float(action[0])
        logical_position = 1
        position_size = 0.0

        if raw_action > self.deadzone:
            logical_position = 2
            position_size = raw_action * self.max_exposure
        elif raw_action < -self.deadzone:
            logical_position = 0
            position_size = abs(raw_action) * self.max_exposure

        current_price = self.prices[self._current_tick]
        last_price = self.prices[self._current_tick - 1]
        price_change_pct = (current_price - last_price) / last_price

        actual_profit_pct = 0.0
        if logical_position == 2:
            actual_profit_pct = price_change_pct * position_size
        elif logical_position == 0:
            actual_profit_pct = -price_change_pct * position_size

        if (
            self._current_tick > self._start_tick
            and logical_position != self._position
        ):
            actual_profit_pct -= self.trade_fee * position_size
            self._last_trade_tick = self._current_tick
            self.days_held = 0
        else:
            self.days_held += 1

        self.current_capital *= 1 + actual_profit_pct
        self._position = logical_position

        step_reward = self._calculate_reward(
            logical_position=logical_position,
            position_size=position_size,
            actual_profit_pct=actual_profit_pct,
        )
        self._total_reward += step_reward

        self.history["step"].append(self._current_tick)
        self.history["reward"].append(step_reward)
        self.history["total_reward"].append(self._total_reward)
        self.history["action"].append(raw_action)
        self.history["capital"].append(self.current_capital)
        self.history["position_size"].append(position_size)

        info = {
            "total_reward": self._total_reward,
            "action": raw_action,
            "logical_pos": logical_position,
            "size": position_size,
            "capital": self.current_capital,
            "history": self.history,
        }

        return self._get_observation(), step_reward, self._done, False, info


# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================
def get_model_path() -> Path:
    if BEST_MODEL.exists():
        return BEST_MODEL
    if FINAL_MODEL.exists():
        return FINAL_MODEL
    raise FileNotFoundError(
        "No PPO model found. Put best_model.zip or ppo_quant_bot_final.zip "
        "inside the artifacts/ folder."
    )


def select_backtest_slice(
    full_df: pd.DataFrame,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if requested_start > requested_end:
        raise ValueError("Start date must be before or equal to end date.")

    selected = full_df.loc[requested_start:requested_end].copy()
    if len(selected) < MINIMUM_BACKTEST_DAYS:
        raise ValueError(
            f"Select at least {MINIMUM_BACKTEST_DAYS} trading days. "
            f"Current selection has {len(selected)} trading days."
        )

    actual_start = selected.index[0]
    actual_end = selected.index[-1]
    start_pos = full_df.index.get_loc(actual_start)

    if start_pos < WINDOW_SIZE:
        raise ValueError(
            f"The agent needs {WINDOW_SIZE} prior trading days as warm-up. "
            "Choose a later start date."
        )

    warmup = full_df.iloc[start_pos - WINDOW_SIZE:start_pos]
    run_df = pd.concat([warmup, selected], axis=0)

    return run_df, actual_start, actual_end


def action_to_position(raw_action: float, deadzone: float = 0.20) -> str:
    if raw_action > deadzone:
        return "Long"
    if raw_action < -deadzone:
        return "Short"
    return "Flat"


def run_backtest(
    full_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    starting_capital: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if starting_capital <= 0:
        raise ValueError("Starting capital must be greater than 0.")

    selected_for_min = full_df.loc[start_date:end_date]
    if not selected_for_min.empty:
        min_required_capital = float(selected_for_min["Close"].min())
        if starting_capital < min_required_capital:
            raise ValueError(
                "Starting capital is below the selected period's lowest close price. "
                f"Minimum required: ${min_required_capital:,.2f}"
            )

    model_path = get_model_path()
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "VecNormalize scaler not found. Put vec_normalize_stats.pkl "
            "inside the artifacts/ folder."
        )

    run_df, actual_start, actual_end = select_backtest_slice(
        full_df=full_df,
        requested_start=start_date,
        requested_end=end_date,
    )

    raw_env = Env_Quant_PPO(
        df=run_df,
        window_size=WINDOW_SIZE,
        frame_bound=(WINDOW_SIZE, len(run_df)),
        initial_capital=starting_capital,
    )

    vec_env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(str(SCALER_PATH), vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    obs = env.reset()
    done = False
    final_info = None

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = bool(dones[0])
        if done:
            final_info = infos[0]

    if final_info is None:
        raise RuntimeError("Backtest finished without returning final environment info.")

    history = final_info["history"]
    actual_env = env.venv.envs[0]

    step_indices = np.asarray(history["step"], dtype=int)
    raw_actions = np.asarray(history["action"], dtype=float)
    exposures = np.asarray(history["position_size"], dtype=float)
    capitals = np.asarray(history["capital"], dtype=float)
    rewards = np.asarray(history["reward"], dtype=float)

    prices = actual_env.prices[step_indices]
    dates = actual_env.df.index[step_indices]

    positions = [action_to_position(a, actual_env.deadzone) for a in raw_actions]

    trend = actual_env.df["Trend_MA200"].iloc[step_indices].to_numpy(dtype=float)
    ma200 = np.divide(
        prices,
        1.0 + trend,
        out=np.full_like(prices, np.nan, dtype=float),
        where=np.abs(1.0 + trend) > 1e-12,
    )

    daily = pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "MA200": ma200,
            "Raw action": raw_actions,
            "Position": positions,
            "Exposure": exposures,
            "Capital": capitals,
            "Reward": rewards,
        }
    )

    daily = daily[daily["Date"] >= actual_start].reset_index(drop=True)

    if daily.empty:
        raise RuntimeError("No visible backtest rows were produced.")

    trades = build_trade_segments(daily, starting_capital)

    final_capital = float(daily["Capital"].iloc[-1])
    profit_usd = final_capital - starting_capital
    profit_pct = (profit_usd / starting_capital) * 100

    buy_hold_final = starting_capital * (daily["Close"].iloc[-1] / daily["Close"].iloc[0])
    buy_hold_pct = ((buy_hold_final / starting_capital) - 1) * 100

    running_peak = daily["Capital"].cummax()
    max_drawdown_pct = float(((daily["Capital"] / running_peak) - 1).min() * 100)

    n_days = len(daily)
    annualized_return = (
        ((final_capital / starting_capital) ** (252 / n_days) - 1) * 100
        if final_capital > 0 and n_days > 0
        else np.nan
    )

    wins = int((trades["P&L ($)"] > 0).sum()) if not trades.empty else 0
    win_rate = (wins / len(trades)) * 100 if len(trades) else 0.0

    state_changes = int(daily["Position"].ne(daily["Position"].shift(1)).sum() - 1)

    summary = {
        "actual_start": actual_start,
        "actual_end": actual_end,
        "first_result_date": daily["Date"].iloc[0],
        "starting_capital": starting_capital,
        "final_capital": final_capital,
        "profit_usd": profit_usd,
        "profit_pct": profit_pct,
        "buy_hold_final": float(buy_hold_final),
        "buy_hold_pct": float(buy_hold_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "annualized_return": float(annualized_return),
        "total_reward": float(final_info["total_reward"]),
        "state_changes": state_changes,
        "completed_trade_count": len(trades),
        "win_rate": win_rate,
        "model_path": str(model_path),
    }

    return summary, daily, trades


def build_trade_segments(daily: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    rows = []
    active_side = "Flat"
    entry_idx = None
    entry_capital = starting_capital

    for i, row in daily.iterrows():
        current_side = row["Position"]

        if current_side == active_side:
            continue

        if active_side != "Flat" and entry_idx is not None:
            exit_idx = max(i - 1, entry_idx)
            exit_capital = float(daily.iloc[exit_idx]["Capital"])
            pnl = exit_capital - entry_capital
            rows.append(
                {
                    "Entry date": daily.iloc[entry_idx]["Date"].strftime("%Y-%m-%d"),
                    "Exit date": daily.iloc[exit_idx]["Date"].strftime("%Y-%m-%d"),
                    "Side": active_side,
                    "Days held": int(exit_idx - entry_idx + 1),
                    "Entry exposure": f"{daily.iloc[entry_idx]['Exposure'] * 100:.1f}%",
                    "Entry capital": round(entry_capital, 2),
                    "Exit capital": round(exit_capital, 2),
                    "P&L ($)": round(pnl, 2),
                    "P&L (%)": round((pnl / entry_capital) * 100, 2)
                    if entry_capital != 0
                    else np.nan,
                }
            )

        if current_side != "Flat":
            entry_idx = i
            entry_capital = (
                starting_capital if i == 0 else float(daily.iloc[i - 1]["Capital"])
            )
        else:
            entry_idx = None

        active_side = current_side

    if active_side != "Flat" and entry_idx is not None:
        exit_idx = len(daily) - 1
        exit_capital = float(daily.iloc[exit_idx]["Capital"])
        pnl = exit_capital - entry_capital
        rows.append(
            {
                "Entry date": daily.iloc[entry_idx]["Date"].strftime("%Y-%m-%d"),
                "Exit date": daily.iloc[exit_idx]["Date"].strftime("%Y-%m-%d"),
                "Side": active_side,
                "Days held": int(exit_idx - entry_idx + 1),
                "Entry exposure": f"{daily.iloc[entry_idx]['Exposure'] * 100:.1f}%",
                "Entry capital": round(entry_capital, 2),
                "Exit capital": round(exit_capital, 2),
                "P&L ($)": round(pnl, 2),
                "P&L (%)": round((pnl / entry_capital) * 100, 2)
                if entry_capital != 0
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def build_action_annotations(
    daily: pd.DataFrame,
    starting_capital: float,
) -> tuple[list[dict], list[pd.Timestamp]]:
    """
    Build annotation labels for each action/state change so the user can see
    the result of each action:
    - Profit / Loss when a position is closed
    - Open LONG / SHORT with exposure %
    - Capital at the switch point
    """
    view = daily.reset_index(drop=True).copy()

    annotations: list[dict] = []
    vlines: list[pd.Timestamp] = []

    if view.empty:
        return annotations, vlines

    prev_position = str(view.loc[0, "Position"])
    trade_entry_capital = float(starting_capital)

    if prev_position != "Flat":
        first_exposure = float(view.loc[0, "Exposure"]) * 100
        first_capital = float(starting_capital)
        invested = first_capital * float(view.loc[0, "Exposure"])
        open_color = "green" if prev_position == "Long" else "red"

        annotations.append(
            dict(
                x=view.loc[0, "Date"],
                y=float(view.loc[0, "Close"]),
                text=(
                    f"Open {prev_position.upper()}: {first_exposure:.1f}% capital"
                    f"<br>(${invested:,.0f} / ${first_capital:,.0f})"
                ),
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-80 if prev_position == "Long" else 80,
                bgcolor="#fff8e8",
                bordercolor=open_color,
                borderwidth=1.2,
                font=dict(color=open_color, size=10),
                align="left",
            )
        )
        vlines.append(view.loc[0, "Date"])

    for i in range(1, len(view)):
        current_position = str(view.loc[i, "Position"])
        if current_position == prev_position:
            continue

        current_date = view.loc[i, "Date"]
        current_price = float(view.loc[i, "Close"])
        current_exposure_pct = float(view.loc[i, "Exposure"]) * 100

        capital_before_switch = float(view.loc[i - 1, "Capital"])
        event_lines = []
        main_color = "gray"

        if prev_position != "Flat":
            trade_pnl = capital_before_switch - trade_entry_capital
            sign = "+" if trade_pnl > 0 else ""
            status = "Profit" if trade_pnl > 0 else ("Loss" if trade_pnl < 0 else "Break-even")

            event_lines.append(f"{status}: {sign}${trade_pnl:,.0f}")
            event_lines.append(
                f"Close {prev_position.upper()} → "
                f"{current_position.upper() if current_position != 'Flat' else 'FLAT'}"
            )
            event_lines.append(f"Capital: ${capital_before_switch:,.0f}")

            main_color = "green" if trade_pnl > 0 else ("red" if trade_pnl < 0 else "gray")

        if current_position != "Flat":
            open_capital = capital_before_switch
            invested = open_capital * float(view.loc[i, "Exposure"])
            event_lines.append(f"Open {current_position.upper()}: {current_exposure_pct:.1f}% capital")
            event_lines.append(f"(${invested:,.0f} / ${open_capital:,.0f})")

            if prev_position == "Flat":
                main_color = "green" if current_position == "Long" else "red"

            trade_entry_capital = open_capital
        else:
            if prev_position == "Flat":
                event_lines.append("Stay FLAT")
            else:
                event_lines.append("Switch to FLAT")

        if current_position == "Long":
            ay = -90
        elif current_position == "Short":
            ay = 90
        else:
            ay = -90 if prev_position == "Short" else 90

        annotations.append(
            dict(
                x=current_date,
                y=current_price,
                text="<br>".join(event_lines),
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=ay,
                bgcolor="#fff8e8",
                bordercolor=main_color,
                borderwidth=1.2,
                font=dict(color=main_color, size=10),
                align="left",
            )
        )
        vlines.append(current_date)
        prev_position = current_position

    return annotations, vlines


def plot_price_actions(daily: pd.DataFrame, starting_capital: float, upto: int | None = None, show_action_labels: bool = True) -> go.Figure:
    view = daily if upto is None else daily.iloc[: upto + 1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["Close"],
            mode="lines",
            name="S&P 500 Close",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["MA200"],
            mode="lines",
            name="MA200",
            line=dict(width=2, dash="dash"),
        )
    )

    state_change = view["Position"].ne(view["Position"].shift(1))
    state_change.iloc[0] = True

    marker_symbols = {
        "Long": "triangle-up",
        "Short": "triangle-down",
        "Flat": "circle",
    }

    for side, symbol in marker_symbols.items():
        mask = state_change & view["Position"].eq(side)
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[mask, "Date"],
                    y=view.loc[mask, "Close"],
                    mode="markers",
                    name=f"Switch to {side}",
                    marker=dict(symbol=symbol, size=12),
                    customdata=np.c_[
                        view.loc[mask, "Raw action"],
                        view.loc[mask, "Exposure"] * 100,
                        view.loc[mask, "Capital"],
                    ],
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Close: %{y:.2f}<br>"
                        "Raw action: %{customdata[0]:.3f}<br>"
                        "Exposure: %{customdata[1]:.1f}%<br>"
                        "Capital: $%{customdata[2]:,.2f}<extra></extra>"
                    ),
                )
            )

    if upto is not None and len(view) > 0:
        current = view.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[current["Date"]],
                y=[current["Close"]],
                mode="markers",
                name="Current replay step",
                marker=dict(size=18, symbol="star"),
            )
        )

    if show_action_labels and len(view) > 0:
        action_annotations, action_vlines = build_action_annotations(
            daily=view,
            starting_capital=float(starting_capital),
        )
        for xline in action_vlines:
            fig.add_vline(x=xline, line_dash="dash", line_color="gray", opacity=0.35)

        for ann in action_annotations:
            fig.add_annotation(**ann)

    fig.update_layout(
        title="Agent trading process on S&P 500 price",
        xaxis_title="Date",
        yaxis_title="S&P 500 Close",
        hovermode="x unified",
        height=560,
        legend_title="Agent action",
    )

    return fig


def plot_equity(daily: pd.DataFrame, starting_capital: float, upto: int | None = None) -> go.Figure:
    view = daily if upto is None else daily.iloc[: upto + 1]
    buy_hold = starting_capital * (view["Close"] / view["Close"].iloc[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["Capital"],
            mode="lines",
            name="PPO Agent Portfolio",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=buy_hold,
            mode="lines",
            name="Buy & Hold Benchmark",
            line=dict(width=2, dash="dash"),
        )
    )

    running_peak = view["Capital"].cummax()
    drawdown_pct = ((view["Capital"] / running_peak) - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=drawdown_pct,
            mode="lines",
            name="Drawdown %",
            yaxis="y2",
            line=dict(width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Portfolio value through time",
        xaxis_title="Date",
        yaxis_title="Capital",
        yaxis2=dict(
            title="Drawdown %",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        height=560,
        legend_title="Series",
    )

    return fig


def plot_position_exposure(daily: pd.DataFrame, upto: int | None = None) -> go.Figure:
    view = daily if upto is None else daily.iloc[: upto + 1]

    side_value = view["Position"].map({"Short": -1, "Flat": 0, "Long": 1}).astype(float)
    signed_exposure = side_value * view["Exposure"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=view["Date"],
            y=signed_exposure,
            name="Signed exposure %",
            customdata=np.c_[view["Position"], view["Raw action"]],
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Exposure: %{y:.1f}%<br>"
                "Position: %{customdata[0]}<br>"
                "Raw action: %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="How much capital the agent uses each day",
        xaxis_title="Date",
        yaxis_title="Signed exposure (%)",
        height=360,
    )

    return fig



def plot_drawdown_only(daily: pd.DataFrame, upto: int | None = None) -> go.Figure:
    view = daily if upto is None else daily.iloc[: upto + 1]

    running_peak = view["Capital"].cummax()
    drawdown_pct = ((view["Capital"] / running_peak) - 1) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=drawdown_pct,
            mode="lines",
            name="Portfolio drawdown %",
            fill="tozeroy",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Portfolio drawdown / underwater curve",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=430,
        hovermode="x unified",
    )

    return fig


def plot_daily_returns_comparison(
    daily: pd.DataFrame,
    starting_capital: float,
    upto: int | None = None,
) -> go.Figure:
    view = daily if upto is None else daily.iloc[: upto + 1]
    view = view.copy()

    agent_return = view["Capital"].pct_change().fillna(
        (view["Capital"].iloc[0] / starting_capital) - 1
    ) * 100
    market_return = view["Close"].pct_change().fillna(0) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=view["Date"],
            y=agent_return,
            name="Agent daily return",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Agent return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=market_return,
            mode="lines",
            name="S&P 500 daily return",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Market return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Daily return comparison: agent vs market",
        xaxis_title="Date",
        yaxis_title="Daily return (%)",
        height=430,
        hovermode="x unified",
    )

    return fig


def plot_return_distribution(daily: pd.DataFrame, starting_capital: float) -> go.Figure:
    view = daily.copy()

    agent_return = view["Capital"].pct_change().fillna(
        (view["Capital"].iloc[0] / starting_capital) - 1
    ) * 100
    market_return = view["Close"].pct_change().fillna(0) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=agent_return,
            name="Agent daily return",
            opacity=0.75,
            nbinsx=50,
            hovertemplate="Agent return bin: %{x:.2f}%<br>Days: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=market_return,
            name="S&P 500 daily return",
            opacity=0.55,
            nbinsx=50,
            hovertemplate="Market return bin: %{x:.2f}%<br>Days: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Return distribution",
        xaxis_title="Daily return (%)",
        yaxis_title="Trading days",
        barmode="overlay",
        height=430,
    )

    return fig


def plot_rolling_performance(
    daily: pd.DataFrame,
    starting_capital: float,
    window: int = 20,
) -> go.Figure:
    view = daily.copy()

    agent_return = view["Capital"].pct_change().fillna(
        (view["Capital"].iloc[0] / starting_capital) - 1
    )
    market_return = view["Close"].pct_change().fillna(0)

    agent_rolling = ((1 + agent_return).rolling(window).apply(np.prod, raw=True) - 1) * 100
    market_rolling = ((1 + market_return).rolling(window).apply(np.prod, raw=True) - 1) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=agent_rolling,
            mode="lines",
            name=f"Agent {window}-day return",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Agent rolling return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=market_rolling,
            mode="lines",
            name=f"S&P 500 {window}-day return",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Market rolling return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Rolling {window}-trading-day return",
        xaxis_title="Date",
        yaxis_title=f"{window}-day return (%)",
        height=430,
        hovermode="x unified",
    )

    return fig


def plot_reward_and_action(daily: pd.DataFrame, upto: int | None = None) -> go.Figure:
    view = daily if upto is None else daily.iloc[: upto + 1]
    cumulative_reward = view["Reward"].cumsum()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=cumulative_reward,
            mode="lines",
            name="Cumulative reward",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Cumulative reward: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["Raw action"],
            mode="lines",
            name="Raw PPO action",
            yaxis="y2",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Raw action: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot")

    fig.update_layout(
        title="PPO reward and raw action strength",
        xaxis_title="Date",
        yaxis_title="Cumulative reward",
        yaxis2=dict(
            title="Raw action",
            overlaying="y",
            side="right",
            range=[-1.05, 1.05],
        ),
        height=430,
        hovermode="x unified",
    )

    return fig


def plot_position_share(daily: pd.DataFrame) -> go.Figure:
    counts = daily["Position"].value_counts().reset_index()
    counts.columns = ["Position", "Trading days"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Position"],
                values=counts["Trading days"],
                hole=0.45,
                hovertemplate="%{label}<br>Days: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Time spent in each position",
        height=430,
    )

    return fig


def plot_trade_pnl(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if trades.empty:
        fig.update_layout(
            title="Trade P&L by completed segment",
            height=430,
            annotations=[
                dict(
                    text="No completed Long/Short trade segments in this period.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        return fig

    trade_labels = [
        f"{i + 1}. {row['Side']}<br>{row['Entry date']}→{row['Exit date']}"
        for i, row in trades.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=trade_labels,
            y=trades["P&L ($)"],
            name="Trade P&L",
            customdata=np.c_[
                trades["Days held"],
                trades["Entry exposure"],
                trades["P&L (%)"],
            ],
            hovertemplate=(
                "%{x}<br>"
                "P&L: $%{y:,.2f}<br>"
                "P&L %%: %{customdata[2]:.2f}%<br>"
                "Days held: %{customdata[0]}<br>"
                "Entry exposure: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Trade P&L by completed segment",
        xaxis_title="Trade segment",
        yaxis_title="P&L ($)",
        height=430,
    )

    return fig


def plot_capital_vs_market_scatter(
    daily: pd.DataFrame,
    starting_capital: float,
) -> go.Figure:
    view = daily.copy()

    agent_return = view["Capital"].pct_change().fillna(
        (view["Capital"].iloc[0] / starting_capital) - 1
    ) * 100
    market_return = view["Close"].pct_change().fillna(0) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=market_return,
            y=agent_return,
            mode="markers",
            name="Trading days",
            customdata=np.c_[view["Date"].dt.strftime("%Y-%m-%d"), view["Position"], view["Exposure"] * 100],
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Market return: %{x:.2f}%<br>"
                "Agent return: %{y:.2f}%<br>"
                "Position: %{customdata[1]}<br>"
                "Exposure: %{customdata[2]:.1f}%<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=0, line_dash="dot")

    fig.update_layout(
        title="Agent daily return vs market daily return",
        xaxis_title="S&P 500 daily return (%)",
        yaxis_title="Agent daily return (%)",
        height=430,
    )

    return fig


def show_current_step(daily: pd.DataFrame, idx: int, starting_capital: float) -> None:
    row = daily.iloc[idx]

    previous_capital = starting_capital if idx == 0 else float(daily.iloc[idx - 1]["Capital"])
    daily_change = float(row["Capital"] - previous_capital)
    daily_change_pct = (daily_change / previous_capital) * 100 if previous_capital else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Replay date", row["Date"].strftime("%Y-%m-%d"))
    c2.metric("Agent position", row["Position"])
    c3.metric("Exposure", f"{row['Exposure'] * 100:.1f}%")
    c4.metric("Capital", f"${row['Capital']:,.2f}", f"{daily_change:+,.2f}")
    c5.metric("Step return", f"{daily_change_pct:+.2f}%")

    with st.expander("Current step details", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Date": row["Date"].strftime("%Y-%m-%d"),
                        "Close": round(float(row["Close"]), 2),
                        "Raw action": round(float(row["Raw action"]), 4),
                        "Position": row["Position"],
                        "Exposure %": round(float(row["Exposure"]) * 100, 2),
                        "Reward": round(float(row["Reward"]), 4),
                        "Capital": round(float(row["Capital"]), 2),
                    }
                ]
            ),
            use_container_width=True,
        )


# =============================================================================
# 6. STREAMLIT UI
# =============================================================================
st.set_page_config(
    page_title="PPO Trading Agent Demo",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Local PPO Trading Agent Demo")

with st.sidebar:
    st.header("Required local files")

    file_checks = {
        "data/SP500.csv": HISTORICAL_CSV,
        "data/sp500_2025_20260.csv": RECENT_CSV,
        "artifacts/best_model.zip OR artifacts/ppo_quant_bot_final.zip": BEST_MODEL if BEST_MODEL.exists() else FINAL_MODEL,
        "artifacts/vec_normalize_stats.pkl": SCALER_PATH,
    }

    all_files_ready = True
    for label, path in file_checks.items():
        ok = path.exists()
        if ok:
            st.success(label)
            st.caption(f"Found: {path}")
        else:
            all_files_ready = False
            st.error(label)
            if " OR " not in label:
                with st.expander(f"Checked paths for {label}", expanded=False):
                    for checked in checked_paths(label):
                        st.code(str(checked), language="text")
            else:
                with st.expander("Checked model paths", expanded=False):
                    for checked in checked_paths("artifacts/best_model.zip"):
                        st.code(str(checked), language="text")
                    for checked in checked_paths("artifacts/ppo_quant_bot_final.zip"):
                        st.code(str(checked), language="text")

    st.divider()
    st.write("Expected project folder:")
    st.code(
        """trading_agent/
├── app.py
├── requirements.txt
├── data/
│   ├── SP500.csv
│   └── sp500_2025_20260.csv
└── artifacts/
    ├── best_model.zip
    └── vec_normalize_stats.pkl""",
        language="text",
    )

    with st.expander("Path debug", expanded=False):
        st.write("App base directory:")
        st.code(str(BASE_DIR), language="text")
        st.write("Terminal working directory:")
        st.code(str(Path.cwd()), language="text")
        st.write("Candidate base folders:")
        for base in candidate_base_dirs():
            st.code(str(base), language="text")

if not all_files_ready:
    st.error("Some required files were not found. Check the exact paths in the sidebar. The files must be inside a `data` folder and an `artifacts` folder near `app.py`, or inside the folder where you launched Streamlit.")
    st.stop()

try:
    market_df = load_and_prepare_market_data()
except Exception as exc:
    st.exception(exc)
    st.stop()

st.success(
    f"Loaded processed market data: {len(market_df):,} trading rows "
    f"from {market_df.index.min().date()} to {market_df.index.max().date()}."
)

with st.expander("Show rebuilt feature columns", expanded=False):
    st.write("These are the columns rebuilt from raw CSV before running the saved PPO model:")
    st.code("\n".join(market_df.columns.tolist()), language="text")

date_list = market_df.index.to_pydatetime().tolist()

default_start = pd.Timestamp("2022-01-01")
if default_start < market_df.index.min() or default_start > market_df.index.max():
    default_start = market_df.index[max(WINDOW_SIZE, len(market_df) - 252)]

default_end = market_df.index.max()

st.header("1. Choose backtest settings")

c1, c2, c3 = st.columns(3)
with c1:
    start_date = st.date_input(
        "Start date",
        value=default_start.date(),
        min_value=market_df.index.min().date(),
        max_value=market_df.index.max().date(),
    )

with c2:
    end_date = st.date_input(
        "End date",
        value=default_end.date(),
        min_value=market_df.index.min().date(),
        max_value=market_df.index.max().date(),
    )

# Minimum starting capital is based on the lowest price in the selected period.
# This makes the demo feel more realistic: the user must have enough money
# to afford at least one unit/share at the cheapest close inside the timeline.
selected_period_for_min = market_df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

if selected_period_for_min.empty:
    min_starting_capital = 1.0
    min_price_date_text = "N/A"
else:
    min_price_idx = selected_period_for_min["Close"].idxmin()
    min_starting_capital = float(selected_period_for_min.loc[min_price_idx, "Close"])
    min_price_date_text = min_price_idx.strftime("%Y-%m-%d")

# Streamlit keeps widget values in session_state. When the date range changes,
# the previous capital may become invalid, so update it before drawing the widget.
if (
    "starting_capital_input" not in st.session_state
    or float(st.session_state["starting_capital_input"]) < min_starting_capital
):
    st.session_state["starting_capital_input"] = max(
        float(DEFAULT_CAPITAL),
        min_starting_capital,
    )

with c3:
    starting_capital = st.number_input(
        "Starting capital",
        min_value=min_starting_capital,
        step=100.0,
        format="%.2f",
        key="starting_capital_input",
        help=(
            "Minimum capital is automatically set to the lowest S&P 500 close "
            "inside the selected timeline."
        ),
    )

st.caption(
    f"Minimum starting capital for this selected timeline: "
    f"${min_starting_capital:,.2f} "
    f"(lowest close on {min_price_date_text})."
)

run_clicked = st.button("Run agent backtest", type="primary")

if run_clicked:
    with st.spinner("Running PPO agent on the selected period..."):
        try:
            summary, daily, trades = run_backtest(
                full_df=market_df,
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date),
                starting_capital=float(starting_capital),
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.session_state["summary"] = summary
    st.session_state["daily"] = daily
    st.session_state["trades"] = trades

if (
    "summary" not in st.session_state
    or "daily" not in st.session_state
    or "trades" not in st.session_state
):
    st.info("Choose a timeframe and starting capital, then press **Run agent backtest**.")
    st.stop()

summary = st.session_state["summary"]
daily = st.session_state["daily"]
trades = st.session_state["trades"]

st.header("2. Final result")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Starting money", f"${summary['starting_capital']:,.2f}")
m2.metric("Final money", f"${summary['final_capital']:,.2f}", f"{summary['profit_usd']:+,.2f}")
m3.metric("Agent return", f"{summary['profit_pct']:+.2f}%")
m4.metric("Buy & Hold return", f"{summary['buy_hold_pct']:+.2f}%")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%")
m6.metric("Annualized return", f"{summary['annualized_return']:+.2f}%")
m7.metric("Trade segments", f"{summary['completed_trade_count']}")
m8.metric("Win rate", f"{summary['win_rate']:.1f}%")

st.caption(
    f"Model used: {summary['model_path']} | "
    f"Backtest period: {summary['actual_start'].date()} → {summary['actual_end'].date()}"
)

st.header("3. Visualize how the agent traded")

tab_full, tab_risk, tab_agent, tab_trades, tab_replay, tab_table = st.tabs(
    [
        "Full backtest charts",
        "Risk / return analysis",
        "Agent behavior",
        "Trade analysis",
        "Replay / process view",
        "Daily data",
    ]
)

with tab_full:
    show_action_labels_full = st.checkbox(
        "Show profit/loss labels at each action point",
        value=True,
        key="show_action_labels_full",
        help="Display annotations for each state change, including profit/loss and the next action.",
    )

    st.plotly_chart(
        plot_price_actions(
            daily,
            float(summary["starting_capital"]),
            show_action_labels=show_action_labels_full,
        ),
        use_container_width=True,
        key="full_price_actions_chart",
    )
    st.plotly_chart(
        plot_equity(daily, float(summary["starting_capital"])),
        use_container_width=True,
        key="full_equity_chart",
    )
    st.plotly_chart(
        plot_position_exposure(daily),
        use_container_width=True,
        key="full_position_exposure_chart",
    )

with tab_risk:
    st.subheader("Risk and return analysis")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_drawdown_only(daily),
            use_container_width=True,
            key="risk_drawdown_only_chart",
        )
    with right:
        st.plotly_chart(
            plot_return_distribution(daily, float(summary["starting_capital"])),
            use_container_width=True,
            key="risk_return_distribution_chart",
        )

    st.plotly_chart(
        plot_daily_returns_comparison(daily, float(summary["starting_capital"])),
        use_container_width=True,
        key="risk_daily_return_comparison_chart",
    )

    st.plotly_chart(
        plot_rolling_performance(daily, float(summary["starting_capital"]), window=20),
        use_container_width=True,
        key="risk_rolling_performance_chart",
    )


with tab_agent:
    st.subheader("Agent behavior and decision strength")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_reward_and_action(daily),
            use_container_width=True,
            key="agent_reward_action_chart",
        )
    with right:
        st.plotly_chart(
            plot_position_share(daily),
            use_container_width=True,
            key="agent_position_share_chart",
        )

    st.plotly_chart(
        plot_capital_vs_market_scatter(daily, float(summary["starting_capital"])),
        use_container_width=True,
        key="agent_market_scatter_chart",
    )


with tab_trades:
    st.subheader("Trade-level analysis")
    st.plotly_chart(
        plot_trade_pnl(trades),
        use_container_width=True,
        key="trade_pnl_chart",
    )

    if trades.empty:
        st.info("The agent did not complete any Long/Short trade segment in this period.")
    else:
        win_trades = int((trades["P&L ($)"] > 0).sum())
        loss_trades = int((trades["P&L ($)"] < 0).sum())
        avg_pnl = float(trades["P&L ($)"].mean())
        best_trade = float(trades["P&L ($)"].max())
        worst_trade = float(trades["P&L ($)"].min())

        a, b, c, d, e = st.columns(5)
        a.metric("Winning trades", win_trades)
        b.metric("Losing trades", loss_trades)
        c.metric("Average trade P&L", f"${avg_pnl:,.2f}")
        d.metric("Best trade", f"${best_trade:,.2f}")
        e.metric("Worst trade", f"${worst_trade:,.2f}")

        st.dataframe(trades, use_container_width=True)


with tab_replay:
    st.write(
        "Move the slider to inspect what the agent had done up to a specific trading day."
    )

    replay_idx = st.slider(
        "Replay step",
        min_value=0,
        max_value=len(daily) - 1,
        value=len(daily) - 1,
        step=1,
    )

    show_current_step(daily, replay_idx, float(summary["starting_capital"]))

    show_action_labels_replay = st.checkbox(
        "Show profit/loss labels in replay chart",
        value=True,
        key="show_action_labels_replay",
        help="Show trade result annotations up to the selected replay step.",
    )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            plot_price_actions(
                daily,
                float(summary["starting_capital"]),
                upto=replay_idx,
                show_action_labels=show_action_labels_replay,
            ),
            use_container_width=True,
            key="replay_price_actions_chart",
        )
    with right:
        st.plotly_chart(
            plot_equity(daily, float(summary["starting_capital"]), upto=replay_idx),
            use_container_width=True,
            key="replay_equity_chart",
        )

    st.plotly_chart(
        plot_position_exposure(daily, upto=replay_idx),
        use_container_width=True,
        key="replay_position_exposure_chart",
    )

    st.plotly_chart(
        plot_reward_and_action(daily, upto=replay_idx),
        use_container_width=True,
        key="replay_reward_action_chart",
    )

    replay_rows = daily.iloc[: replay_idx + 1].tail(15).copy()
    replay_rows["Date"] = replay_rows["Date"].dt.strftime("%Y-%m-%d")
    st.subheader("Latest replay rows")
    st.dataframe(replay_rows, use_container_width=True)

with tab_table:
    st.subheader("Completed Long/Short segments")
    if trades.empty:
        st.info("The agent did not complete any Long/Short trade segment in this period.")
    else:
        st.dataframe(trades, use_container_width=True)

    st.subheader("Daily agent log")
    daily_show = daily.copy()
    daily_show["Date"] = daily_show["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(daily_show, use_container_width=True)

    csv = daily_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download daily backtest CSV",
        data=csv,
        file_name="ppo_backtest_daily_log.csv",
        mime="text/csv",
    )
