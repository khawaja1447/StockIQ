"""
Bridge between StockIQ and the Trading-Bot DQN signal engine.

Uses importlib.util to load the engine directly from its file path,
avoiding any 'src' package namespace collision between the two projects.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import streamlit as st

# Resolve Trading-Bot location relative to this file:
#   StockIQ/src/dqn_signals.py → ../../ → GitHub/ → Trading-Bot/
_TRADING_BOT = Path(__file__).parent.parent.parent / "Trading-Bot"
_ENGINE_FILE = _TRADING_BOT / "src" / "agent" / "signal_engine.py"
_CHECKPOINTS = _TRADING_BOT / "checkpoints"


def _load_engine():
    if not _ENGINE_FILE.exists():
        return None, f"signal_engine.py not found — expected at {_ENGINE_FILE}"
    try:
        spec = importlib.util.spec_from_file_location("_dqn_engine", _ENGINE_FILE)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, str(e)


_engine, ENGINE_ERROR = _load_engine()
SIGNAL_ENGINE_AVAILABLE = _engine is not None


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def list_checkpoints() -> list[str]:
    """Return all best.pt paths, newest first."""
    if not _CHECKPOINTS.exists():
        return []
    return sorted(
        [str(p) for p in _CHECKPOINTS.glob("*/best.pt")],
        key=os.path.getmtime,
        reverse=True,
    )


def default_checkpoint() -> str:
    pts = list_checkpoints()
    return pts[0] if pts else ""


def checkpoint_label(path: str) -> str:
    """Turn a full path into a readable label like 'BTC_DQN_1h_20260605'."""
    return Path(path).parent.name if path else "none"


# ── Cached inference ───────────────────────────────────────────────────────────

@st.cache_data(ttl=20, show_spinner=False)
def cached_btc_signal(checkpoint_path: str, interval: str) -> dict | None:
    if not SIGNAL_ENGINE_AVAILABLE or not checkpoint_path:
        return None
    try:
        return _engine.get_btc_signal(checkpoint_path, interval)
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=20, show_spinner=False)
def cached_signal_history(checkpoint_path: str, interval: str, n: int = 60):
    if not SIGNAL_ENGINE_AVAILABLE or not checkpoint_path:
        return None
    try:
        return _engine.get_signal_history(checkpoint_path, interval, n_candles=n)
    except Exception:
        return None
