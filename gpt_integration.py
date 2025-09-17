import os
import pandas as pd
from typing import Dict, List

# Absolute imports for "flat" script mode in Replit (no package context)
from gpt_recommender import TickerSnapshot, rerank_with_gpt5, blend_scores


def df_to_snapshots(features_df: pd.DataFrame) -> List[TickerSnapshot]:
    snaps = []
    for t, row in features_df.iterrows():
        def as_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default
        snaps.append(TickerSnapshot(
            ticker=str(t),
            price=as_float(row.get("price", 0.0)),
            avg_vol_20d=as_float(row.get("avg_vol_20d", row.get("avg_volume_20d", 0.0))),
            ret_5d=as_float(row.get("ret_5d", 0.0)),
            ret_20d=as_float(row.get("ret_20d", 0.0)),
            vol_20d=as_float(row.get("vol_20d", row.get("volatility_20d", 0.0))),
            rsi_14=None if pd.isna(row.get("rsi_14", None)) else as_float(row.get("rsi_14", None), None),
            macd=None if pd.isna(row.get("macd", None)) else as_float(row.get("macd", None), None),
            sector=None if pd.isna(row.get("sector", None)) else str(row.get("sector"))
        ))
    return snaps

def get_llm_scores(features_df: pd.DataFrame, horizon_days: int, max_picks: int) -> Dict[str, float]:
    snaps = df_to_snapshots(features_df)
    res = rerank_with_gpt5(snaps, horizon_days=horizon_days, max_picks=max_picks)
    return {p.ticker: p.score for p in res.picks}
