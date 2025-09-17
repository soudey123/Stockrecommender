# main_gpt.py — flat Replit-friendly entry (no relative imports)
import os
import streamlit as st
import pandas as pd
import yfinance as yf

from gpt_integration import get_llm_scores
from gpt_recommender import blend_scores
from email_utils import send_email

st.set_page_config(page_title="Stock Recommender (Hybrid GPT-5)", layout="wide")
st.title("📈 Stock Recommender — Classical vs GPT-5 vs Hybrid")
st.caption("Educational demo. Not financial advice.")

# -------------------------------
# Sidebar controls
# -------------------------------
mode = st.sidebar.radio("Engine", ["Classical", "GPT-5", "Hybrid"], index=2)
horizon = st.sidebar.number_input("Horizon (trading days)", 1, 30, 5)
max_picks = st.sidebar.number_input("Max picks", 1, 50, 10)
blend_w = st.sidebar.slider("Blend → GPT-5 (Hybrid only)", 0.0, 1.0, 0.35, 0.05) if mode == "Hybrid" else 0.0
tickers_csv = st.sidebar.text_input("Tickers (CSV)", "AAPL,MSFT,TSLA,COST,GOOGL,AMZN")
period = st.sidebar.selectbox("Data window", ["6mo","1y","2y"], index=1)

# -------------------------------
# Simple features from yfinance
# -------------------------------
def fetch_features(tickers, period="1y"):
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        closes = data["Close"]
        vols = data["Volume"]
    else:
        closes = data
        vols = None

    out = []
    for t in tickers:
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        if len(s) < 25:  # need at least 20d window
            continue
        price = float(s.iloc[-1])
        ret_5d = float((s.iloc[-1] / s.iloc[-6]) - 1) if len(s) > 6 else 0.0
        ret_20d = float((s.iloc[-1] / s.iloc[-21]) - 1) if len(s) > 21 else 0.0
        vol_20d = float(s.pct_change().dropna().rolling(20).std().iloc[-1]) if len(s) > 21 else 0.0
        avg_vol_20d = float(vols[t].rolling(20).mean().iloc[-1]) if isinstance(vols, pd.DataFrame) and t in vols.columns and len(vols[t]) > 20 else 0.0
        out.append({
            "ticker": t, "price": price, "avg_vol_20d": avg_vol_20d,
            "ret_5d": ret_5d, "ret_20d": ret_20d, "vol_20d": vol_20d,
            # optional fields expected by gpt_integration (None is fine):
            "rsi_14": None, "macd": None, "sector": None,
        })
    df = pd.DataFrame(out).set_index("ticker")
    return df

tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
features_df = fetch_features(tickers, period=period)

st.subheader("Feature Snapshot")
if features_df.empty:
    st.warning("No data. Try a different ticker list or period.")
    st.stop()
st.dataframe(features_df, use_container_width=True)

# -------------------------------
# Baseline numeric score (very simple)
# -------------------------------
def compute_numeric_baseline(df: pd.DataFrame) -> dict:
    if "ret_5d" not in df.columns:  # safety
        return {t: 0.5 for t in df.index}
    s = df["ret_5d"].fillna(0.0)
    if s.max() == s.min():
        return {t: 0.5 for t in df.index}
    norm = (s - s.min()) / (s.max() - s.min())
    return {t: float(norm.loc[t]) for t in df.index}

numeric_scores = compute_numeric_baseline(features_df)

# -------------------------------
# GPT-5 path (optional)
# -------------------------------
llm_scores = {}
if mode in ("GPT-5", "Hybrid"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not set; GPT-5 disabled. Set it in Secrets to use GPT-5 modes.")
        mode = "Classical"
    else:
        try:
            llm_scores = get_llm_scores(features_df, horizon_days=int(horizon), max_picks=int(max_picks))
        except Exception as e:
            st.error(f"GPT-5 reranker failed: {e}")
            mode = "Classical"

# -------------------------------
# Combine & show
# -------------------------------
if mode == "Classical":
    final_scores = numeric_scores
elif mode == "GPT-5":
    final_scores = {t: llm_scores.get(t, 0.0) for t in features_df.index}
else:
    final_scores = blend_scores(numeric_scores, llm_scores or {}, float(blend_w))

ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[: int(max_picks)]
out_rows = [{"Ticker": t, "Score": round(s, 3)} for t, s in ranked]
st.subheader("Recommendations")
st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

# -------------------------------
# Email panel
# -------------------------------
with st.expander("Email these recommendations"):
    default_subject = f"Stock Recommender — {mode} picks"
    html_rows = "".join([f"<tr><td>{t}</td><td>{s:.3f}</td></tr>" for t, s in ranked])
    default_body = f"""
    <h3>Stock Recommender — {mode} picks</h3>
    <p>Horizon: {int(horizon)} days · Max picks: {int(max_picks)} · Blend: {float(blend_w):.2f}</p>
    <table border='1' cellpadding='6' cellspacing='0'>
      <tr><th>Ticker</th><th>Score</th></tr>
      {html_rows}
    </table>
    <p style='color:#666'>Educational demo; not financial advice.</p>
    """
    subject = st.text_input("Subject", value=default_subject, key="email_subject")
    body = st.text_area("Body (HTML allowed)", value=default_body, height=220, key="email_body")
    recipients_raw = st.text_input("Recipients (comma-separated emails)", value=os.environ.get("EMAIL_TO",""))
    if st.button("Send Email"):
        try:
            recipients = [x.strip() for x in recipients_raw.split(",") if x.strip()]
            if not recipients:
                st.error("Please provide at least one recipient email.")
            else:
                send_email(subject, body, recipients)
                st.success(f"Email sent to {', '.join(recipients)}")
        except Exception as e:
            st.error(f"Failed to send email: {e}")
