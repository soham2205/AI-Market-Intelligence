from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Market Intelligence", page_icon="📈", layout="wide")

CURATED = Path("data/curated")


@st.cache_data(ttl=600)
def load_prices() -> pd.DataFrame:
    frames = []
    for p in sorted((CURATED / "prices").glob("ticker=*.parquet")):
        t = p.stem.removeprefix("ticker=")
        df = pd.read_parquet(p)
        df["ticker"] = t
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


@st.cache_data(ttl=600)
def load_sentiment() -> pd.DataFrame | None:
    path = CURATED / "sentiment_daily.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


prices = load_prices()
sentiment = load_sentiment()

tickers = sorted(prices["ticker"].unique())
ticker = st.sidebar.selectbox("Ticker", tickers)
start = st.sidebar.date_input("Start", prices["date"].min().date())
end = st.sidebar.date_input("End", prices["date"].max().date())

mask = (
    (prices["ticker"] == ticker)
    & (prices["date"] >= pd.Timestamp(start))
    & (prices["date"] <= pd.Timestamp(end))
)
df = prices[mask]

st.title(f"{ticker} — Price & Sentiment")

if len(df):
    last = df.iloc[-1]
    hist = prices[prices["ticker"] == ticker]
    chg = last["close"] / hist["close"].iloc[-2] - 1 if len(hist) > 1 else 0
    st.metric("Close", f"{last['close']:.2f}")
    st.metric("Last change", f"{chg:+.2%}")

chart_df = df.set_index("date")[["close"]]
st.line_chart(chart_df, height=320)
st.area_chart(df.set_index("date")[["volume"]], height=140)

if sentiment is not None:
    sent = sentiment[
        (sentiment["ticker"] == ticker)
        & (sentiment["trade_date"] >= pd.Timestamp(start))
        & (sentiment["trade_date"] <= pd.Timestamp(end))
    ]
    if len(sent):
        st.subheader("Daily news sentiment (lag-safe)")
        st.bar_chart(sent.set_index("trade_date")["sent_mean"], height=180)
    else:
        st.caption("No sentiment coverage in this window.")
else:
    st.caption("Run `ingest-news` to build the sentiment table.")

st.divider()
st.caption("Educational research tool — not investment advice.")
