import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from src.bot.trading_bot import TradingBot
from config.config import CFG

def ui_dashboard(bot: TradingBot):
    """실시간 대시보드 실행 (Streamlit)"""
    st_autorefresh(interval=60_000, key="refresh")  # 60초마다 자동 새로고침
    st.set_page_config(page_title="MTF‑XGB Futures Bot", layout="wide")
    st.title("🚀 Multi‑TF + XGB  Futures Trading Bot Dashboard")

    # 사이드바 정보 출력
    bal = bot.order.balance
    mode_label = "PAPER" if bot.order.paper else "LIVE"
    st.sidebar.header(f"{mode_label} Mode | Balance: ${bal:,.0f} | Leverage: {CFG.LEVERAGE}×")
    # 모델 마지막 학습 시각 (UTC)
    st.sidebar.info(f"Model Last Trained: {bot.model.t_last_train:%H:%M UTC, %m-%d}")
    # 현재 오픈 포지션 상태
    pos = bot.order.pos
    if pos:
        st.sidebar.success(f"OPEN {pos['side'].upper()} @ {pos['entry']:.1f}")
    else:
        st.sidebar.warning("No open position")

    # 15분봉 차트 (캔들 + EMA12/26)
    df = bot.get_df()
    if not df.empty:
        st.subheader("15m OHLC Candles + EMA(12,26)")
        fig = go.Figure([
            go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                           low=df["low"], close=df["close"], name="15m Candles"),
            go.Scatter(x=df.index, y=df["ema_fast"], name="EMA12",
                       line=dict(color="blue")),
            go.Scatter(x=df.index, y=df["ema_slow"], name="EMA26",
                       line=dict(color="red")),
        ])
        st.plotly_chart(fig, use_container_width=True)

        # RSI & MACD 지표 차트
        st.subheader("RSI (14) & MACD (12/26/9)")
        fig2 = go.Figure([
            go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                       line=dict(color="purple")),
            go.Scatter(x=df.index, y=df["macd"], name="MACD",
                       line=dict(color="green")),
            go.Scatter(x=df.index, y=df["macd_sig"], name="Signal",
                       line=dict(color="orange")),
        ])
        st.plotly_chart(fig2, use_container_width=True)

        # 최근 신호 표 (확률 및 Long/Short/Exit flags)
        st.subheader("Latest Signals (last 5 intervals)")
        st.dataframe(df[["close", "prob_up", "long", "short", "exit_l", "exit_s"]].tail(5))

    # 거래 내역 및 잔고 곡선
    if bot.order.trades:
        hist = pd.DataFrame(bot.order.trades)
        st.subheader("Trade History (last 20)")
        st.dataframe(hist.tail(20))
        st.subheader("Balance Curve")
        fig3 = go.Figure([
            go.Scatter(x=hist["time"], y=hist["bal"], mode="lines+markers", name="Balance")
        ])
        st.plotly_chart(fig3, use_container_width=True)
