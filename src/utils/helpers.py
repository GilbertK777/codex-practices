import logging
import pandas as pd
import ta
from telegram.ext import Updater
from config.config import CFG

def tg(msg: str) -> None:
    """텔레그램 알림 전송 (오류 발생 시 무시하고 진행)"""
    try:
        Updater(CFG.TG_TOKEN).bot.send_message(chat_id=CFG.TG_CHAT, text=msg)
        logging.info(f"Telegram ▶ {msg}")
    except Exception as e:
        logging.error(f"Telegram 전송 오류: {e}")

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    가격 DataFrame에 기술적 지표 열 추가 (EMA12/26, RSI14, ATR14, MACD, Bollinger Bands).
    반환: 지표 컬럼이 포함된 DataFrame (NaN 제거됨).
    """
    df = df.copy()
    # 이동평균 (추세 지표)
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 12).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 26).ema_indicator()
    # RSI (모멘텀 지표)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    # ATR (변동성 지표)
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    # MACD (추세 지표)
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    # 볼린저 밴드 (상단/하단 밴드)
    bb = ta.volatility.BollingerBands(df["close"], 20, 2)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_high"] = bb.bollinger_hband()
    return df.dropna()
