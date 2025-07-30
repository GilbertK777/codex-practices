import pandas as pd
from config.config import CFG

class Strategy:
    """
    진입/청산 신호 생성 (규칙 기반 + ML 확률 혼합).
    """
    @staticmethod
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame에 규칙기반 신호 열과 최종 long/short/exit 신호 열을 추가하여 리턴.
        """
        df = df.copy()
        # Long 진입 규칙: 상승 추세 & 과매도 상태 & MACD 골든크로스 (단기 TF 기준)
        df["rule_long"] = (
            (df["ema_fast"] > df["ema_slow"]) &
            (df["rsi"] < 40) &
            (df["macd"] > df["macd_sig"])
        )
        # Short 진입 규칙: 하락 추세 & 과매수 상태 & MACD 데드크로스
        df["rule_short"] = (
            (df["ema_fast"] < df["ema_slow"]) &
            (df["rsi"] > 60) &
            (df["macd"] < df["macd_sig"])
        )
        # ML 확률 결합: 규칙 만족 + 확률 임계값 넘어야 최종 신호 True
        df["long"]  = df["rule_long"]  & (df["prob_up"] > CFG.BUY_TH)
        df["short"] = df["rule_short"] & (df["prob_up"] < CFG.SHORT_TH)
        # Long 청산 조건: 확률 급락 또는 RSI 과열 또는 MACD 데드크로스
        df["exit_l"] = (
            (df["prob_up"] < CFG.SELL_TH) |
            (df["rsi"] > 70) |
            (df["macd"] < df["macd_sig"])
        )
        # Short 청산 조건: 확률 급등 또는 RSI 과매도 또는 MACD 골든크로스
        df["exit_s"] = (
            (df["prob_up"] > CFG.BUY_TH) |
            (df["rsi"] < 30) |
            (df["macd"] > df["macd_sig"])
        )
        return df
