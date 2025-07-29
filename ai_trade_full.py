#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
  ✅  Multi‑Timeframe + XGBoost Futures Trading Bot  (v2.3)
------------------------------------------------------------------------
  · 지원 거래소 : Binance USD‑M Futures, **Bybit USDT Perpetual**  (ccxt)
        - 격리마진 (isolated) 사용, 포지션 모드: 한쪽만 (hedge mode off)
  · 타임프레임 : 15m  (기본)  + 1h / 4h 지표 병합
  · 전략 요약 :
        └ 규칙 필터  : EMA(12/26) 추세, RSI(14) 과매수/과매도, MACD 골든/데드크로스  (+ 상위 TF 동향)
        └ ML 필터    : XGBoost 모델 상승 확률 (SMOTE로 불균형 보정 후 학습)
        └ 진입 신호  : rule 기준 충족 시, Long/Short 확률 임계값 돌파 시 포지션 진입
        └ 포지션 관리: 진입 후 ➜ 즉시 지정가 청산 주문(TP/SL) 2개 서버 등록
                      (타입: TAKE_PROFIT_MARKET / STOP_MARKET, reduce‑only=True)
  · 위험 관리 :
        ▸ **증거금 고정 거래** 옵션 – POSITION_MARGIN 사용 시 거래 당 일정 USDT만 사용 (레버리지 적용)
        ▸ 실시간 펀딩비용 PnL 보정 (Binance, Bybit는 미지원시 0 처리)
        ▸ 연속 손실 제한 및 자동 일시정지 기능 (MAX_CONSECUTIVE_LOSSES, PAUSE_HR)
  · 기타 기능 :
        ▸ Paper trading 모드 / Live 모드 스위치 (.env 설정)
        ▸ Streamlit 대시보드 제공 (실시간 차트, 백테스트 트리거 버튼 등)
========================================================================
"""

# ─────────────────────────────── Imports ───────────────────────────────
import os, sys, time, logging, threading, abc, math
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 데이터 처리 & 머신러닝
import numpy as np
import pandas as pd
import ta                                          # 기술적 지표 계산 라이브러리
from xgboost import XGBClassifier                  # XGBoost 모델
from imblearn.over_sampling import SMOTE           # SMOTE 오버샘플링 (클래스 불균형 해결)
from sklearn.model_selection import GridSearchCV   # 하이퍼파라미터 튜닝
import joblib                                      # 모델 저장/로드

# 거래소 API & 외부 연동
import ccxt                                        # CCXT - 다중 거래소 지원
from dotenv import load_dotenv                     # .env 환경변수 로더
from telegram.ext import Updater                   # 텔레그램 알림 (봇)

# Streamlit UI
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# 로깅 핸들러
from logging.handlers import RotatingFileHandler, StreamHandler

# ─────────────────────────── 0) 환경 설정 ─────────────────────────────
load_dotenv()  # ➊ .env 파일에서 환경변수 불러오기

# ➋ 필수 키 존재 여부 검사
REQUIRED_ENV = ["BINANCE_API_KEY", "BINANCE_API_SECRET",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    sys.exit(f"[FATAL] .env에 다음 필수 변수가 없습니다: {missing}")

class CFG:
    """전역 설정값 (상수 모음) – C# static class처럼 사용"""
    # ── API & 계정 정보 ──
    API_KEY      = os.getenv("BINANCE_API_KEY")
    API_SECRET   = os.getenv("BINANCE_API_SECRET")
    TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
    TG_CHAT      = os.getenv("TELEGRAM_CHAT_ID")
    EXCHANGE_NAME= os.getenv("EXCHANGE", "BINANCE").upper()  # 추가: 기본 BINANCE, 옵션으로 BYBIT
    
    # ── 트레이딩 기본 파라미터 ──
    SYMBOL       = os.getenv("SYMBOL", "BTC/USDT")
    LEVERAGE     = int(os.getenv("LEVERAGE", 3))               # 기본 레버리지 3x
    POS_SIZE     = float(os.getenv("POSITION_SIZE", 0.001))    # 포지션 크기 (코인 수량, e.g., BTC 0.001)
    MARGIN_PER_TRADE = float(os.getenv("POSITION_MARGIN", 0))  # 추가: 거래당 사용할 증거금 (USDT)
    TEST_MODE    = os.getenv("TEST_MODE", "true").lower() == "true"  # Paper trading 여부
    INIT_BAL     = float(os.getenv("INIT_BALANCE", 10_000))    # Paper 모드 시작 USDT 잔고
    ISOLATED     = True    # 격리마진 (Binance/Bybit 모두 격리모드 사용)
    
    # ── 전략/진입 임계값 파라미터 ──
    BUY_TH       = float(os.getenv("PROB_BUY_TH",   0.65))     # Long 진입 확률 임계값 (상승확률 65% 이상)
    SELL_TH      = float(os.getenv("PROB_SELL_TH",  0.40))     # Long 청산 확률 임계값 (상승확률 40% 미만)
    SHORT_TH     = float(os.getenv("PROB_SHORT_TH", 0.35))     # Short 진입 임계값 (상승확률 35% 미만)
    SL_PCT       = float(os.getenv("STOP_LOSS_PCT", 0.02))     # 손절 기준 2%
    TP_PCT       = float(os.getenv("TP_PCT",       0.05))      # 목표이익 5%
    SLIP_PCT     = float(os.getenv("SLIPPAGE_PCT", 0.0005))    # 슬리피지 0.05%
    
    # ── 시스템 주기 설정 ──
    SLEEP_SEC    = int(os.getenv("SLEEP_SEC", 60))             # 메인 루프 간격 (초)
    RETRAIN_HR   = int(os.getenv("TRAIN_HR", 24))              # 모델 재학습 주기 (시간)
    GRID_DAYS    = int(os.getenv("GRIDSEARCH_INTERVAL_DAYS", 7))  # GridSearch 주기 (일)
    
    # ── 리스크 관리 한도 ──
    MAX_QTY      = float(os.getenv("MAX_POSITION_LIMIT", 0.02))  # 최대 포지션 코인 수량 (예: BTC 0.02)
    MAX_LOSS     = int(os.getenv("MAX_CONSECUTIVE_LOSSES", 3))   # 연속 손실 허용 횟수
    PAUSE_HR     = int(os.getenv("PAUSE_HR", 1))                 # 연속 손실시 휴지기간 (시간)
    
    # ── 경로 설정 ──
    DATA_DIR     = Path(os.getenv("DATA_DIR",  "data"))
    MODEL_DIR    = Path(os.getenv("MODEL_DIR", "models"))
    MODEL_FP     = MODEL_DIR / f"xgb_{SYMBOL.replace('/', '_')}_fut.joblib"
    
    # ── 설정값 검증 및 디렉토리 생성 ──
    @staticmethod
    def validate() -> None:
        if CFG.POS_SIZE <= 0 and CFG.MARGIN_PER_TRADE <= 0:
            raise ValueError("POSITION_SIZE 또는 POSITION_MARGIN 중 하나는 0보다 커야 합니다.")
        if CFG.INIT_BAL <= 0:
            raise ValueError("INIT_BALANCE > 0 필요")
        if CFG.LEVERAGE < 1:
            raise ValueError("LEVERAGE >= 1 필요")
        # 데이터/모델 디렉토리 생성
        CFG.DATA_DIR.mkdir(exist_ok=True)
        CFG.MODEL_DIR.mkdir(exist_ok=True)

# 설정값 확인
CFG.validate()

# ─────────────────────────── 1) 로깅 설정 ─────────────────────────────
log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    handlers=[
        RotatingFileHandler("bot_futures.log", maxBytes=5_000_000, backupCount=3),
        StreamHandler(sys.stdout)  # 표준출력에도 로그 출력
    ]
)

# ─────────────────────────── 2) 유틸리티 함수 ─────────────────────────
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

# ─────────────────────── 3) ExchangeClient 추상클래스 ───────────────────────
class ExchangeClient(abc.ABC):
    """거래소 API 인터페이스 (파생 클래스에서 구체 구현)"""
    @abc.abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, since=None, limit=500):
        ...
    @abc.abstractmethod
    def create_market_order(self, symbol: str, side: str, qty: float):
        ...
    @abc.abstractmethod
    def set_leverage(self, symbol: str, leverage: int, isolated: bool):
        ...
    @abc.abstractmethod
    def create_exit_order(self, symbol: str, side: str, qty: float,
                          stop_price: float, tp: bool = True):
        ...
    @abc.abstractmethod
    def fetch_funding_rate(self, symbol: str) -> float:
        ...

# ──────────────────────── 3-1) Binance Futures 구현 ────────────────────────
class BinanceFutures(ExchangeClient):
    """Binance USD-M 선물 거래소 구현 클래스"""
    def __init__(self, key: str, secret: str):
        # ccxt Binance 선물 클라이언트 초기화
        self.client = ccxt.binance({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}   # Binance USD-M Futures
        })
        try:
            self.client.fetch_time()  # API 연결 테스트 (ping)
        except Exception as e:
            raise RuntimeError(f"Binance API 연결 실패: {e}")
        logging.info("Binance 선물 API 연결 성공")
    
    def set_leverage(self, symbol: str, leverage: int, isolated: bool):
        """격리(isolated) 마진모드 및 레버리지 설정"""
        try:
            if isolated:
                self.client.set_margin_mode("ISOLATED", symbol)
            self.client.set_leverage(leverage, symbol)
            logging.info(f"[Binance] 레버리지 {leverage}x (isolated={isolated}) 설정 완료")
        except Exception as e:
            raise RuntimeError(f"Binance 레버리지 설정 실패: {e}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, since=None, limit=500):
        return self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    
    def create_market_order(self, symbol: str, side: str, qty: float):
        """
        시장가 주문 실행 (side: 'buy' 또는 'sell').
        Binance는 hedge 모드 off이므로 'buy'는 Long 진입, 'sell'은 Short 진입으로 동작.
        """
        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise ValueError("주문 side는 'buy' 또는 'sell' 이어야 합니다")
        return self.client.create_order(symbol=symbol, type="MARKET",
                                        side=side_upper, amount=qty)
    
    def create_exit_order(self, symbol: str, side: str, qty: float,
                          stop_price: float, tp: bool = True):
        """
        Binance: 포지션 청산용 TP/SL 주문 생성 (reduce-only).
        tp=True  ➜ TAKE_PROFIT_MARKET 주문, tp=False ➜ STOP_MARKET 주문.
        서버측 closePosition=True 설정으로 남은 포지션 전량 청산.
        """
        order_type = "TAKE_PROFIT_MARKET" if tp else "STOP_MARKET"
        side_upper = side.upper()
        return self.client.create_order(
            symbol=symbol,
            type=order_type,
            side=side_upper,
            amount=qty,
            params={
                "stopPrice": stop_price,
                "reduceOnly": True,
                "closePosition": True  # 포지션 전량정리 플래그
            }
        )
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        최근 8시간 Binance 펀딩 비율 조회 (양수: Long 이 받음, 음수: Short 이 받음).
        """
        try:
            res = self.client.fapiPublicGetPremiumIndex({"symbol": symbol.replace("/", "")})
            return float(res["lastFundingRate"])
        except Exception as e:
            logging.error(f"Binance 펀딩률 조회 실패: {e}")
            return 0.0

# ───────────────────────── 3-2) Bybit Futures 구현 ─────────────────────────
class BybitFutures(ExchangeClient):
    """Bybit USDT-M 선물 거래소 구현 클래스"""
    def __init__(self, key: str, secret: str):
        # ccxt Bybit 선물 클라이언트 초기화
        self.client = ccxt.bybit({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}   # Bybit USDT Perpetual Futures
        })
        try:
            self.client.fetch_time()  # API 연결 테스트
        except Exception as e:
            raise RuntimeError(f"Bybit API 연결 실패: {e}")
        logging.info("Bybit 선물 API 연결 성공")
    
    def set_leverage(self, symbol: str, leverage: int, isolated: bool):
        """
        Bybit: 격리 마진 및 레버리지 설정.
        (Bybit의 경우 교차/격리 구분 set_margin_mode 사용, 이미 격리면 오류 반환 가능)
        """
        try:
            if isolated:
                # Bybit도 ISOLATED로 설정 (Unified 계정의 경우 반영 안 될 수 있음)
                self.client.set_margin_mode("ISOLATED", symbol)
            self.client.set_leverage(leverage, symbol)
            logging.info(f"[Bybit] 레버리지 {leverage}x (isolated={isolated}) 설정 완료")
        except Exception as e:
            raise RuntimeError(f"Bybit 레버리지 설정 실패: {e}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, since=None, limit=500):
        # Binance와 동일한 CCXT 인터페이스 사용
        return self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    
    def create_market_order(self, symbol: str, side: str, qty: float):
        """
        시장가 주문 실행.
        side: 'buy' => Long 진입, 'sell' => Short 진입 (Bybit one-way 모드)
        """
        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise ValueError("주문 side는 'buy' 또는 'sell' 이어야 합니다")
        # Bybit 경우도 create_order 통일 인터페이스로 사용
        return self.client.create_order(symbol=symbol, type="MARKET",
                                        side=side_upper, amount=qty)
    
    def create_exit_order(self, symbol: str, side: str, qty: float,
                          stop_price: float, tp: bool = True):
        """
        Bybit: 포지션 청산용 주문 생성 (감시주문 – 조건부 시장가).
        - tp=True: 이익실현 주문, tp=False: 손절 주문.
        ※ Bybit는 'STOP_MARKET'/'TAKE_PROFIT_MARKET' 타입 지원 (ccxt에서 내부 처리).
          triggerPrice(stopPrice)와 triggerDirection 사용.
        """
        order_type = "TAKE_PROFIT_MARKET" if tp else "STOP_MARKET"
        side_upper = side.upper()
        # 트리거 방향 설정: 가격이 현재보다 높으면 1, 낮으면 2
        # (tp=True인 경우 Long은 가격상승 트리거(1), Short은 가격하락 트리거(2))
        trigger_dir = 1
        if tp:
            trigger_dir = 1 if side_upper == "SELL" else 2   # Long TP(매도) → 1, Short TP(매수) → 2
        else:
            trigger_dir = 2 if side_upper == "SELL" else 1   # Long SL(매도) → 2, Short SL(매수) → 1
        params = {
            "stopPrice": stop_price,
            "reduceOnly": True,
            "triggerDirection": trigger_dir
            # Bybit의 경우 closeOnTrigger 옵션 대신 reduceOnly로 포지션 청산 처리
        }
        return self.client.create_order(symbol=symbol, type=order_type,
                                        side=side_upper, amount=qty, params=params)
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Bybit: 펀딩 비율 조회 (현재 Bybit는 이력 API만 제공, 여기서는 0으로 처리).
        """
        try:
            # Bybit API v5에 펀딩 이력 조회 endpoint가 있음 (추후 구현 가능)
            # 일단 펀딩률 계산 생략 (dry-run 계산):contentReference[oaicite:18]{index=18} 
            return 0.0
        except Exception as e:
            logging.error(f"Bybit 펀딩률 조회 실패: {e}")
            return 0.0

# ───────────────────────── 4) 데이터 레이어 ──────────────────────────
class IndicatorRepository:
    """
    OHLCV 데이터 + 지표 병합 제공 클래스 (다중 타임프레임 지원).
    """
    def __init__(self, exchange: ExchangeClient, symbol: str):
        self.exchange = exchange
        self.symbol   = symbol
    
    def _fetch_cache(self, tf: str, limit: int = 500) -> pd.DataFrame:
        """
        개별 타임프레임 데이터 로드 (로컬 캐시 -> 신호 데이터 가져오기 -> 병합).
        캐시: 마지막 1개 진행중 캔들 제외하고 저장. 네트워크 오류 시 캐시 활용.
        """
        fp = CFG.DATA_DIR / f"{self.symbol.replace('/', '_')}_{tf}.parquet"
        cached = pd.read_parquet(fp) if fp.exists() else pd.DataFrame()
        # since: 캐시된 데이터 중 마지막에서 두번째 캔들 시각 (마지막 캔들은 진행중 가능성이 있으므로)
        since = int(cached.index[-2].value / 1e6) if len(cached) > 2 else None
        try:
            # CCXT로 신규 데이터 가져오기 (since 지정하면 그 시점부터 이후 데이터 2개 정도 요청)
            need = 2 if since else limit
            rows = self.exchange.fetch_ohlcv(self.symbol, tf, since=since, limit=need)
            df_new = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df_new["ts"] = pd.to_datetime(df_new["ts"], unit="ms")
            df_new.set_index("ts", inplace=True)
            # 만약 빈 데이터가 오면 예외 발생시켜 캐시 사용
            if df_new.empty:
                raise ValueError(f"No new data for {tf}")
            # 캐시 데이터와 신규 데이터를 병합 (중복시 최신값으로 덮어쓰기)
            full = pd.concat([cached, df_new]).drop_duplicates(keep="last")
            full.to_parquet(fp)  # 캐시 업데이트 저장
            return full.tail(limit)
        except ccxt.NetworkError as e:
            logging.warning(f"{tf} 데이터 네트워크 오류: {e} (캐시 데이터로 대체)")
            return cached.tail(limit)
        except Exception as e:
            logging.error(f"{tf} 데이터 fetch 실패: {e} (캐시 데이터 사용)")
            return cached.tail(limit)
    
    def get_merged(self) -> pd.DataFrame:
        """
        멀티 타임프레임 지표 병합:
        15m (기준) + 1h + 4h 지표를 모두 포함한 15m 그리드 DataFrame 리턴.
        """
        # 각 타임프레임 데이터 가져와 지표 계산
        df15 = add_indicators(self._fetch_cache("15m"))
        df1h = add_indicators(self._fetch_cache("1h")).resample("15T").ffill()   # 1시간봉 → 15분 간격 보간
        df4h = add_indicators(self._fetch_cache("4h")).resample("15T").ffill()   # 4시간봉 → 15분 보간
        
        if df15.empty or df1h.empty or df4h.empty:
            raise ValueError("타임프레임 데이터 부족 (하나 이상의 데이터프레임이 비어있음)")
        
        base = df15.copy()
        # 상위 TF 지표 값을 15m 기준 데이터프레임에 추가
        base["rsi_1h"]      = df1h["rsi"]
        base["ema_fast_4h"] = df4h["ema_fast"]
        base["ema_slow_4h"] = df4h["ema_slow"]
        # 예측 타깃 (다음 캔들의 종가 상승 여부 0/1)
        base["target"]      = (base["close"].shift(-1) > base["close"]).astype(int)
        return base.dropna()

# ─────────────────────────── 5) Model Layer ───────────────────────────
class ModelService:
    """
    XGBoost 모델 서비스 (학습/예측/저장 관리).
    """
    # 사용 피처 목록 (feature columns)
    FEATURES = ["close", "ema_fast", "ema_slow", "rsi",
                "rsi_1h", "ema_fast_4h", "ema_slow_4h",
                "atr", "macd", "macd_sig"]
    
    def __init__(self, path: Path):
        self.path = path
        # 기존 모델 파일이 있으면 로드
        self.model = joblib.load(path) if path.exists() else None
        self.t_last_train = datetime.min   # 마지막 학습 시각
        self.t_last_grid  = datetime.min   # 마지막 그리드서치 시각
    
    def train(self, df: pd.DataFrame) -> None:
        """
        주어진 DataFrame으로 XGBoost 모델 재학습 (80%:20% train:valid split).
        불균형 타깃은 SMOTE로 보정. 주 1회 GridSearch 수행, 그 외에는 warm-start 추가학습.
        """
        X, y = df[self.FEATURES], df["target"]
        split = int(len(df) * 0.8)
        # SMOTE 적용하여 train 셋의 0/1 분포 균형 조정
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X[:split], y[:split])
        
        # ➊ 주기적(Grid_DAYS마다)으로는 GridSearchCV로 최적 하이퍼파라미터 탐색
        if self.model is None or (datetime.utcnow() - self.t_last_grid).days >= CFG.GRID_DAYS:
            base = XGBClassifier(subsample=0.8, colsample_bytree=0.8,
                                 use_label_encoder=False, eval_metric="logloss")
            grid = GridSearchCV(
                base,
                param_grid={"n_estimators": [120, 160],
                            "max_depth": [3, 4],
                            "learning_rate": [0.05, 0.1]},
                cv=3, n_jobs=-1
            )
            grid.fit(X_tr, y_tr)
            self.model = grid.best_estimator_
            self.t_last_grid = datetime.utcnow()
            logging.info(f"GridSearch 최적 파라미터: {grid.best_params_}")
        else:
            # ➋ 그 외에는 기존 모델 기반으로 추가 학습(warm-start): 트리 40개씩 추가
            n_old = self.model.get_params().get("n_estimators", 0)
            self.model.set_params(n_estimators=n_old + 40)
            # 기존 booster 가져와 이어붙여 학습
            self.model.fit(X_tr, y_tr, xgb_model=self.model.get_booster())
        
        # 모델 저장 및 메타데이터 갱신
        joblib.dump(self.model, self.path)
        self.t_last_train = datetime.utcnow()
        tg("📈 모델 재학습 완료")  # 텔레그램으로 알림
    
    def add_prob(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame에 상승 확률(predicted probability of class=1)을 계산하여 'prob_up' 컬럼으로 추가.
        모델이 없는 경우 0.5를 기본값으로 사용.
        """
        df = df.copy()
        if self.model:
            prob_arr = self.model.predict_proba(df[self.FEATURES])[:, 1]
            df["prob_up"] = prob_arr
        else:
            df["prob_up"] = 0.5
        return df

# ─────────────────────────── 6) 전략 (Strategy) ───────────────────────────
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

# ──────────────────────── 7) 주문 서비스 (OrderService) ───────────────────────
class OrderService:
    """
    포지션 상태 및 주문 실행 관리.
    - Paper 모드: 가상 체결 처리 및 PnL 계산.
    - Live 모드: 실시간 거래소 주문 (TP/SL는 거래소 서버 처리).
    """
    def __init__(self, ex: ExchangeClient, paper: bool = True, init_balance: float = 0.0):
        self.ex      = ex
        self.paper   = paper
        self.balance = init_balance  # paper 모드 전용 가상 잔고
        self.pos     = None          # 현재 포지션 (dict: entry, qty, side)
        self.trades  = []           # 거래 내역 리스트 (딕셔너리 원소)
        self.loss_cnt = 0          # 연속 손실 카운트
        self.pause_until = None    # 일시정지 해제 시각 (UTC datetime)
        self.lock    = threading.Lock()  # thread-safe 처리용 락
        
        # Live 모드: 레버리지 및 마진모드 설정
        if not paper:
            try:
                self.ex.set_leverage(CFG.SYMBOL, CFG.LEVERAGE, CFG.ISOLATED)
            except Exception as e:
                logging.error(f"초기 레버리지 설정 오류: {e}")
                tg(f"⚠️ 레버리지 설정 오류: {e}")
    
    # ---------- 내부 유틸 함수 ----------
    def _pnl(self, exit_px: float) -> float:
        """
        현재 포지션을 exit_px 가격에 청산했을 때의 손익(PnL)을 계산.
        계산식: (진입 대비 가격차 * 수량 * 레버리지) – (왕복 수수료) – (펀딩비)
        """
        if not self.pos:
            return 0.0
        entry = self.pos["entry"]
        qty   = self.pos["qty"]
        side  = self.pos["side"]
        # Long: (종료가 - 진입가), Short: (진입가 - 종료가)
        delta = (exit_px - entry) if side == "long" else (entry - exit_px)
        # 추정 수수료: 포지션 명목 가치 * 0.06% * 2 (진입+청산) 
        fee = abs(exit_px * qty) * 0.0006
        # 펀딩비 (8시간 간격) – 단순 현재 펀딩률로 8시간치 예측
        frate = self.ex.fetch_funding_rate(CFG.SYMBOL)
        funding = abs(entry * qty) * frate
        return delta * qty * CFG.LEVERAGE - fee - funding
    
    # ---------- 진입 후 TP/SL 예약주문 부착 ----------
    def _attach_tp_sl(self, side: str, entry_px: float, qty: float):
        """
        이익실현(TP)와 손절(SL) 주문을 거래소 서버에 등록 (reduce-only).
        Paper 모드에서는 로그만 남기고, Live 모드에서는 실제 주문 요청.
        """
        # 목표가 / 손절가 계산
        if side == "long":
            tp_px = entry_px * (1 + CFG.TP_PCT)
            sl_px = entry_px * (1 - CFG.SL_PCT)
            exit_side = "SELL"  # 롱 포지션 청산은 매도
        else:  # side == "short"
            tp_px = entry_px * (1 - CFG.TP_PCT)
            sl_px = entry_px * (1 + CFG.SL_PCT)
            exit_side = "BUY"   # 숏 포지션 청산은 매수
        if self.paper:
            # 모의모드: 주문 체결은 하지 않고 로그로 남김
            logging.info(f"[PAPER] TP/SL 가상주문 등록 → TP: {tp_px:.2f}, SL: {sl_px:.2f}")
        else:
            # 실거래: 거래소에 주문 제출
            try:
                # 이익실현 주문
                self.ex.create_exit_order(CFG.SYMBOL, side=exit_side, qty=qty,
                                           stop_price=round(tp_px, 2), tp=True)
                # 손절 주문
                self.ex.create_exit_order(CFG.SYMBOL, side=exit_side, qty=qty,
                                           stop_price=round(sl_px, 2), tp=False)
                logging.info(f"TP/SL 주문 제출 완료 ▶ TP:{tp_px:.2f} / SL:{sl_px:.2f}")
            except Exception as e:
                logging.error(f"TP/SL 주문 제출 실패: {e}")
                tg(f"⚠️ TP/SL 주문 실패: {e}")
    
    # ---------- 포지션 오픈 (진입) ----------
    def open_position(self, px: float, qty: float, side: str):
        """
        신규 포지션 진입 처리:
        1) 지정된 side 방향으로 시장가 주문 실행
        2) 즉시 대응되는 TP/SL 청산 주문 등록 (_attach_tp_sl)
        """
        with self.lock:
            if self.pos:
                return  # 이미 포지션 존재 시 신규 진입하지 않음
            # 슬리피지 고려한 체결 가격 추정 (paper모드) - Long이면 조금 높게, Short이면 낮게
            entry_px = px * (1 + CFG.SLIP_PCT) if side == "long" else px * (1 - CFG.SLIP_PCT)
            if not self.paper:
                try:
                    order = self.ex.create_market_order(CFG.SYMBOL,
                                                        "buy" if side == "long" else "sell",
                                                        qty)
                    # 실제 체결 가격 얻기 (일부 거래소는 'price' 필드 없을 수 있음)
                    entry_px = float(order.get("price", order.get("avgPrice", entry_px)))
                except Exception as e:
                    logging.error(f"{side.upper()} 주문 실패: {e}")
                    tg(f"⚠️ {side.upper()} 진입 주문 실패: {e}")
                    return
            # 포지션 상태 기록
            self.pos = {"entry": entry_px, "qty": qty, "side": side}
            # 거래 내역 저장 (진입)
            self.trades.append({
                "time": datetime.utcnow(), 
                "side": side.upper(), 
                "price": entry_px, 
                "bal": self.balance
            })
            tg(f"🚀 {'[PAPER]' if self.paper else '[LIVE]'} {side.upper()} 진입 @ {entry_px:.2f}")
            # 진입 후 바로 TP/SL 예약 주문 설정
            self._attach_tp_sl(side, entry_px, qty)
    
    # ---------- 포지션 종료 체크 (paper 모드 전용) ----------
    def poll_position_closed(self, px_now: float):
        """
        paper 모드 전용: 현재 가격(px_now)이 포지션의 TP 또는 SL에 도달했는지 확인.
        도달 시 포지션을 청산하고 PnL 계산하여 가상 잔고 업데이트.
        (live 모드: TP/SL는 거래소가 자동 처리하므로 봇에서는 확인하지 않음)
        """
        if not self.pos:
            return
        if not self.paper:
            # 실거래 모드: 거래소가 이미 TP/SL로 포지션 정리함. (여기서는 처리 없음)
            return
        entry = self.pos["entry"]; qty = self.pos["qty"]; side = self.pos["side"]
        # 롱 포지션의 TP/SL 조건
        hit_tp = (px_now >= entry * (1 + CFG.TP_PCT)) if side == "long" else (px_now <= entry * (1 - CFG.TP_PCT))
        hit_sl = (px_now <= entry * (1 - CFG.SL_PCT)) if side == "long" else (px_now >= entry * (1 + CFG.SL_PCT))
        if not (hit_tp or hit_sl):
            return  # 아직 청산가격 도달 안함
        
        # 포지션 청산 처리
        pnl = self._pnl(px_now)  # 손익 계산
        self.balance += pnl      # 가상 잔고 갱신
        # 거래 내역 저장 (청산)
        self.trades.append({
            "time": datetime.utcnow(),
            "side": f"CLOSE_{side.upper()}",
            "price": px_now,
            "bal": self.balance,
            "pnl": pnl
        })
        tg(f"✅ [PAPER] {side.upper()} 청산 @ {px_now:.2f}  PnL={pnl:.2f}")
        # 연속 손실 체크
        if pnl < 0:
            self.loss_cnt += 1
            if self.loss_cnt >= CFG.MAX_LOSS:
                self.pause_until = datetime.utcnow() + timedelta(hours=CFG.PAUSE_HR)
                tg(f"⛔ 연속 손실 {self.loss_cnt}회 발생 – {CFG.PAUSE_HR}시간 휴식 모드")
        else:
            self.loss_cnt = 0
        self.pos = None  # 포지션 없음 상태로 리셋
    
    # ---------- 트레이딩 일시정지 여부 확인 ----------
    def is_paused(self) -> bool:
        """
        연속 손실로 인한 일시정지(pause) 상태인지 확인.
        pause 기간이 지났으면 자동 해제.
        """
        if self.pause_until and datetime.utcnow() < self.pause_until:
            return True
        if self.pause_until and datetime.utcnow() >= self.pause_until:
            # 휴식 기간 종료 -> 재개
            self.pause_until = None
            self.loss_cnt = 0
            tg("▶️ 트레이딩 재개")
        return False

# ────────────────────────── 8) Trading Bot 오케스트레이터 ──────────────────────────
class TradingBot:
    """
    주요 클래스 (Repository, ModelService, OrderService)를 묶어 전체 트레이딩 로직을 관리.
    """
    def __init__(self, repo: IndicatorRepository,
                 model_svc: ModelService,
                 order_svc: OrderService):
        self.repo  = repo
        self.model = model_svc
        self.order = order_svc
        self.df_latest = pd.DataFrame()  # 최근 지표 데이터 (UI 표시용)
        self.lock = threading.Lock()
    
    def loop(self) -> None:
        """
        메인 트레이딩 루프: 무한 반복하면서 데이터 갱신, 모델 예측, 신호 판단, 주문 실행.
        """
        while True:
            try:
                # (1) 일시정지 상태 확인
                if self.order.is_paused():
                    time.sleep(CFG.SLEEP_SEC)
                    continue
                
                # (2) 데이터 수집 및 통합 지표 생성
                df = self.repo.get_merged()
                
                # (3) 모델 재학습 조건 확인 (지정한 주기 경과 또는 최초 실행시)
                need_train = (self.model.model is None or 
                              (datetime.utcnow() - self.model.t_last_train).seconds > CFG.RETRAIN_HR * 3600)
                if need_train:
                    self.model.train(df)
                
                # (4) 최신 데이터에 대해 예측확률 추가 및 전략 신호 계산
                df = self.model.add_prob(df)
                df = Strategy.enrich(df)
                # 최근 500개 데이터프레임을 저장 (UI에서 참조)
                with self.lock:
                    self.df_latest = df.tail(500).copy()
                
                # (5) 마지막 캔들의 신호 확인하여 포지션 진입/청산 처리
                last = df.iloc[-1]
                if self.order.pos is None:
                    # --- 신규 포지션 진입 로직 ---
                    if last["long"]:
                        # 증거금 모드 vs 수량 모드에 따라 포지션 수량 결정
                        if CFG.MARGIN_PER_TRADE > 0:
                            # 사용자가 지정한 증거금 기반 수량 계산
                            qty = (CFG.MARGIN_PER_TRADE * CFG.LEVERAGE) / max(last["close"], 1e-6)
                        else:
                            # 기존 방식: 고정 코인수 * (1/ATR) (변동성 클수록 수량 축소)
                            qty = CFG.POS_SIZE / max(last["atr"], 1e-6)
                        qty = min(qty, CFG.MAX_QTY)  # 최대 수량 제한 적용
                        self.order.open_position(last["close"], qty, "long")
                    elif last["short"]:
                        if CFG.MARGIN_PER_TRADE > 0:
                            qty = (CFG.MARGIN_PER_TRADE * CFG.LEVERAGE) / max(last["close"], 1e-6)
                        else:
                            qty = CFG.POS_SIZE / max(last["atr"], 1e-6)
                        qty = min(qty, CFG.MAX_QTY)
                        self.order.open_position(last["close"], qty, "short")
                else:
                    # --- 포지션 보유 중이면 청산 조건 확인 (paper 모드) ---
                    # Live 모드에서는 거래소의 주문으로 청산되므로 따로 처리하지 않음
                    self.order.poll_position_closed(last["close"])
                
                time.sleep(CFG.SLEEP_SEC)
            except Exception as e:
                logging.error(f"메인 루프 오류: {e}")
                tg(f"⚠️ 루프 오류: {e}")
                time.sleep(30)
    
    def get_df(self) -> pd.DataFrame:
        """UI 대시보드용 최근 데이터 DataFrame 안전 반환 (복사본 제공)."""
        with self.lock:
            return self.df_latest.copy()

# ────────────────────────── 9) Streamlit 대시보드 UI ──────────────────────────
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

# ─────────────────────────── 10) Main 실행부 ───────────────────────────
def main():
    # ➊ 선택된 거래소에 따라 ExchangeClient 구현체 인스턴스 생성
    exchange_client: ExchangeClient
    try:
        if CFG.EXCHANGE_NAME == "BYBIT":
            exchange_client = BybitFutures(CFG.API_KEY, CFG.API_SECRET)
        else:
            exchange_client = BinanceFutures(CFG.API_KEY, CFG.API_SECRET)
    except Exception as e:
        sys.exit(f"[FATAL] {e}")
    
    # ➋ 주요 서비스 클래스 초기화
    repo  = IndicatorRepository(exchange_client, CFG.SYMBOL)
    model = ModelService(CFG.MODEL_FP)
    order = OrderService(exchange_client, paper=CFG.TEST_MODE, init_balance=CFG.INIT_BAL)
    bot   = TradingBot(repo, model, order)
    
    # ➌ 백그라운드 트레이딩 루프 시작 (데몬 쓰레드)
    threading.Thread(target=bot.loop, daemon=True).start()
    # ➍ Streamlit 대시보드 실행
    ui_dashboard(bot)

if __name__ == "__main__":
    main()
