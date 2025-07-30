import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ➊ .env 파일에서 환경변수 불러오기
load_dotenv()

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
