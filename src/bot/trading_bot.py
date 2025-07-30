import logging
import threading
import time
from datetime import datetime
import pandas as pd
from src.data.indicator_repository import IndicatorRepository
from src.model.model_service import ModelService
from src.order.order_service import OrderService
from src.strategy.strategy import Strategy
from config.config import CFG
from src.utils.helpers import tg

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
                    # --- 포지션 보유 중: 청산 조건 확인 ---
                    pos_side = self.order.pos["side"]
                    should_exit = (pos_side == "long" and last["exit_l"]) or \
                                  (pos_side == "short" and last["exit_s"])

                    if should_exit:
                        # 1) 전략에 따른 청산 신호 발생
                        self.order.close_position(last["close"], reason="STRATEGY")
                    else:
                        # 2) Paper 모드에서 TP/SL 도달 여부 확인
                        # (Live 모드에서는 서버에서 처리되므로 이 로직은 무시됨)
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
