import logging
import threading
from datetime import datetime, timedelta
from src.exchange.exchange_client import ExchangeClient
from config.config import CFG
from src.utils.helpers import tg

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
        계산식: (진입 대비 가격차 * 수량) – (왕복 수수료) – (펀딩비)
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
        # PnL = (가격 변화 * 수량) - 수수료 - 펀딩비. 레버리지는 PnL 자체에 곱해지지 않음.
        return delta * qty - fee - funding

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

    # ---------- 포지션 강제 종료 (전략/수동) ----------
    def close_position(self, px: float, reason: str = "STRATEGY"):
        """
        현재 포지션을 시장가로 즉시 종료.
        - Live 모드: 기존 TP/SL 주문 취소 -> 시장가 청산 주문.
        - Paper 모드: 가상으로 청산 처리 및 PnL 계산.
        """
        with self.lock:
            if not self.pos:
                return

            side = self.pos["side"]
            qty = self.pos["qty"]
            exit_px = px
            exit_side = "sell" if side == "long" else "buy"

            if not self.paper:
                try:
                    # 1) Live 모드: 기존 TP/SL 주문 모두 취소
                    self.ex.cancel_all_orders(CFG.SYMBOL)
                    # 2) 시장가로 포지션 종료 주문
                    order = self.ex.create_market_order(CFG.SYMBOL, exit_side, qty)
                    exit_px = float(order.get("price", order.get("avgPrice", px)))
                except Exception as e:
                    logging.error(f"강제청산 주문 실패: {e}")
                    tg(f"⚠️ {side.upper()} 강제청산 실패: {e}")
                    return  # 청산 실패 시 포지션 유지
            else:
                # Paper 모드: 슬리피지 적용
                exit_px = px * (1 - CFG.SLIP_PCT) if side == "long" else px * (1 + CFG.SLIP_PCT)

            # --- 공통 청산 후 처리 ---
            pnl = self._pnl(exit_px)
            if self.paper:
                self.balance += pnl

            self.trades.append({
                "time": datetime.utcnow(),
                "side": f"CLOSE_{side.upper()}",
                "price": exit_px,
                "bal": self.balance,
                "pnl": pnl,
                "reason": reason
            })
            tg(f"✅ {'[PAPER]' if self.paper else '[LIVE]'} {side.upper()} 청산({reason}) @ {exit_px:.2f}, PnL={pnl:.2f}")

            if pnl < 0:
                self.loss_cnt += 1
                if self.loss_cnt >= CFG.MAX_LOSS:
                    self.pause_until = datetime.utcnow() + timedelta(hours=CFG.PAUSE_HR)
                    tg(f"⛔ 연속 손실 {self.loss_cnt}회 발생 – {CFG.PAUSE_HR}시간 휴식 모드")
            else:
                self.loss_cnt = 0

            self.pos = None  # 포지션 리셋

    # ---------- 포지션 종료 체크 (paper 모드 전용) ----------
    def poll_position_closed(self, px_now: float):
        """
        paper 모드 전용: 현재 가격(px_now)이 포지션의 TP 또는 SL에 도달했는지 확인.
        도달 시 포지션을 청산하고 PnL 계산하여 가상 잔고 업데이트.
        (live 모드: TP/SL는 거래소가 자동 처리하므로 봇에서는 확인하지 않음)
        """
        if not self.pos or not self.paper:
            return

        entry = self.pos["entry"]; side = self.pos["side"]
        # 롱 포지션의 TP/SL 조건
        hit_tp = (px_now >= entry * (1 + CFG.TP_PCT)) if side == "long" else (px_now <= entry * (1 - CFG.TP_PCT))
        hit_sl = (px_now <= entry * (1 - CFG.SL_PCT)) if side == "long" else (px_now >= entry * (1 + CFG.SL_PCT))

        if hit_tp:
            self.close_position(px_now, reason="TP")
        elif hit_sl:
            self.close_position(px_now, reason="SL")

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
