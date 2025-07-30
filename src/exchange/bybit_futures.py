import logging
import ccxt
from src.exchange.exchange_client import ExchangeClient

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
            "triggerDirection": trigger_dir,
            "closeOnTrigger": True  # Bybit: 트리거 시 포지션 전체 종료
        }
        return self.client.create_order(symbol=symbol, type=order_type,
                                        side=side_upper, amount=qty, params=params)

    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Bybit: 펀딩 비율 조회 (현재 Bybit는 이력 API만 제공, 여기서는 0으로 처리).
        """
        try:
            # Bybit API v5에 펀딩 이력 조회 endpoint가 있음 (추후 구현 가능)
            return 0.0
        except Exception as e:
            logging.error(f"Bybit 펀딩률 조회 실패: {e}")
            return 0.0

    def cancel_all_orders(self, symbol: str):
        """지정된 심볼의 모든 대기 주문 취소"""
        try:
            self.client.cancel_all_orders(symbol)
            logging.info(f"[Bybit] {symbol}의 모든 대기 주문 취소 완료")
        except Exception as e:
            logging.error(f"[Bybit] 주문 취소 실패: {e}")
            pass
