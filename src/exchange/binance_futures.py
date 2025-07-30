import logging
import ccxt
from src.exchange.exchange_client import ExchangeClient

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

    def cancel_all_orders(self, symbol: str):
        """지정된 심볼의 모든 대기 주문 취소"""
        try:
            self.client.cancel_all_orders(symbol)
            logging.info(f"[Binance] {symbol}의 모든 대기 주문 취소 완료")
        except Exception as e:
            logging.error(f"[Binance] 주문 취소 실패: {e}")
            # 주문 취소 실패는 치명적일 수 있으므로 예외를 다시 발생시킬 수 있음
            # 여기서는 로깅만 하고 계속 진행
            pass
