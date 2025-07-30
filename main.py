import sys
import logging
import threading
from logging.handlers import RotatingFileHandler, StreamHandler

from config.config import CFG
from src.exchange.exchange_client import ExchangeClient
from src.exchange.binance_futures import BinanceFutures
from src.exchange.bybit_futures import BybitFutures
from src.data.indicator_repository import IndicatorRepository
from src.model.model_service import ModelService
from src.order.order_service import OrderService
from src.bot.trading_bot import TradingBot
from src.ui.dashboard import ui_dashboard

def main():
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
