import logging
import pandas as pd
import ccxt
from config.config import CFG
from src.exchange.exchange_client import ExchangeClient
from src.utils.helpers import add_indicators

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
