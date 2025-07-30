import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib
from config.config import CFG
from src.utils.helpers import tg

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
