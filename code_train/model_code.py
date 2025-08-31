from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

@dataclass(frozen=True)
class TrainConfig:
    csv_path: Path
    target: str = "Latency (ms)"
    out_dir: Path = Path("distill")

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    num_cols: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return pre
def train_and_save(cfg: TrainConfig) -> Dict[str, float]:
    df = pd.read_csv(cfg.csv_path)
    if "Timestamp" in df.columns and "Hour" not in df.columns:
        df["Hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour
    if cfg.target not in df.columns:
        raise ValueError(f"Target column '{cfg.target}' not found.")
    y = df[cfg.target].astype(float)
    X = df.drop(columns=[cfg.target])
    pre = build_preprocessor(X)
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("fs", SelectKBest(score_func=f_regression, k="all")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=190, max_depth=5, n_jobs=-1, random_state=42
                ),
            ),
        ]
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    metrics = {
        "R2": float(r2_score(yte, pred)),
        "MAE": float(mean_absolute_error(yte, pred)),
    }
    print(metrics)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe.named_steps["pre"], cfg.out_dir / "pre.pkl")
    joblib.dump(pipe.named_steps["fs"], cfg.out_dir / "fs.pkl")
    joblib.dump(pipe.named_steps["model"], cfg.out_dir / "rf.pkl")
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        n_feats = pre.transform(Xtr.iloc[:1]).shape[1]
        feat_names = [f"f{i}" for i in range(n_feats)]
    (cfg.out_dir / "schema.json").write_text(
        json.dumps(
            {
                "target": cfg.target,
                "numeric": X.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": X.select_dtypes(exclude=[np.number]).columns.tolist(),
                "feat_names_after_pre": feat_names,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics

if __name__ == "__main__":
    cfg = TrainConfig(csv_path=Path("data/signal_metrics.csv"))
    train_and_save(cfg)