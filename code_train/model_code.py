import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA = r'C:\Users\sripa\OneDrive\Desktop\slm\data\signal_metrics.csv'
OUT = Path("distill"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = "Latency (ms)"

df = pd.read_csv(DATA)

# keep your hour feature
if "Timestamp" in df.columns and "Hour" not in df.columns:
    df["Hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour

y = df[TARGET].astype(float)
X = df.drop(columns=[TARGET])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", RobustScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

pipe = Pipeline([
    ("pre", pre),
    ("fs", SelectKBest(score_func=f_regression, k="all")),
    ("model", RandomForestRegressor(n_estimators=190, max_depth=5, n_jobs=-1, random_state=42))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)

metrics = {
    "R2": float(r2_score(yte, pred)),
    "MAE": float(mean_absolute_error(yte, pred))
}
print(metrics)

# persist artifacts
joblib.dump(pipe.named_steps["pre"], OUT / "pre.pkl")
joblib.dump(pipe.named_steps["fs"],  OUT / "fs.pkl")
joblib.dump(pipe.named_steps["model"], OUT / "rf.pkl")

# feature names after preprocessing+OHE (before SelectKBest has no effect since k='all')
try:
    feat_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
except:
    feat_names = [f"f{i}" for i in range(pipe.named_steps["pre"].transform(Xtr).shape[1])]

(Path(OUT / "schema.json")).write_text(json.dumps({
    "target": TARGET,
    "numeric": num_cols,
    "categorical": cat_cols,
    "feat_names_after_pre": feat_names,
    "metrics": metrics
}, indent=2))
