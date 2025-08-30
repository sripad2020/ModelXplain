import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("data/signal_metrics.csv")
if "Timestamp" in df.columns:
    df["Hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour
y = df["Latency (ms)"]
X = df.drop(columns=["Latency (ms)"])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  RobustScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
rf_pipe = Pipeline([
    ("pre", pre),
    ("fs", SelectKBest(score_func=f_regression, k=5)),
    ("model", RandomForestRegressor(
        n_estimators=190, max_depth=5
    ))
])
et_pipe = Pipeline([
    ("pre", pre),
    ("fs", SelectKBest(score_func=f_regression, k=5)),
    ("model", ExtraTreesRegressor(
        n_estimators=210, max_depth=2
    ))
])
for name, pipe in [("RandomForest", rf_pipe), ("ExtraTrees", et_pipe)]:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    mask = pipe.named_steps["fs"].get_support()
    selected_features = feature_names[mask]

    print(f"\n{name}")
    print("Selected Features:", selected_features)
    print("R2  :", round(r2_score(y_test, pred), 4))
    print("MAE :", round(mean_absolute_error(y_test, pred), 3))