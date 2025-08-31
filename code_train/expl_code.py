# expl_code.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse
import shap

# -----------------------------
# CONFIG — change if needed
# -----------------------------
DATA = "CSV file"  # your CSV
DISTILL_DIR = Path("distill")
PRE_PATH = DISTILL_DIR / "pre.pkl"
FS_PATH = DISTILL_DIR / "fs.pkl"
RF_PATH = DISTILL_DIR / "rf.pkl"
SCHEMA_PATH = DISTILL_DIR / "schema.json"
OUT_JSONL = DISTILL_DIR / "corpus.jsonl"

assert PRE_PATH.exists(), f"Missing {PRE_PATH}"
assert FS_PATH.exists(), f"Missing {FS_PATH}"
assert RF_PATH.exists(), f"Missing {RF_PATH}"
assert SCHEMA_PATH.exists(), f"Missing {SCHEMA_PATH}"

pre = joblib.load(PRE_PATH)
fs = joblib.load(FS_PATH)
rf = joblib.load(RF_PATH)
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
TARGET = schema["target"]
df = pd.read_csv(DATA)
if "Timestamp" in df.columns and "Hour" not in df.columns:
    df["Hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in CSV.")

y = df[TARGET].astype(float)
X = df.drop(columns=[TARGET])

X_pre = pre.transform(X)
X_pre = fs.transform(X_pre)  # no-op if k='all', but keeps API consistent
if sparse.issparse(X_pre):
    X_pre = X_pre.toarray()

X_pre = X_pre.astype(np.float64, copy=False)
feat_names = schema.get("feat_names_after_pre", None)
if not feat_names:
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_pre.shape[1])]
if len(feat_names) != X_pre.shape[1]:
    feat_names = [f"f{i}" for i in range(X_pre.shape[1])]
expl = shap.TreeExplainer(rf)
sh = expl.shap_values(X_pre)  # shape: (n, d)
preds = rf.predict(X_pre)
def topk(idx: int, k: int = 6):
    sv = sh[idx]
    pairs = sorted(zip(feat_names, sv), key=lambda p: abs(p[1]), reverse=True)[:k]
    return [{"feature": f, "impact": float(v)} for f, v in pairs]

def row_to_dict(ix: int) -> dict:
    row = {}
    for c in X.columns:
        val = X.iloc[ix][c]
        if pd.isna(val):
            row[c] = None
        elif isinstance(val, (np.integer, np.floating)):
            row[c] = float(val)
        else:
            row[c] = val
    return row

DISTILL_DIR.mkdir(parents=True, exist_ok=True)
count = 0
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    n = len(X)
    for i in range(n):
        row = row_to_dict(i)
        analysis = {
            "missing_fields": [k for k, v in row.items() if v is None],
            "note": "RobustScaler on numeric; unseen categories handled by OHE(ignore)."
        }
        output = {
            "prediction": float(preds[i]),
            "analysis": analysis,
            "explanation": topk(i, 6)
        }
        prompt = {
            "instruction": "Given a network signal row, predict latency (ms), analyze data quality, and explain key drivers.",
            "input": {"row": row}
        }
        f.write(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False) + "\n")
        count += 1

print(f"Wrote: {OUT_JSONL} ({count} examples)")
