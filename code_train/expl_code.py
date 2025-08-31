from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypedDict, Union
import joblib
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
from scipy import sparse

RowValue = Union[str, float, int, None]
RowDict = Dict[str, RowValue]

class ExplanationItem(TypedDict):
    feature: str
    impact: float
class AnalysisDict(TypedDict):
    missing_fields: List[str]
    note: str
class OutputDict(TypedDict):
    prediction: float
    analysis: AnalysisDict
    explanation: List[ExplanationItem]
class PromptInput(TypedDict):
    row: RowDict
class PromptDict(TypedDict):
    instruction: str
    input: PromptInput
class CorpusEntry(TypedDict):
    prompt: PromptDict
    output: OutputDict

@dataclass(frozen=True)
class DistillPaths:
    data_csv: Path
    distill_dir: Path = Path("distill")

    @property
    def pre_path(self) -> Path: return self.distill_dir / "pre.pkl"
    @property
    def fs_path(self) -> Path: return self.distill_dir / "fs.pkl"
    @property
    def rf_path(self) -> Path: return self.distill_dir / "rf.pkl"
    @property
    def schema_path(self) -> Path: return self.distill_dir / "schema.json"
    @property
    def out_jsonl(self) -> Path: return self.distill_dir / "corpus.jsonl"


def ensure_dense_float(X: Union[NDArray[Any], "sparse.spmatrix"]) -> NDArray[np.float64]:
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    return X


def _row_to_dict(df: pd.DataFrame, ix: int) -> RowDict:
    row: RowDict = {}
    for c, val in df.iloc[ix].items():
        if pd.isna(val):
            row[c] = None
        elif isinstance(val, (np.integer, int)):
            row[c] = int(val)  # keep ints as ints
        elif isinstance(val, (np.floating, float)):
            row[c] = float(val)
        else:
            row[c] = str(val)
    return row


def build_corpus(paths: DistillPaths, top_k: int = 6) -> int:
    # Load artifacts
    assert paths.pre_path.exists(), f"Missing {paths.pre_path}"
    assert paths.fs_path.exists(), f"Missing {paths.fs_path}"
    assert paths.rf_path.exists(), f"Missing {paths.rf_path}"
    assert paths.schema_path.exists(), f"Missing {paths.schema_path}"

    pre = joblib.load(paths.pre_path)          # type: ignore[no-any-unimported]
    fs = joblib.load(paths.fs_path)            # type: ignore[no-any-unimported]
    rf = joblib.load(paths.rf_path)            # type: ignore[no-any-unimported]
    schema: Dict[str, Any] = json.loads(paths.schema_path.read_text(encoding="utf-8"))
    target: str = str(schema["target"])

    # Load data
    df = pd.read_csv(paths.data_csv)
    if "Timestamp" in df.columns and "Hour" not in df.columns:
        df["Hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")

    y = df[target].astype(float)
    X_df = df.drop(columns=[target])

    # Transform X -> dense float
    X_pre = ensure_dense_float(fs.transform(pre.transform(X_df)))  # type: ignore[no-any-unimported]

    # Feature names after preprocessing
    feat_names: List[str]
    if "feat_names_after_pre" in schema and isinstance(schema["feat_names_after_pre"], list):
        feat_names = [str(x) for x in schema["feat_names_after_pre"]]
    else:
        try:
            feat_names = list(pre.get_feature_names_out())  # type: ignore[no-any-unimported]
        except Exception:
            feat_names = [f"f{i}" for i in range(X_pre.shape[1])]
    if len(feat_names) != X_pre.shape[1]:
        feat_names = [f"f{i}" for i in range(X_pre.shape[1])]
    expl = shap.TreeExplainer(rf)  # type: ignore[no-any-unimported]
    sh_vals: NDArray[np.float64] = np.asarray(expl.shap_values(X_pre))  # type: ignore[no-any-unimported]
    preds: NDArray[np.float64] = np.asarray(rf.predict(X_pre), dtype=np.float64)  # type: ignore[no-any-unimported]

    def topk(idx: int, k: int) -> List[ExplanationItem]:
        sv = sh_vals[idx]
        pairs = sorted(zip(feat_names, sv), key=lambda p: abs(float(p[1])), reverse=True)[:k]
        return [{"feature": f, "impact": float(v)} for f, v in pairs]
    paths.distill_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with paths.out_jsonl.open("w", encoding="utf-8") as f:
        for i in range(X_df.shape[0]):
            row = _row_to_dict(X_df, i)
            analysis: AnalysisDict = {
                "missing_fields": [k for k, v in row.items() if v is None],
                "note": "RobustScaler on numeric; unseen categories handled by OHE(ignore).",
            }
            output: OutputDict = {
                "prediction": float(preds[i]),
                "analysis": analysis,
                "explanation": topk(i, top_k),
            }
            prompt: PromptDict = {
                "instruction": "Given a network signal row, predict latency (ms), analyze data quality, and explain key drivers.",
                "input": {"row": row},
            }
            entry: CorpusEntry = {"prompt": prompt, "output": output}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count
if __name__ == "__main__":
    paths = DistillPaths(data_csv=Path("data/signal_metrics.csv"))
    n = build_corpus(paths, top_k=6)
    print(f"Wrote: {paths.out_jsonl} ({n} examples)")
