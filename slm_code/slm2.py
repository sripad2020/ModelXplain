import json, torch, joblib, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

DISTILL_DIR = Path("distill")
pre = joblib.load(DISTILL_DIR / "pre.pkl")
fs = joblib.load(DISTILL_DIR / "fs.pkl")
rf = joblib.load(DISTILL_DIR / "rf.pkl")
schema = json.loads((DISTILL_DIR / "schema.json").read_text(encoding="utf-8"))

BASE = "distilgpt2"
ADAPTER = "slm/out-lora-no-trl"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER).to("cpu")

def predict_with_rf(row_dict):
    X = pd.DataFrame([row_dict])
    X_pre = pre.transform(X)
    X_pre = fs.transform(X_pre)
    pred = rf.predict(X_pre)[0]
    return float(pred)

def ask(row_dict, max_new_tokens=276):
    pred = predict_with_rf(row_dict)
    sys = "You are a small language model for latency explanation. Output strict JSON: prediction, analysis, explanation."
    user = f"Row: {json.dumps(row_dict)}\nPrediction: {pred:.2f} ms\nRespond in JSON."
    prompt = f"<|system|>\n{sys}\n<|user|>\n{user}\n<|assistant|>\n"
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip()

print(ask({"Hour": 18, "SignalStrength(dBm)": -78, "Bandwidth(Mbps)": 35.0, "Protocol": "5G"}))