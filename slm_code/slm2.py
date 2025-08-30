import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "distilgpt2"
ADAPTER = "slm/out-lora"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER).to("cpu")

def ask(row_dict, max_new_tokens=256):
    sys = "You are a small language model for latency prediction. Output strict JSON: prediction, analysis, explanation."
    user = f"Row: {json.dumps(row_dict)}\nRespond in JSON."
    prompt = f"<|system|>\n{sys}\n<|user|>\n{user}\n<|assistant|>\n"
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip()


print(ask({"Hour": 18, "SignalStrength(dBm)": -78, "Bandwidth(Mbps)": 35.0, "Protocol": "5G"}))