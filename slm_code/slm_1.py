import os, json
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


BASE_MODEL = "distilgpt2"
CORPUS = r"C:\Users\sripa\OneDrive\Desktop\slm\code_train\distill\corpus.jsonl"  # produced by your distill step
OUTDIR = "slm/out-lora-no-trl"
MAX_LEN = 1024

SYSTEM_PROMPT = (
    "You are a small language model that predicts a numeric target from a tabular row, "
    "analyzes data quality, and explains the key drivers. "
    "Output strict JSON with fields: prediction, analysis, explanation."
)
SYSTEM_TAG = "<|system|>\n"
USER_TAG = "<|user|>\n"
ASSISTANT_TAG = "<|assistant|>\n"

def format_example(ex: Dict[str, Any]) -> Dict[str, str]:
    row  = json.dumps(ex["prompt"]["input"]["row"], ensure_ascii=False)
    pred = ex["output"]["prediction"]
    ana  = ex["output"]["analysis"]
    exp  = ex["output"]["explanation"]
    user = f"Row: {row}\nPlease respond strictly in JSON."
    assistant = json.dumps({"prediction": pred, "analysis": ana, "explanation": exp}, ensure_ascii=False)
    text = (
        f"{SYSTEM_TAG}{SYSTEM_PROMPT}\n"
        f"{USER_TAG}{user}\n"
        f"{ASSISTANT_TAG}{assistant}"
    )
    return {"text": text}

print("Loading corpus:", CORPUS)
ds = load_dataset("json", data_files=CORPUS, split="train")
ds = ds.map(format_example, remove_columns=ds.column_names)


print("Loading tokenizer/model (CPU)...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# Load the smaller DistilGPT-2 model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,   # CPU
    low_cpu_mem_usage=True
)
base_model.resize_token_embeddings(len(tok))

# Correct target modules for DistilGPT-2
lora_cfg = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["attn.c_attn", "mlp.c_fc"]  # Adjust target layers
)
model = get_peft_model(base_model, lora_cfg)

def tokenize(batch):
    return tok(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False
    )

ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

@dataclass
class DataCollatorMaskAssistantOnly:
    tokenizer: Any
    response_template: str = ASSISTANT_TAG
    max_length: int = MAX_LEN
    def __post_init__(self):
        self.template_ids = self.tokenizer(self.response_template, add_special_tokens=False)["input_ids"]
    def _find_subsequence(self, source: List[int], subseq: List[int]) -> int:
        n, m = len(source), len(subseq)
        if m == 0 or m > n:
            return -1
        for i in range(n - m + 1):
            if source[i:i+m] == subseq:
                return i
        return -1
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = self._find_subsequence(ids, self.template_ids)
            if start == -1:
                labels[i, :] = -100
                continue
            end = start + len(self.template_ids)
            labels[i, :end] = -100
        batch["labels"] = labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

collator = DataCollatorMaskAssistantOnly(tokenizer=tok)
args = TrainingArguments(
    output_dir=OUTDIR,
    per_device_train_batch_size=15,
    gradient_accumulation_steps=25,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=100,
    save_steps=500,
    bf16=False, fp16=False,
    report_to=[],
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=ds_tok,
)
trainer.train()
print("Saving adapters/tokenizer to", OUTDIR)
trainer.save_model(OUTDIR)
tok.save_pretrained(OUTDIR)
print("Done.")