from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
TextBatch = Dict[str, List[str]]

SYSTEM_PROMPT = (
    "You are a small language model that predicts a numeric target from a tabular row, "
    "analyzes data quality, and explains the key drivers. "
    "Output strict JSON with fields: prediction, analysis, explanation."
)
SYSTEM_TAG = "<|system|>\n"
USER_TAG = "<|user|>\n"
ASSISTANT_TAG = "<|assistant|>\n"

@dataclass(frozen=True)
class TrainPaths:
    corpus_jsonl: str
    out_dir: str = "slm/out-lora-no-trl"
    base_model: str = "distilgpt2"
    max_len: int = 512  # smaller sequence length for CPU memory

def _format_example(ex: Dict[str, Any]) -> Dict[str, str]:
    row = json.dumps(ex["prompt"]["input"]["row"], ensure_ascii=False)
    pred = ex["output"]["prediction"]
    ana = ex["output"]["analysis"]
    exp = ex["output"]["explanation"]
    user = f"Row: {row}\nPlease respond strictly in JSON."
    assistant = json.dumps(
        {"prediction": pred, "analysis": ana, "explanation": exp}, ensure_ascii=False
    )
    text = f"{SYSTEM_TAG}{SYSTEM_PROMPT}\n{USER_TAG}{user}\n{ASSISTANT_TAG}{assistant}"
    return {"text": text}
def _tokenize(tok, batch: TextBatch, max_len: int) -> Dict[str, Any]:
    return tok(batch["text"], truncation=True, max_length=max_len, padding=False)

class DataCollatorMaskAssistantOnly:
    def __init__(self, tokenizer, response_template: str = ASSISTANT_TAG, max_length: int = 512):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.max_length = max_length
        self.template_ids: List[int] = tokenizer(
            response_template, add_special_tokens=False
        )["input_ids"]

    @staticmethod
    def _find_subsequence(source: Sequence[int], subseq: Sequence[int]) -> int:
        n, m = len(source), len(subseq)
        if m == 0 or m > n:
            return -1
        for i in range(n - m + 1):
            if list(source[i : i + m]) == list(subseq):
                return i
        return -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: torch.Tensor = batch["attention_mask"]

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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def train_lora(paths: TrainPaths) -> None:
    print("Loading corpus:", paths.corpus_jsonl)
    ds: Dataset = load_dataset("json", data_files=paths.corpus_jsonl, split="train")
    ds = ds.map(_format_example, remove_columns=ds.column_names)
    print("Loading tokenizer/model (CPU)...")
    tok = AutoTokenizer.from_pretrained(paths.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        paths.base_model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    base_model.resize_token_embeddings(len(tok))
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["attn.c_attn", "mlp.c_fc"],
    )
    model = get_peft_model(base_model, lora_cfg)
    ds_tok = ds.map(lambda b: _tokenize(tok, b, paths.max_len), batched=True, remove_columns=["text"])
    collator = DataCollatorMaskAssistantOnly(tokenizer=tok, max_length=paths.max_len)
    args = TrainingArguments(
        output_dir=paths.out_dir,
        per_device_train_batch_size=8,  # tune to your RAM; 8 is a starting point
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=500,
        bf16=False,
        fp16=False,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=ds_tok)
    trainer.train()
    print("Saving adapters/tokenizer to", paths.out_dir)
    trainer.save_model(paths.out_dir)
    tok.save_pretrained(paths.out_dir)
    print("Done.")
if __name__ == "__main__":
    train_lora(TrainPaths(corpus_jsonl=r"distill\corpus.jsonl"))