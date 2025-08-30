# 🚀 Small Language Model (SLM) for Network Latency Prediction  

> ⚡ Built with the guidance of **Openai.com** and powered by **DistilGPT-2 + LoRA fine-tuning**.  

This project implements a **Small Language Model (SLM)** that can:  
1. **Predict network latency (ms)** from tabular signal metrics.  
2. **Analyze data quality issues** in the input row.  
3. **Explain key drivers** behind the predictions using model interpretability techniques.  

The pipeline combines **classical ML (Random Forest + SHAP)** with **LLM distillation (LoRA-tuned DistilGPT-2)** to create a lightweight yet explainable prediction system.  

---

## ✨ Features

- 📊 **Data Preprocessing & Modeling**  
  - Handles numeric & categorical features via robust preprocessing.  
  - Trains a **Random Forest Regressor** for strong baseline performance.  
  - Persists preprocessing + feature selection artifacts.  

- 🔎 **Explainability with SHAP**  
  - Extracts feature contributions for each prediction.  
  - Converts SHAP insights into natural-language explanations.  

- 🧠 **Distillation into SLM**  
  - Distils ML model outputs into a **DistilGPT-2 backbone**.  
  - Fine-tuned using **LoRA (Low-Rank Adaptation)** for efficiency.  
  - Produces JSON outputs with `prediction`, `analysis`, and `explanation`.  

- 💡 **Inference Ready**  
  - Query the SLM with a row of network metrics.  
  - Get strict JSON responses:  
    ```json
    {
      "prediction": 32.5,
      "analysis": { "missing_fields": [], "note": "Robust scaling applied" },
      "explanation": [
        {"feature": "SignalStrength(dBm)", "impact": -12.4},
        {"feature": "Bandwidth(Mbps)", "impact": 8.9}
      ]
    }
    ```

---

## 🛠️ Technical Breakdown

### 1. `model_code.py` → **Classical ML Backbone**  
- Loads **signal metrics dataset** (`signal_metrics.csv`).  
- Preprocesses:
  - Numeric → Imputation + Robust Scaling.  
  - Categorical → Imputation + One-Hot Encoding.  
- Trains a **RandomForestRegressor** (`190 trees, max_depth=5`).  
- Evaluates with **R² & MAE**.  
- Saves artifacts: `pre.pkl`, `fs.pkl`, `rf.pkl`, `schema.json`.  

---

### 2. `expl_code.py` → **Knowledge Distillation with SHAP**  
- Uses trained RF model to generate predictions.  
- Computes **SHAP values** to identify top-k drivers.  
- Builds a **distillation corpus (`corpus.jsonl`)** with:  
  - Row input.  
  - RF prediction.  
  - Data quality notes.  
  - SHAP explanations.  

---

### 3. `slm_1.py` → **LoRA Fine-Tuning of DistilGPT-2**  
- Loads **DistilGPT-2**.  
- Applies **LoRA adapters** (`attn.c_attn`, `mlp.c_fc`).  
- Trains on `corpus.jsonl` with Hugging Face `Trainer`.  
- Saves tuned adapters + tokenizer → `slm/out-lora-no-trl`.  

---

### 4. `slm2.py` → **Inference with SLM**  
- Loads DistilGPT-2 + LoRA adapters.  
- Constructs structured prompt with system/user tags.  
- Generates **strict JSON responses**.  
- Example query:  
  ```python
  print(ask({"Hour": 18, "SignalStrength(dBm)": -78, "Bandwidth(Mbps)": 35.0, "Protocol": "5G"}))
  ```
  ✅ Output:  
  ```json
  {"prediction": 28.7, "analysis": {...}, "explanation": [...]}
  ```

---

## 📈 Workflow Diagram

```
Raw Data (signal_metrics.csv)
        │
        ▼
 Classical ML (RandomForest) ──▶ Predictions + SHAP Explanations
        │
        ▼
   Distillation Corpus (JSONL)
        │
        ▼
 LoRA-Tuned DistilGPT-2 (SLM)
        │
        ▼
  Inference → JSON Output
```

---

## 📂 Project Structure

```
├── model_code.py      # Train RF model + save preprocessing
├── expl_code.py       # Generate distillation corpus with SHAP
├── slm_1.py           # Fine-tune DistilGPT-2 with LoRA
├── slm2.py            # Inference with SLM
├── distill/           # Saved preprocessing, model, schema, corpus
└── slm/               # LoRA adapter outputs
```

---

## 🚀 Getting Started

1. **Train ML Model**  
   ```bash
   python model_code.py
   ```

2. **Generate Corpus**  
   ```bash
   python expl_code.py
   ```

3. **Fine-Tune SLM**  
   ```bash
   python slm_1.py
   ```

4. **Run Inference**  
   ```bash
   python slm2.py
   ```

---

## 📊 Metrics (Example)

- **Random Forest Baseline**  
  - R² ≈ 0.82  
  - MAE ≈ 5.3 ms  

- **SLM Inference**  
  - Produces interpretable JSON outputs.  
  - Compact, lightweight, explainable.  

---

## 🤝 Acknowledgments

- Built with **transformers**, **scikit-learn**, **SHAP**, **PEFT/LoRA**.  
- Special thanks to **ChatGPT** for guiding the design & implementation.  

---

## 🔮 Future Improvements

- Extend corpus with synthetic scenarios for robustness.  
- Experiment with quantization for deployment efficiency.  
- Add **multi-task support** (e.g., jitter, packet loss prediction).  