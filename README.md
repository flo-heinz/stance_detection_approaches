# Stance Detection Approaches for Between-Document Analysis

This repository contains data and code for detecting **stance** in scientific abstracts related to environmental sustainability.

It supports two main methods:

- **Supervised regression** using SciBERT
- **Prompt-based inference** using Mistral-7B (via [Ollama](https://ollama.ai)) with:
  - Zero-shot prompting
  - Few-shot prompting (10, 30, 50 examples)
  - Chain-of-Stance prompting

---

## Repository Structure

### `codes/` — **Core scripts**

- `model_train.py` — Fine-tunes SciBERT on stance-labeled abstracts
- `zero_shot_approach.py` — Zero-shot prompting with Mistral-7B
- `few_shot_approach_10_stances.py` — Few-shot prompting (10 examples)
- `few_shot_approach_30_stances.py` — Few-shot prompting (30 examples)
- `few_shot_approach_50_stances.py` — Few-shot prompting (50 examples)
- `chain_of_stance_approach.py` — Chain-of-Stance prompting strategy
- `make_predictions.py` — Helper script for batch predictions
- `evaluation.py` — Evaluation script (F1, MAE, MSE, Pearson, Spearman)

---

### `data/` — **Data and annotations**

The dataset contains a total of 200 annotated scientific abstracts:  
- **100** abstracts are used for training and few-shot examples  
- **100** abstracts are reserved as a held-out evaluation set


#### `data/json/` — **Raw annotated abstracts**

Each abstract is stored in a separate `.json` file. Example format:

```json
{
  "title": "Paper title",
  "abstract": "Abstract text...",
  "stance": 0.5
}
```

---

#### `data/outputs/` — **Predictions and evaluation results**

- `NLP-Predictions_mistral_zero_shot.json` — Predictions from Mistral-7B using zero-shot prompting.  
- `NLP-Predictions_mistral_few_shot_10.json` — Predictions from Mistral-7B with 10 in-context examples.  
- `NLP-Predictions_mistral_few_shot_30.json` — Predictions from Mistral-7B with 30 in-context examples.  
- `NLP-Predictions_mistral_few_shot_50.json` — Predictions from Mistral-7B with 50 in-context examples.  
- `NLP-Predictions_mistral_chain_of_stance.json` — Predictions from Mistral-7B using the Chain-of-Stance reasoning approach.  
- `NLP-Predictions_SciBERT-regression.json` — Predictions from the fine-tuned SciBERT regression baseline.  
- `evaluation_results.txt` — Text summary of all evaluation metrics (classification + regression) for every method.  
- `all_json_in_one.json` — Unified file merging gold labels and predictions from all methods, useful for joint comparison and plotting.  

---

#### `data/outputs/figures/` — **Evaluation plots**

- `accuracy_comparison.png` — Comparison of raw accuracy across methods.  
- `f1_macro_comparison.png` — Macro-averaged F1 score (treats all classes equally).  
- `f1_micro_comparison.png` — Micro-averaged F1 score (aggregates across all samples).  
- `f1_weighted_comparison.png` — Weighted F1 score (accounts for class imbalance).  
- `per_class_f1_grouped.png` — Per-class F1 performance, grouped for direct comparison of stance categories.  
- `regression_metrics_all_in_one.png` — Combined bar chart of regression metrics (MAE, MSE, Pearson’s *r*, Spearman’s ρ).  


---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create virtual environment & install dependencies


python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


### 3. Pull and serve Mistral model via Ollama

```bash
ollama pull mistral
ollama serve
```

---

## Running Experiments

### Train SciBERT

```bash
python codes/model_train.py \
  --train data/training_part.json \
  --test data/evaluation_part.json \
  --out data/outputs/NLP-Predictions_SciBERT-regression.json
```

### Zero-Shot Prompting

```bash
python codes/zero_shot_approach.py \
  --input data/evaluation_part.json \
  --output data/outputs/NLP-Predictions_mistral_zero_shot.json
```

### Few-Shot Prompting (example: 30 examples)

```bash
python codes/few_shot_approach_30_stances.py \
  --input data/evaluation_part.json \
  --output data/outputs/NLP-Predictions_mistral_few_shot_30.json
```

### Chain-of-Stance Prompting

```bash
python codes/chain_of_stance_approach.py \
  --input data/evaluation_part.json \
  --output data/outputs/NLP-Predictions_mistral_chain_of_stance.json
```

---

## Evaluation

To evaluate predictions and generate plots:

```bash
python codes/evaluation.py \
  --predictions_dir data/outputs \
  --gold data/evaluation_part.json \
  --out_dir data/outputs
```



---

## Notes

- Prompting uses [Ollama](https://ollama.ai) to run Mistral-7B locally.
- Evaluation covers both classification and regression metrics.
- Outputs are saved in `data/outputs/` and `data/outputs/figures/`.


