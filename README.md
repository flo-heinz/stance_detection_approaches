# Stance Detection Approaches for Between-Document Analysis

This repository contains data and code for detecting **stance** in scientific abstracts on environmental sustainability.  
It includes both a **supervised regression baseline** with SciBERT and **prompt-based inference** with Mistral-7B (via [Ollama](https://ollama.ai)), using zero-shot, few-shot, and **Chain-of-Stance** prompting strategies.

---

## Repository Structure

### `codes/`
Python scripts for training, inference, and evaluation:

- `chain_of_stance_approach.py` — Implements structured Chain-of-Stance reasoning.  
- `evaluation.py` — Computes metrics (F1, MAE, MSE, Pearson, Spearman) and generates plots.  
- `few_shot_approach_10_stances.py` — Few-shot prompting with 10 demonstrations.  
- `few_shot_approach_30_stances.py` — Few-shot prompting with 30 demonstrations.  
- `few_shot_approach_50_stances.py` — Few-shot prompting with 50 demonstrations.  
- `make_predictions.py` — Helper for running batch predictions.  
- `model_train.py` — Fine-tunes SciBERT on training data for stance regression.  
- `zero_shot_approach.py` — Runs Mistral-7B with zero-shot prompting.  

---

### `data/`
Annotated data, raw inputs, and experiment outputs:

- `training_part.json` — Training split (used for SciBERT and few-shot examples).  
- `evaluation_part.json` — Evaluation split (held-out test set).  

#### `data/json/`
Raw annotated abstracts in JSON format (individual files). Example entry:
```json
{
  "title": "Paper title",
  "abstract": "Abstract text...",
  "stance": 0.5
}

### `data/outputs/`

**Prediction files and evaluation summaries:**
- `NLP-Predictions_mistral_zero_shot.json`
- `NLP-Predictions_mistral_few_shot_10.json`
- `NLP-Predictions_mistral_few_shot_30.json`
- `NLP-Predictions_mistral_few_shot_50.json`
- `NLP-Predictions_mistral_chain_of_stance.json`
- `NLP-Predictions_SciBERT-regression.json`
- `evaluation_results.txt` — Evaluation summary
- `all_json_in_one.json` — Combined predictions

### `data/outputs/figures/`

**Generated plots:**
- `accuracy_comparison.png`
- `f1_macro_comparison.png`
- `f1_micro_comparison.png`
- `f1_weighted_comparison.png`
- `per_class_f1_grouped.png`
- `regression_metrics_all_in_one.png`


## Quick Start

# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create environment & install dependencies
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Pull the Mistral model via Ollama
ollama pull mistral
ollama serve

## Running Experiments

### Train SciBERT

python codes/model_train.py \
  --train data/training_part.json \
  --test data/evaluation_part.json \
  --out data/outputs/NLP-Predictions_SciBERT-regression.json


### Zero-shot Prompting

python codes/zero_shot_approach.py \
  --input data/evaluation_part.json \
  --output data/outputs/NLP-Predictions_mistral_zero_shot.json


### Few-Shot Prompting (example: 30 shots)

python codes/few_shot_approach_30_stances.py \
  --input data/evaluation_part.json \
  --output data/outputs/NLP-Predictions_mistral_few_shot_30.json


### Chain-of-Stance Prompting

python codes/chain_of_stance_approach.py \
  --input data/evaluation_part.json \
  --output data/outputs/NLP-Predictions_mistral_chain_of_stance.json


### Evalution:

python codes/evaluation.py \
  --predictions_dir data/outputs \
  --gold data/evaluation_part.json \
  --out_dir data/outputs





