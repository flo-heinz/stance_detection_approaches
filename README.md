# Stance Detection Approaches for Between-Document Analysis

This repository contains data and code for detecting **stance** in scientific abstracts on environmental sustainability.  
It includes both a **supervised regression baseline** with SciBERT and **prompt-based inference** with Mistral-7B (via [Ollama](https://ollama.ai)), using zero-shot, few-shot, and **Chain-of-Stance** prompting strategies.

---

## Repository Structure

### `codes/`
Python scripts for training, inference, and evaluation:

- `model_train.py` — Fine-tunes SciBERT on training data for stance regression.  
- `zero_shot_approach.py` — Runs Mistral-7B with zero-shot prompting.  
- `few_shot_approach_10_stances.py` — Few-shot prompting with 10 demonstrations.  
- `few_shot_approach_30_stances.py` — Few-shot prompting with 30 demonstrations.  
- `few_shot_approach_50_stances.py` — Few-shot prompting with 50 demonstrations.  
- `chain_of_stance_approach.py` — Implements structured Chain-of-Stance reasoning.  
- `make_predictions.py` — Helper for running batch predictions.  
- `evaluation.py` — Computes metrics and generates plots.  

### `data/`
Annotated data, splits, and outputs:

- `training_part.json` — Training split (used for SciBERT and few-shot examples).  
- `evaluation_part.json` — Evaluation split (held-out test set).  

#### `data/json/`
Individual annotated abstracts in JSON format:
```json
{
  "title": "Paper title",
  "abstract": "Abstract text...",
  "stance": 0.5
}
