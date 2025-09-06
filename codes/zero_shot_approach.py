"""
This zero-shot script was created with ChatGPT from the following prompt themes:

1) Goal
   - "Use Ollama (model: mistral:latest) to score each paper abstract for stance
      toward environmentally friendly/renewable tech, zero-shot (no examples)."

2) I/O
   - "Read titles/abstracts/labels from a JSON file; write a JSON file with
      predicted stance_score and stance_category."

3) Output schema (strict)
   - "Return ONLY a JSON object:
        'stance_score'  ∈ [-1.0, 1.0]
        'stance_category' ∈
        ['Strongly Pro','Pro','Neutral','Contra','Strongly Contra','Irrelevant']"

4) Prompt shape
   - "Short instruction header that defines the task + schema,
      then the single target item (title + abstract) followed by 'Output:'."

5) API usage
   - "POST to http://localhost:11434/api/chat with messages [{system},{user}],
      model='mistral:latest', stream=False, temperature=0.0."

6) Robustness
   - "If the model returns extra text, extract the first JSON block; validate and
      clamp stance_score; if invalid, fall back to mapping from score or Irrelevant."

7) Tokenization helper
   - "Use transformers AutoTokenizer (Mistral) to estimate token counts; keep the
      prompt compact for zero-shot."

8) UX
   - "Show tqdm progress and add a short delay between calls."
"""

import re
import json
import time
import html
import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Tokenizer for estimating token counts with Mistral
from transformers import AutoTokenizer
_TOKENIZER = None

def get_tokenizer():
    """Load and cache the Mistral tokenizer once."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            use_fast=True
        )
    return _TOKENIZER

def token_len(text: str) -> int:
    """Return number of tokens in text (no special tokens)."""
    tok = get_tokenizer()
    return len(tok(text, add_special_tokens=False)["input_ids"])

# Configuration
OLLAMA_HOST         = "http://localhost:11434"
MODEL_NAME          = "mistral:latest"
# Resolve paths relative to this script
CODES_DIR = Path(__file__).resolve().parent
DATA_DIR  = CODES_DIR.parent / "data"
DATA_FILE = DATA_DIR / "evaluation_part.json"
OUTPUT_FILE = DATA_DIR / "NLP-Predictions_mistral_zero_shot.json"

SLEEP_BETWEEN_CALLS = 1.0
REQUEST_TIMEOUT     = 500
TEMPERATURE         = 0.0
NUM_CTX             = 4096
REPLY_HEADROOM      = 96  # reserve tokens for the model's reply

# Categories allowed in final output
ALLOWED_CATEGORIES = {
    "Strongly Pro", "Pro", "Neutral", "Contra", "Strongly Contra", "Irrelevant"
}

# Remove HTML tags/entities from input text
def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()

# Map numeric score to the nearest category
def map_category(score: float) -> str:
    if abs(score) < 1e-6: return "Irrelevant"
    if score <= -0.75:   return "Strongly Contra"
    if score <= -0.25:   return "Contra"
    if score < 0.25:     return "Neutral"
    if score < 0.75:     return "Pro"
    return "Strongly Pro"

# Instruction header for the model
def _prompt_intro() -> str:
    return (
        "Question: Is the technology or solution described in the paper environmentally friendly?\n\n"
        'Return ONLY a JSON object with keys "stance_score" (float in [-1.0,1.0]) and '
        '"stance_category" (one of ["Strongly Pro","Pro","Neutral","Contra","Strongly Contra","Irrelevant"]).\n\n'
    )

# User prompt containing the paper’s title + abstract
def _prompt_user(title: str, abstract: str) -> str:
    title_clean = strip_html(title)
    abstract_clean = strip_html(abstract)
    return (
        f"Title: {title_clean}\n"
        f"Abstract: {abstract_clean}\n"
        "Output:"
    )

# Build the full zero-shot prompt
def build_prompt(title: str, abstract: str, num_ctx: int, reply_headroom: int) -> tuple[str, int]:
    intro = _prompt_intro()
    user = _prompt_user(title, abstract)
    prompt = intro + "\n\n" + user
    token_count = token_len(prompt)
    return prompt, token_count

# Try to parse model output into strict JSON
def extract_json(content: str):
    fallback = {"stance_score": 0.0, "stance_category": "Irrelevant"}
    if not content:
        return fallback
    try:
        # extract first {...} block
        start = content.index("{"); end = content.rindex("}") + 1
        obj = json.loads(content[start:end])
    except Exception:
        return fallback
    try:
        score = float(obj.get("stance_score", 0.0))
    except Exception:
        score = 0.0
    score = max(-1.0, min(1.0, score))
    category = str(obj.get("stance_category", "Irrelevant")).strip()
    if category not in ALLOWED_CATEGORIES:
        category = map_category(score)
    return {"stance_score": round(score, 3), "stance_category": category}

# Send a single request to Ollama
def call_ollama(prompt: str):
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": 'Return only a single valid JSON object with keys "stance_score" and "stance_category". No extra text.'},
            {"role": "user",   "content": prompt},
        ],
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")

# Main loop: load data, query model, save predictions
def main():
    df = pd.read_json(DATA_FILE)
    results = []

    with tqdm(total=len(df), desc="🔍 Evaluating", unit="it") as pbar:
        for _, row in df.iterrows():
            title    = row.get("title", "")
            abstract = row.get("abstract", "")
            gold     = row.get("stance", None)

            prompt, tokens_used = build_prompt(title, abstract, NUM_CTX, REPLY_HEADROOM)

            try:
                content = call_ollama(prompt)
                pred = extract_json(content)
            except Exception as e:
                print("Error:", e)
                pred = {"stance_score": 0.0, "stance_category": "Irrelevant"}

            results.append({
                "title": title,
                "abstract": abstract,
                "gold_stance": gold,
                "predicted_stance_score": pred["stance_score"],
                "predicted_stance_category": pred["stance_category"]
            })

            time.sleep(SLEEP_BETWEEN_CALLS)  # avoid overloading Ollama
            pbar.update(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ Saved predictions to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
