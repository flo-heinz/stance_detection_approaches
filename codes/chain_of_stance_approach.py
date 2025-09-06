"""
This script was generated with the assistance of ChatGPT based on the following
kinds of prompts/instructions:

1. General request / purpose
   - "Write a Python script that uses the Ollama API with the model mistral:latest
      to evaluate research article abstracts and classify their stance towards
      renewable energy technologies using a chain-of-stance reasoning approach."

2. Data handling
   - "The script should load input data from a JSON file containing titles, abstracts,
      and stance labels, then produce an output JSON file with predictions, stance
      scores, and categories."

3. Output format constraints
   - "The LLM must return only a JSON object with two keys:
        'stance_score': float between -1.0 and 1.0
        'stance_category': one of
        ['Strongly Pro', 'Pro', 'Neutral', 'Contra', 'Strongly Contra', 'Irrelevant']"

4. Prompt design
   - "Include an instruction header that defines the task and the valid JSON schema.
      Add a chain-of-stance reasoning scaffold to encourage structured internal reasoning
      (context → viewpoint → tone → comparison → inference → final score)."

   Prompt structure:
   -----------------
   Instruction text →
   Chain-of-stance scaffold →
   Target item (title + abstract) →
   'Output:' marker where the model must return JSON only

5. API details
   - "Use Ollama’s HTTP API at http://localhost:11434/api/chat with POST requests.
      Set model='mistral:latest', stream=False, and pass messages with 'system' and 'user'."

6. Error handling & postprocessing
   - "Write a helper function to extract valid JSON from the model response.
      If JSON parsing fails, fall back to extracting a numeric score and map it
      to the closest stance category."

7. Context budgeting
   - "Use the Hugging Face tokenizer to count tokens, and trim the abstract so that
      (system + user + reply) ≤ NUM_CTX."

8. Main loop & saving
   - "Iterate over all abstracts in the input file, run predictions, and save results
      to a new JSON file with stance predictions alongside gold labels."

9. Developer experience
   - "Show progress with tqdm, print status messages, and wait a short time between API calls
      to avoid overload."
"""

from __future__ import annotations

import json
import os
import re
import time
import html
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer  # Hugging Face tokenizer

# Configuration
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME          = "mistral:latest"
CODES_DIR = Path(__file__).resolve().parent
DATA_DIR  = CODES_DIR.parent / "data"
DATA_FILE = DATA_DIR / "evaluation_part.json"
OUTPUT_FILE = DATA_DIR / "NLP-Predictions_mistral_chain_of_stance.json"



SLEEP_BETWEEN_CALLS = 1.0   # avoid overloading Ollama
REQUEST_TIMEOUT     = 500
TEMPERATURE         = 0.0   # deterministic outputs
NUM_CTX             = 4096  # context window
REPLY_HEADROOM      = 96    # reserved tokens for model reply

ALLOWED_CATEGORIES = {
    "Strongly Pro", "Pro", "Neutral", "Contra", "Strongly Contra", "Irrelevant"
}

# Tokenizer (for context budgeting)
_TOKENIZER = None
def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", use_fast=True
        )
    return _TOKENIZER

def token_len(text: str) -> int:
    tok = get_tokenizer()
    return len(tok(text, add_special_tokens=False)["input_ids"])

# Prompt construction
def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()

def _prompt_intro() -> str:
    return (
        "Question: Is the technology or solution described in the paper environmentally friendly?\n\n"
        'Return ONLY a JSON object with keys "stance_score" (float in [-1.0,1.0]) and '
        '"stance_category" (one of ["Strongly Pro","Pro","Neutral","Contra","Strongly Contra","Irrelevant"]).\n\n'
    )

# Chain-of-stance reasoning scaffold (internal model use only)
CHAIN_OF_STANCE = """
Analyze the abstract below using the following reasoning steps:

1. Context Understanding: What domain is the paper in? What is the solution proposed?
2. Main Viewpoint: What is the core idea or conclusion?
3. Tone and Emotion: What emotional or evaluative language is used?
4. Stance Comparison: Compare the text to each possible stance (positive, neutral, negative).
5. Logical Inference: Based on the above, what is the likely position of the paper?
6. Final Score: Output a stance score from –1.0 to +1.0, and provide one-sentence justification.

Important: Perform these steps internally. Do NOT reveal the steps or the justification in your final output.
""".strip()

SYSTEM_PROMPT = (
    'Return only a single valid JSON object with keys "stance_score" and '
    '"stance_category". No extra text.'
)

def build_user_prompt(title: str, abstract: str) -> str:
    title_clean = strip_html(title)
    abstract_clean = strip_html(abstract)
    return (
        f"{_prompt_intro()}"
        f"{CHAIN_OF_STANCE}\n\n"
        f"Text:\nTitle: {title_clean}\nAbstract: {abstract_clean}\n\n"
        "Output:"
    )

# Context budgeting (trim abstract if needed)
def fit_prompt_to_budget(title: str, abstract: str) -> str:
    user_prompt = build_user_prompt(title, abstract)
    sys_tokens = token_len(SYSTEM_PROMPT)
    user_tokens = token_len(user_prompt)
    budget = NUM_CTX - REPLY_HEADROOM

    if sys_tokens + user_tokens <= budget:
        return user_prompt

    # If too long → binary search for max fitting abstract length
    tok = get_tokenizer()
    abs_ids = tok(strip_html(abstract), add_special_tokens=False)["input_ids"]

    lo, hi = 0, len(abs_ids)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_abs = tok.decode(abs_ids[:mid], skip_special_tokens=True)
        candidate_user = build_user_prompt(
            title, candidate_abs + (" …" if mid < len(abs_ids) else "")
        )
        total = sys_tokens + token_len(candidate_user)
        if total <= budget:
            best = candidate_user
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best else build_user_prompt(title, "")

# Call Ollama API
def call_ollama(user_prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")

# Post-processing helpers
def _category_from_score(score: float) -> str:
    if abs(score) < 1e-9: return "Irrelevant"
    if score <= -0.75:   return "Strongly Contra"
    if score <= -0.25:   return "Contra"
    if score < 0.25:     return "Neutral"
    if score < 0.75:     return "Pro"
    return "Strongly Pro"

def extract_json(content: str) -> Dict[str, Any]:
    fallback = {"stance_score": 0.0, "stance_category": "Irrelevant"}
    if not content:
        return fallback
    try:
        i = content.index("{"); j = content.rindex("}") + 1
        obj = json.loads(content[i:j])
    except Exception:
        # fallback: look for a number in plain text
        m = re.search(r"-?\d+(?:\.\d+)?", content)
        if not m:
            return fallback
        sc = max(-1.0, min(1.0, float(m.group(0))))
        return {"stance_score": round(sc, 3), "stance_category": _category_from_score(sc)}

    try:
        score = float(obj.get("stance_score", 0.0))
    except Exception:
        score = 0.0
    score = max(-1.0, min(1.0, score))

    cat = str(obj.get("stance_category", "Irrelevant")).strip()
    if cat not in ALLOWED_CATEGORIES:
        cat = _category_from_score(score)

    return {"stance_score": round(score, 3), "stance_category": cat}

# Main execution
def main() -> None:
    df = pd.read_json(DATA_FILE)
    results: list[dict[str, Any]] = []

    print(f"Using Ollama model: {MODEL_NAME} @ {OLLAMA_HOST}")
    with tqdm(total=len(df), desc="🔍 Evaluating", unit="it") as pbar:
        for _, row in df.iterrows():
            title    = row.get("title", "")
            abstract = row.get("abstract", "")
            gold     = row.get("stance", None)

            user_prompt = fit_prompt_to_budget(title, abstract)
            try:
                content = call_ollama(user_prompt)
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

            time.sleep(SLEEP_BETWEEN_CALLS)
            pbar.update(1)

    out = Path(OUTPUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ Saved predictions to", out)

if __name__ == "__main__":
    main()
