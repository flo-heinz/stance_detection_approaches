"""
This script was generated with the assistance of ChatGPT based on the following
kinds of prompts/instructions:

1. General request / purpose
   - "Write a Python script that uses the Ollama API with the model mistral:latest
      to evaluate research article abstracts and classify their stance towards
      renewable energy technologies."

2. Data handling
   - "The script should load input data from a JSON file containing titles, abstracts,
      and stance labels, then produce an output JSON file with predictions, stance
      scores, and categories."

3. Output format constraints
   - "The LLM must return only a JSON object with two keys:
        'stance_score': float between -1.0 and 1.0
        'stance_category': one of
        ['Strongly Pro', 'Pro', 'Neutral', 'Contra', 'Strongly Contra', 'Irrelevant']"

4. Few-shot prompting
   - "Use a clear instruction block that defines the task, the valid output schema,
      and how to handle irrelevant abstracts. Provide multiple few-shot examples
      (title + abstract + gold stance) to guide the model’s behavior before
      asking it to evaluate the target abstract."

   Prompt structure:
   -----------------
   Instruction text →
   Few-shot examples (from FEW_SHOT_EXAMPLES) →
   Target item to classify (title + abstract) →
   'Output:' marker where the model must return JSON only

5. API details
   - "Use Ollama’s HTTP API at http://localhost:11434/api/chat with POST requests.
      Set model='mistral:latest', stream=False, and pass messages with 'system' and 'user'."

6. Error handling & postprocessing
   - "Write a helper function to extract valid JSON from the model response.
      If JSON parsing fails, fall back to extracting a numeric score and map it
      to the closest stance category."

7. Main loop & saving
   - "Iterate over all abstracts in the input file, run predictions, and save results
      to a new JSON file with stance predictions alongside gold labels."

8. Developer experience
   - "Show progress with tqdm, print status messages (using emojis is fine),
      and wait a short time between API calls to avoid overload."
"""

import re
import json
import time
import html
import requests
import pandas as pd
from tqdm import tqdm
from typing import List

# Simple heuristic token length estimator (no transformers dependency)
def token_len(text: str) -> int:
    # Approximate 4 characters ≈ 1 token
    return max(1, len(text) // 4)


# Configuration
OLLAMA_HOST         = "http://localhost:11434"
MODEL_NAME          = "mistral:latest"
DATA_FILE           = r"C:\Users\flori\OneDrive\Desktop\Programmier Pro\data\evaluation_part.json"
OUTPUT_FILE         = r"C:\Users\flori\OneDrive\Desktop\Programmier Pro\NLP-Predictions_mistral_few_shot_30.json"

SLEEP_BETWEEN_CALLS = 1.0
REQUEST_TIMEOUT     = 1000
TEMPERATURE         = 0.0
NUM_CTX             = 8000
REPLY_HEADROOM      = 96

# Allowed stance labels for model output
ALLOWED_CATEGORIES = {
    "Strongly Pro", "Pro", "Neutral", "Contra", "Strongly Contra", "Irrelevant"
}

# Few-shot examples for guiding the model
FEW_SHOT_EXAMPLES = [
  {
    "title":"PdNi Biatomic Clusters from Metallene Unlock Record‐Low Onset Dehydrogenation Temperature for Bulk‐MgH<sub>2</sub>",
    "abstract":"Abstract Hydrogen storage has long been a priority on the renewable energy research agenda. Due to its high volumetric and gravimetric hydrogen density, MgH 2 is a desirable candidate for solid‐state hydrogen storage. However, its practical use is constrained by high thermal stability and sluggish kinetics. Here, PdNi bilayer metallenes are reported as catalysts for hydrogen storage of bulk‐MgH 2 near ambient temperature. Unprecedented 422 K beginning dehydrogenation temperature and up to 6.36 wt.% reliable hydrogen storage capacity are achieved. Fast hydrogen desorption is also provided by the system (5.49 wt.% in 1 h, 523 K). The in situ generated PdNi alloy clusters with suitable d ‐band centers are identified as the main active sites during the de/re‐hydrogenation process by aberration‐corrected transmission electron microscopy and theoretical simulations, while other active species including Pd/Ni pure phase clusters and Pd/Ni single atoms obtained via metallene ball milling, also enhance the reaction. These findings present fundamental insights into active species identification and rational design of highly efficient hydrogen storage materials.",
    "stance":0.4
  },
  {
    "title":"Low‐Cost Hydrogen Production from Alkaline/Seawater over a Single‐Step Synthesis of Mo<sub>3</sub>Se<sub>4</sub>‐NiSe Core–Shell Nanowire Arrays",
    "abstract":"Abstract The rational design and steering of earth‐abundant, efficient, and stable electrocatalysts for hydrogen generation is highly desirable but challenging with catalysts free of platinum group metals (PGMs). Mass production of high‐purity hydrogen fuel from seawater electrolysis presents a transformative technology for sustainable alternatives. Here, a heterostructure of molybdenum selenide‐nickel selenide (Mo 3 Se 4 ‐NiSe) core–shell nanowire arrays constructed on nickel foam by a single‐step in situ hydrothermal process is reported. This tiered structure provides improved intrinsic activity and high electrical conductivity for efficient charge transfer and endows excellent hydrogen evolution reaction (HER) activity in alkaline and natural seawater conditions. The Mo 3 Se 4 ‐NiSe freestanding electrodes require small overpotentials of 84.4 and 166 mV to reach a current density of 10 mA cm −2 in alkaline and natural seawater electrolytes, respectively. It maintains an impressive balance between electrocatalytic activity and stability. Experimental and theoretical calculations reveal that the Mo 3 Se 4 ‐NiSe interface provides abundant active sites for the HER process, which modulate the binding energies of adsorbed species and decrease the energetic barrier, providing a new route to design state‐of‐the‐art, PGM‐free catalysts for hydrogen production from alkaline and seawater electrolysis.",
    "stance":0.6
  },
  {
    "title":"Rapid Synthesis of Trimetallic Nanozyme for Sustainable Cascaded Catalytic Therapy via Tumor Microenvironment Remodulation",
    "abstract":"Abstract Tumor microenvironment (TME)‐responsive nanozyme‐catalyzed cancer therapy shows great potential due to its specificity and efficiency. However, breaking the self‐adaption of tumors and improving the sustainable remodeling TME ability remains a major challenge for developing novel nanozymes. Here, a rapid method is developed first to synthesize unprecedented trimetalic nanozyme (AuMnCu, AMC) with a targeting peptide (AMCc), which exhibits excellent peroxidase‐like, catalase‐like, and glucose oxidase‐like activities. The released Cu and Mn ions in TME consume endogenous H 2 O 2 and produce O 2 , while the AMCccatalyzes glucose oxidation reaction to generate H 2 O 2 and gluconic acid, which achieves the starvation therapy by depleting the energy and enhances the chemodynamic therapy effect by lowering the pH of the TME and producing extra H 2 O 2 . Meanwhile, the reactive oxygen species damage is amplified, as AMCc can constantly oxidize intracellular reductive glutathione through the cyclic valence alternation of Cu and Mn ions, and the generated Cu + elevate the production of ·OH from H 2 O 2 . Further studies depict that the well‐designed AMCc exhibits the excellent photothermal performance and achieves TME‐responsive sustainable starvation/photothermal‐enhanced chemodynamic synergistic effects in vitro and in vivo. Overall, a promising approach is demonstrated here to design “all‐in‐one” nanozyme for theranostics by remodeling the TME.",
    "stance":0.0
  }
]

# Remove HTML tags and clean whitespace
def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()

# Map numeric stance score to discrete category
def map_category(score: float) -> str:
    if abs(score) < 1e-6: return "Irrelevant"
    if score <= -0.75:   return "Strongly Contra"
    if score <= -0.25:   return "Contra"
    if score < 0.25:     return "Neutral"
    if score < 0.75:     return "Pro"
    return "Strongly Pro"

# Build a formatted few-shot example block
def format_example_block(ex) -> str:
    title = strip_html(ex.get("title", ""))
    abstract = strip_html(ex.get("abstract", ""))
    score = float(ex.get("stance", 0.0))
    out = {"stance_score": round(score, 3), "stance_category": map_category(score)}
    return f"Text:\nTitle: {title}\nAbstract: {abstract}\nOutput:\n{json.dumps(out, ensure_ascii=False)}"

# Prompt header with instructions and schema
def _prompt_intro() -> str:
    return (
        "Question: Is the technology or solution described in the paper environmentally friendly?\n\n"
        'Return ONLY a JSON object with keys "stance_score" (float in [-1.0,1.0]) and '
        '"stance_category" (one of ["Strongly Pro","Pro","Neutral","Contra","Strongly Contra","Irrelevant"]).\n\n'
        "Examples:\n"
    )

# Prompt for one input item (title + abstract)
def _prompt_user(title: str, abstract: str) -> str:
    title_clean = strip_html(title)
    abstract_clean = strip_html(abstract)
    return (
        f"\n\nNow evaluate the following:\n"
        f"Title: {title_clean}\n"
        f"Abstract: {abstract_clean}\n"
        "Output:"
    )

# Build a prompt with as many few-shots as possible without exceeding context
def build_prompt_fit_tokenizer(title: str, abstract: str, num_ctx: int, reply_headroom: int) -> tuple[str, int, int]:
    intro = _prompt_intro()
    user = _prompt_user(title, abstract)
    budget = num_ctx - reply_headroom

    base = intro + user
    base_tokens = token_len(base)
    if base_tokens > budget:
        return base, 0, base_tokens

    selected: List[str] = []
    running = base_tokens
    for ex in FEW_SHOT_EXAMPLES:
        blk = "\n\n" + format_example_block(ex)
        blk_tokens = token_len(blk)
        if running + blk_tokens <= budget:
            selected.append(blk)
            running += blk_tokens
        else:
            break

    prompt = intro + "".join(selected) + user
    return prompt, len(selected), running

# Extract JSON safely from model response
def extract_json(content: str):
    fallback = {"stance_score": 0.0, "stance_category": "Irrelevant"}
    if not content:
        return fallback
    try:
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

# Makes a request to the Ollama API with the prompt
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

            prompt, n_used, tokens_used = build_prompt_fit_tokenizer(title, abstract, NUM_CTX, REPLY_HEADROOM)

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

            time.sleep(SLEEP_BETWEEN_CALLS)
            pbar.update(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ Saved predictions to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
