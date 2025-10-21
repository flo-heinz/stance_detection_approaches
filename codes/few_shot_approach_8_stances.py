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
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List

# Tokenizer setup (for keeping prompts within model context)
from transformers import AutoTokenizer
_TOKENIZER = None
def get_tokenizer():
    # Load and cache tokenizer once
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            use_fast=True
        )
    return _TOKENIZER

def token_len(text: str) -> int:
    # Count number of tokens without special tokens
    tok = get_tokenizer()
    return len(tok(text, add_special_tokens=False)["input_ids"])

# Configuration
OLLAMA_HOST         = "http://localhost:11434"
MODEL_NAME          = "mistral:latest"
CODES_DIR = Path(__file__).resolve().parent
DATA_DIR  = CODES_DIR.parent / "data"
DATA_FILE = DATA_DIR / "evaluation_part.json"
OUTPUT_FILE = DATA_DIR / "NLP-Predictions_mistral_few_shot_8.json"

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
    "title":"PdNi Biatomic Clusters from Metallene Unlock Record‐Low Onset Dehydrogenation Temperature for Bulk‐MgH<sub>2<\/sub>",
    "abstract":"Abstract Hydrogen storage has long been a priority on the renewable energy research agenda. Due to its high volumetric and gravimetric hydrogen density, MgH 2 is a desirable candidate for solid‐state hydrogen storage. However, its practical use is constrained by high thermal stability and sluggish kinetics. Here, PdNi bilayer metallenes are reported as catalysts for hydrogen storage of bulk‐MgH 2 near ambient temperature. Unprecedented 422 K beginning dehydrogenation temperature and up to 6.36 wt.% reliable hydrogen storage capacity are achieved. Fast hydrogen desorption is also provided by the system (5.49 wt.% in 1 h, 523 K). The in situ generated PdNi alloy clusters with suitable d ‐band centers are identified as the main active sites during the de\/re‐hydrogenation process by aberration‐corrected transmission electron microscopy and theoretical simulations, while other active species including Pd\/Ni pure phase clusters and Pd\/Ni single atoms obtained via metallene ball milling, also enhance the reaction. These findings present fundamental insights into active species identification and rational design of highly efficient hydrogen storage materials.",
    "stance":0.4
  },
  {
    "title":"Low‐Cost Hydrogen Production from Alkaline\/Seawater over a Single‐Step Synthesis of Mo<sub>3<\/sub>Se<sub>4<\/sub>‐NiSe Core–Shell Nanowire Arrays",
    "abstract":"Abstract The rational design and steering of earth‐abundant, efficient, and stable electrocatalysts for hydrogen generation is highly desirable but challenging with catalysts free of platinum group metals (PGMs). Mass production of high‐purity hydrogen fuel from seawater electrolysis presents a transformative technology for sustainable alternatives. Here, a heterostructure of molybdenum selenide‐nickel selenide (Mo 3 Se 4 ‐NiSe) core–shell nanowire arrays constructed on nickel foam by a single‐step in situ hydrothermal process is reported. This tiered structure provides improved intrinsic activity and high electrical conductivity for efficient charge transfer and endows excellent hydrogen evolution reaction (HER) activity in alkaline and natural seawater conditions. The Mo 3 Se 4 ‐NiSe freestanding electrodes require small overpotentials of 84.4 and 166 mV to reach a current density of 10 mA cm −2 in alkaline and natural seawater electrolytes, respectively. It maintains an impressive balance between electrocatalytic activity and stability. Experimental and theoretical calculations reveal that the Mo 3 Se 4 ‐NiSe interface provides abundant active sites for the HER process, which modulate the binding energies of adsorbed species and decrease the energetic barrier, providing a new route to design state‐of‐the‐art, PGM‐free catalysts for hydrogen production from alkaline and seawater electrolysis.",
    "stance":0.6
  },
  {
    "title":"Rapid Synthesis of Trimetallic Nanozyme for Sustainable Cascaded Catalytic Therapy via Tumor Microenvironment Remodulation",
    "abstract":"Abstract Tumor microenvironment (TME)‐responsive nanozyme‐catalyzed cancer therapy shows great potential due to its specificity and efficiency. However, breaking the self‐adaption of tumors and improving the sustainable remodeling TME ability remains a major challenge for developing novel nanozymes. Here, a rapid method is developed first to synthesize unprecedented trimetalic nanozyme (AuMnCu, AMC) with a targeting peptide (AMCc), which exhibits excellent peroxidase‐like, catalase‐like, and glucose oxidase‐like activities. The released Cu and Mn ions in TME consume endogenous H 2 O 2 and produce O 2 , while the AMCccatalyzes glucose oxidation reaction to generate H 2 O 2 and gluconic acid, which achieves the starvation therapy by depleting the energy and enhances the chemodynamic therapy effect by lowering the pH of the TME and producing extra H 2 O 2 . Meanwhile, the reactive oxygen species damage is amplified, as AMCc can constantly oxidize intracellular reductive glutathione through the cyclic valence alternation of Cu and Mn ions, and the generated Cu + elevate the production of ·OH from H 2 O 2 . Further studies depict that the well‐designed AMCc exhibits the excellent photothermal performance and achieves TME‐responsive sustainable starvation\/photothermal‐enhanced chemodynamic synergistic effects in vitro and in vivo. Overall, a promising approach is demonstrated here to design “all‐in‐one” nanozyme for theranostics by remodeling the TME.",
    "stance":0.0
  },
  {
    "title":"Precipitated Iodine Cathode Enabled by Trifluoromethanesulfonate Oxidation for Cathode\/Electrolyte Mutualistic Aqueous Zn–I Batteries",
    "abstract":"Abstract Aqueous Zn–I batteries hold great potential for high‐safety and sustainable energy storage. However, the iodide shuttling effect and the hydrogen evolution reaction that occur in the aqueous electrolyte remain the main obstacles for their further development. Herein, the design of a cathode\/electrolyte mutualistic aqueous (CEMA) Zn–I battery based on the inherent oxidation ability of aqueous trifluoromethanesulfonate ((OTf) − ) electrolyte toward triiodide species is presented. This results in the formation of iodine sediment particles assembled by fine iodine nanocrystals (≈10 nm). An iodine host cathode with high areal iodine loading is realized via a spontaneous absorption process that enriched redox‐active iodine and iodide species from aqueous electrolyte onto nanoporous carbon based current collector. By tuning iodide redox process and suppressing competitive hydrogen evolution reaction, the assembled CEMA Zn–I batteries demonstrate a remarkable capacity retention of 76.9% over 1000 cycles at 0.5 mA cm −2 . Moreover, they exhibit a notable rate capability, with a capacity retention of 74.6% when the current density is increased from 0.5 to 5.0 mA cm −2 . This study demonstrates the feasibility of using the oxidation effect to repel redox‐active species from the electrolyte to the cathode, paving a new avenue for high‐performance aqueous Zn–I batteries.",
    "stance":0.4
  },
  {
    "title":"Sustainable and Rapid Water Purification at the Confined Hydrogel Interface",
    "abstract":"Abstract Emerging organic contaminants in water matrices have challenged ecosystems and human health safety. Persulfate‐based advanced oxidation processes (PS‐AOPs) have attracted much attention as they address potential water purification challenges. However, overcoming the mass transfer constraint and the catalyst's inherent site agglomeration in the heterogeneous system remains urgent. Herein, the abundant metal‐anchored loading (≈6–8 g m −2 ) of alginate hydrogel membranes coupled with cross‐flow mode as an efficient strategy for water purification applications is proposed. The organic flux of the confined hydrogel interfaces sharply enlarges with the reduction of the thickness of the boundary layer via the pressure field. The normalized property of the system displays a remarkable organic (sulfonamides) elimination rate of 4.87 × 10 4 mg min −1 mol −1 . Furthermore, due to the fast reaction time (&lt;1 min), cross‐flow mode only reaches a meager energy cost (≈2.21 Wh m −3 ) under the pressure drive field. It is anticipated that this finding provides insight into the novel design with ultrafast organic removal performance and low techno‐economic cost (i.e., energy operation cost, material, and reagent cost) for the field of water purification under various PS‐AOPs challenging scenarios.",
    "stance":0.4
  },
  {
    "title":"Photothermal CO<sub>2<\/sub> Catalysis toward the Synthesis of Solar Fuel: From Material and Reactor Engineering to Techno‐Economic Analysis",
    "abstract":"Abstract Carbon dioxide (CO 2 ), a member of greenhouse gases, contributes significantly to maintaining a tolerable environment for all living species. However, with the development of modern society and the utilization of fossil fuels, the concentration of atmospheric CO 2 has increased to 400 ppm, resulting in a serious greenhouse effect. Thus, converting CO 2 into valuable chemicals is highly desired, especially with renewable solar energy, which shows great potential with the manner of photothermal catalysis. In this review, recent advancements in photothermal CO 2 conversion are discussed, including the design of catalysts, analysis of mechanisms, engineering of reactors, and the corresponding techno‐economic analysis. A guideline for future investigation and the anthropogenic carbon cycle are provided.",
    "stance":0.8
  },
  {
    "title":"Boosting Formate Electrooxidation by Heterostructured PtPd Alloy and Oxides Nanowires",
    "abstract":"Abstract Direct formate fuel cells (DFFCs) receive increasing attention as promising technologies for the future energy mix and environmental sustainability, as formate can be made from carbon dioxide utilization and is carbon neutral. Herein, heterostructured platinum‐palladium alloy and oxides nanowires (PtPd‐ox NWs) with abundant defect sites are synthesized through a facile self‐template method and demonstrated high activity toward formate electrooxidation reaction (FOR). The electronic tuning arising from the heterojunction between alloy and oxides influence the work function of PtPd‐ox NWs. The sample with optimal work function reveals the favorable adsorption behavior for intermediates and strong interaction in the d − p orbital hybridization between Pt site and oxygen in formate, favoring the FOR direct pathway with a low energy barrier. Besides the thermodynamic regulation, the heterostructure can also provide sufficient hydroxyl species to facilitate the formation of carbon dioxide due to the ability of combining absorbed hydrogen and carbon monoxide at adjacent active sites, which contributes to the improvement of FOR kinetics on PtPd‐ox NWs. Thus, heterostructured PtPd‐ox NWs achieve dual regulation of FOR thermodynamics and kinetics, exhibiting remarkable performance and demonstrating potential in practical systems.",
    "stance":0.3
  },
  {
    "title":"Recent Advances on Carbon‐Based Metal‐Free Electrocatalysts for Energy and Chemical Conversions",
    "abstract":"Over the last decade, carbon-based metal-free electrocatalysts (C-MFECs) have become important in electrocatalysis. This field is started thanks to the initial discovery that nitrogen atom doped carbon can function as a metal-free electrode in alkaline fuel cells. A wide variety of metal-free carbon nanomaterials, including 0D carbon dots, 1D carbon nanotubes, 2D graphene, and 3D porous carbons, has demonstrated high electrocatalytic performance across a variety of applications. These include clean energy generation and storage, green chemistry, and environmental remediation. The wide applicability of C-MFECs is facilitated by effective synthetic approaches, e.g., heteroatom doping, and physical\/chemical modification. These methods enable the creation of catalysts with electrocatalytic properties useful for sustainable energy transformation and storage (e.g., fuel cells, Zn-air batteries, Li-O",
    "stance":0.6
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
