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
OUTPUT_FILE = DATA_DIR / "NLP-Predictions_mistral_few_shot_50.json"

SLEEP_BETWEEN_CALLS = 1.0
REQUEST_TIMEOUT     = 500
TEMPERATURE         = 0.0
NUM_CTX             = 4096
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
  },
  {
    "title":"Tandem Upgrading of Bio‐Furans to Benzene, Toluene, and <i>p<\/i>‐xylene by Pt<sub>1<\/sub>Sn<sub>1<\/sub> Intermetallic Coupling Ordered Mesoporous SnO<sub>2<\/sub> Catalyst",
    "abstract":"Benzene, toluene, and p-xylene (BTpX) are among the most important commodity chemicals, but their productions still heavily rely on fossil resources and thus pose serious environmental burdens and energy crisis. Herein, the tandem upgrading of bio-furans is reported to high-yield BTpX by rationally constructing a versatile Pt",
    "stance":0.7
  },
  {
    "title":"ALD‐Coated Mesoporous Iridium‐Titanium Mixed Oxides: Maximizing Iridium Utilization for an Outstanding OER Performance",
    "abstract":"Abstract With the increasing production of renewable energy and concomitant depletion of fossil resources, the demand for efficient water splitting electrocatalysts continues to grow. Iridium (Ir) and iridium oxides (IrO x ) are currently the most promising candidates for an efficient oxygen evolution reaction (OER) in acidic medium, which remains the bottleneck in water electrolysis. Yet, the extremely high costs for Ir hamper a widespread production of hydrogen (H 2 ) on an industrial scale. Herein, the authors report a concept for the synthesis of electrode coatings with template‐controlled mesoporosity surface‐modified with highly active Ir species. The improved utilization of noble metal species relies on the synthesis of soft‐templated metal oxide supports and a subsequent shape‐conformal deposition of Ir species via atomic layer deposition (ALD) at two different reaction temperatures. The study reveals that a minimum Ir content in the mesoporous titania‐based support is mandatory to provide a sufficient electrical bulk conductivity. After ALD, a significantly enhanced OER activity results in dependency of the ALD cycle number and temperature. The most active developed electrocatalyst film achieves an outstanding mass‐specific activity of 2622 mA mg Ir –1 at 1.60 V RHE in a rotating‐disc electrode (RDE) setup at 25 °C using 0.5 m H 2 SO 4 as a supporting electrolyte.",
    "stance":0.3
  },
  {
    "title":"Solar‐Triggered Engineered 2D‐Materials for Environmental Remediation: Status and Future Insights",
    "abstract":"Abstract Modern‐day society requires advanced technologies based on renewable and sustainable energy resources to face the challenges regarding environmental remediation. Solar‐inspired photocatalytic applications for water purification, hydrogen and oxygen evolution, carbon dioxide reduction, nitrogen fixation, and removal of bacterial species seem to be unique solutions based on green and efficient technologies. Considering the unique electronic features and larger surface area, 2D photocatalysts have been broadly explored for the above‐mentioned applications in the past few years. However, their photocatalytic potential has not been optimized yet to the adequate level of practical and commercial applications. Among many strategies available, surface and interface engineering and the hybridization of different materials have revealed pronounced potential to boost the photocatalytic potential of 2D materials. This feature review recapitulates recent advancements in engineered materials that are 2D for various photocatalysis applications for environmental remediation. Various surface and interface engineering technologies are briefly discussed, like anion–cation vacancies, pits, distortions, associated vacancies, etc., along with rules and parameters. In addition, several hybridization approaches, like 0D\/2D, 1D\/2D, 2D\/2D, and 3D\/2D hybridization, etc., are also deeply investigated. Lastly, the application of these engineered 2D materials for various photocatalytic applications, challenges, and future perspectives is extensively explored.",
    "stance":0.7
  },
  {
    "title":"From Microbial Fuel Cells to Biobatteries: Moving toward On‐Demand Micropower Generation for Small‐Scale Single‐Use Applications",
    "abstract":"Abstract Microbial fuel cells (MFCs) that generate electricity generation from a broad diversity of biomass and organic substrates through microbial metabolism have attracted considerable research interest as an alternative clean energy technology and energy‐efficient wastewater treatment method. Despite encouraging successes and auspicious pilot‐scale experiments of the MFCs, increasing doubts about their viability for practical large‐scale applications are being raised. Low performance, expensive core parts and materials, energy‐intensive operation, and scaling bottlenecks question a sustainable development. Instead, special MFCs for low‐power battery‐reliant devices might be more applicable and potentially realizable. Such bacteria‐powered biobatteries would enable i) a truly stand‐alone device platform suitable for use in resource‐limited and remote regions, ii) simple, on‐demand power generation within a programmed period of time, and iii) a tracelessly biodegradable battery due to the use of the bacteria used for power generation. The biobattery would be an excellent power solution for small‐scale, on‐demand, single‐use, and disposable electronics. Recent progress of small‐scale MFC‐based biobatteries is critically reviewed with specific attention toward various device platforms. Furthermore, comments and outlook related to the potential directions and challenges of the biobatteries are discussed to offer inspiration to the community and induce fruitful future research.",
    "stance":0.3
  },
  {
    "title":"Research and Development on Coatings and Paints for Geothermal Environments: A Review",
    "abstract":"Abstract Geothermal power plants are complex systems, where the interplay between different metallic components transforms the enthalpy of hot brine in the form of electricity or usable heated water. The naturally occurring variety in brine chemistry, linked to the presence of specific key species, and its thermo‐physical properties, leads to the development of different power plant configurations. Key species and power plant configuration in turn determine the extent of damage experienced by each component in the power plant: erosion, corrosion, and\/or scaling, often acting combined. Paints and coatings, compared to changing the component alloy, have represented a preferred solution to mitigate these issues due to advantages in terms of costs and repairability. This is reflected in the large number of publications on research and development in this area within the past ≈50 years, with even an increasing trend in the past 10–20 years, indicating the strong interest to develop this clean and sustainable energy source. Therefore, in this work, the first of its kind after 1980, an in‐depth review of all published work on research performed on paints and coatings for geothermal applications, subdivided by the material system, is provided.",
    "stance":0.4
  },
  {
    "title":"Potential of Aluminum as a Metal Fuel for Supporting EU Long‐Term Energy Storage Needs",
    "abstract":"Abstract The EU's energy transition necessitates availability of green energy carriers with high volumetric energy densities for long‐term energy storage (ES) needs. A fully decarbonized scenario considering renewable energy availability is analyzed underpinning the need for such carriers. Considering the shortcomings of Power‐to‐X technologies in terms of efficiency and low volumetric density, Aluminum (Al) is identified as a potential alternative showing significantly high volumetric energy densities (23.5 kWh L −1 ). In this paper, an Al‐based long‐term ES concept is investigated, taking advantage of the inherent recycling of the active species (i.e., Al 2 O 3 to Al) coupling decarbonized Hall–Héroult process with an Al‐steam oxidation for simultaneous hydrogen (H 2 ) and heat generation. This work demonstrates an innovative lab‐scale fine Al powder‐steam oxidation process at ≈900 °C without use of catalysts or additives, exploiting alumina as inert material. Conducted SEM‐EDX analysis on oxidized Al provides supporting evidence in favor of employed oxidation pathway, hindering tendency of aluminum oxide (Al 2 O 3 ) clumping and enabling direct use of oxides in the smelting process for fully recyclability. Moreover, outcomes of XRD analyses are presented to validate the measured total H 2 yields.",
    "stance":0.6
  },
  {
    "title":"The Structure–Activity Relationship in Membranes for Vanadium Redox Flow Batteries",
    "abstract":"Abstract Vanadium redox flow batteries (VRFBs) are regarded as one of the most promising electrochemical technologies for grid‐connected renewable energy storage systems. The performance of VRFBs, however, strongly depends on the membrane, one of the key components of VRFBs with critical dual functions of promotion of diffusion of active species (H + , H 3 O + , SO 4 2− , or SO 4 H − ) and inhibition of crossover of vanadium ions. This is intrinsically related to the microstructure of membranes. For example, large and connected ionic clusters or pores in membranes are favorable for the ion transfer, but detrimental to the ion selectivity. While small and isolated hydrophilic ion clusters or pores suppress the water uptake and ion transfer, the decreased swelling ratio would enhance chemical stability of membrane. Thus, comprehensive strategies are required to realize the optimal balance between the ion selectivity, proton conductivity, and chemical stability. This review focuses on the effects of microstructure of membranes on the ion transfer and the chemical stability, including introduction of the rigid groups, electron‐withdrawing groups, and hydrophobic backbones, are reviewed. The prospect of the development of membranes with high ion selectivity and high‐performance VRFBs is discussed.",
    "stance":0.5
  },
  {
    "title":"Renewable Energy from Wildflowers—Perennial Wild Plant Mixtures as a Social‐Ecologically Sustainable Biomass Supply System",
    "abstract":"Abstract A growing bioeconomy requires increasing amounts of biomass from residues, wastes, and industrial crops for bio‐based products and bioenergy. There is much discussion about how industrial crop cultivation could promote social–ecological outcomes such as environmental protection, biodiversity conservation, climate change adaptation, food security, greenhouse gas mitigation, and landscape appearance. In Germany, maize ( Zea mays L.) is the main biogas substrate source, despite being associated with problems such as erosion, biodiversity losses, an increase in wild boar populations and lowered landscape diversity. The cultivation of perennial wild plant mixtures (WPM) addresses many of these problems. Despite being less developed than maize, WPM cultivation has received notable attention among scientists in Germany over the past decade. This is mainly because WPMs clearly outperform maize in social–ecological measures, despite their methane yield performance. This review summarizes and discusses the results of 12 years of research and practice with WPMs as a social‐ecologically more benign bioenergy cropping system.",
    "stance":0.9
  },
  {
    "title":"Tunable Tannic Acid–Zein Adhesives for Bonding Different Substrates",
    "abstract":"Abstract Plant‐based, nontoxic and strong adhesives that can work on different substrates allow for better product recycling. Replacing synthetic adhesives with plant based ones that perform equally well may lead to a more sustainable ecosystem. Adhesives based on zein protein, derived from corn, can be made as strong as Super Glue. Unlike petroleum‐based adhesives, this plant protein is removable and degradable. Here, adhesives from the components zein and tannic acid function well on different substrates such as metals, plastics, and wood. This work presents the properties of selected adhesive formulations including bond strengths when the substrates are changed. To achieve near Super Glue strength, each substrate requires a different adhesive formulation. Temperature‐dependent curing, (potential of hydrogen) changes, and additional variables influence adhesion. Tack testing at room temperature is measured to provide comparisons between initial bonding and that achieved after curing. Adhesion to plastic substrates is less than wood or metal, but still plenty strong to measure. Aging of adhesive solutions as well as water resistance after curing is investigated. Infrared spectroscopy data are correlated with changes in color, age, and pH of adhesives. These new adhesives may lead to a sustainable and cleaner environment.",
    "stance":0.7
  },
  {
    "title":"Exploring the Potential of Perennial Nectar‐Producing Wild Plants for Pellet Combustion",
    "abstract":"Abstract Perennial nectar‐producing wild plant species (WPS) cultivation for biogas production helps improve ecosystem services such as habitat functioning, erosion mitigation, groundwater protection, and carbon sequestration. These ecosystem services could be improved when WPS are harvested in late winter to produce pellets and briquettes as solid energy carriers for heat production. This study aims for gaining first insights into the use of WPS biomass as resource for pellet and briquette combustion with focus on two perennial WPS common tansy ( Tanacetum vulgare L.) and mugwort ( Artemisia vulgaris L.), and two biennial WPS yellow melilot ( Melilotus officinalis L.) and wild teasel ( Dipsacus fullonum L.). All WPS are found economically viable for pellet combustion. The main drivers are i) low cultivation costs, ii) subsidies, and iii) low pellet production costs due to low moisture contents. However, high ash contents in WPS biomass justify the need of i) blending with woody‐biomass or ii) supplementing with additives to attain international standards for household stoves. This approach appears technically feasible providing a research field with significant potential impacts. As 70% of the pellet market is demanded as household level, public concern about the legal framework of alternative plant biomass pellets must be overcome to develop this market.",
    "stance":0.6
  },
  {
    "title":"Bifunctional PtCu Nanooctahedrons for the Electrochemical Conversion of Nitrite and Sulfion Into Value‐Added Products",
    "abstract":"Abstract The electrochemical reduction of nitrite (NO 2 − ) contaminants to ammonia (NH 3 ) is a sustainable and energy‐saving strategy for NH 3 synthesis. However, this multi‐electron reduction process requires an efficient electrocatalyst to overcome the kinetic barrier. Herein, the Pt 2 Cu 1 nanooctahedrons are synthesized through a liquid‐phase chemical reduction process. The synergistic effect of bimetallic Pt and Cu sites in the Pt 2 Cu 1 nanooctahedrons is indispensable for accelerated NO 2 − hydrogenation, originating from the strong hydrogen‐atoms adsorption capacity at Pt site and the strong NO 2 − adsorption capacity at Cu site. Specifically, the introduction of Pt sites can accelerate the accumulation of hydrogenated species on the catalyst surface, which promotes the formation of NH 3 . In 0.5 m Na 2 SO 4 solution, the Pt 2 Cu 1 nanooctahedrons can reduce NO 2 − to NH 3 at a yield of 4.22 mg h −1 mg cat −1 and a Faraday efficiency of 95.5% at a potential of −0.14 V versus RHE. Meanwhile, the Pt 2 Cu 1 nanooctahedrons also exhibit excellent activity for the sulfion oxidation reaction (SEOR). Using Pt 2 Cu 1 nanooctahedrons as bifunctional electrocatalyst, a coupled electrolysis system combining the nitrite electrochemical reduction reaction (NO 2 − ERR) with the SEOR requires only 0.3 V total voltage, enabling energy‐saving electrochemical NH 3 production and collective value‐added recovery of nitrite and sulfion waste.",
    "stance":0.5
  },
  {
    "title":"Solubilization of polystyrene into monoterpenes",
    "abstract":"Abstract The ability of certain monoterpenes from the essential oils of Abies sachalinensis and Eucalyptus species to dissolve polystyrene (PS) was studied. These two essential oils themselves were also examined. The aim of this study is to recycle PS without a melting process and without using petroleum‐derived solvents. The relationship between chemical structure of the terpenes and their dissolving power for PS is investigated through the solubility parameter and apparent activation energy for dissolution. α‐Terpinene and its positional isomers on a CC bond have high solvent powers for PS, whereas bicyclic terpenes are inferior in this regard even though they have similar solubility parameters to that of α‐terpinene. It is suggested that the bulky and\/or hydrophilic structures in the solvent molecule prevent the dissolution of PS. Simple steam distillation of solutions of PS in the terpenes gave recovery of more than 97% of the PS and terpenes. The present solvent systems cause little degradation to PS and are promising for the recycling of PS using sustainable solvents. © 2008 Wiley Periodicals, Inc. Adv Polym Techn 27:35–39, 2008; Published online in Wiley InterScience ( www.interscience.wiley.com ). DOI 10.1002\/adv.20115",
    "stance":0.6
  },
  {
    "title":"One‐Dimensional Earth‐Abundant Nanomaterials for Water‐Splitting Electrocatalysts",
    "abstract":"Hydrogen fuel acquisition based on electrochemical or photoelectrochemical water splitting represents one of the most promising means for the fast increase of global energy need, capable of offering a clean and sustainable energy resource with zero carbon footprints in the environment. The key to the success of this goal is the realization of robust earth‐abundant materials and cost‐effective reaction processes that can catalyze both hydrogen evolution reaction (HER) and oxygen evolution reaction (OER), with high efficiency and stability. In the past decade, one‐dimensional (1D) nanomaterials and nanostructures have been substantially investigated for their potential in serving as these electrocatalysts for reducing overpotentials and increasing catalytic activity, due to their high electrochemically active surface area, fast charge transport, efficient mass transport of reactant species, and effective release of gas produced. In this review, we summarize the recent progress in developing new 1D nanomaterials as catalysts for HER, OER, as well as bifunctional electrocatalysts for both half reactions. Different categories of earth‐abundant materials including metal‐based and metal‐free catalysts are introduced, with their representative results presented. The challenges and perspectives in this field are also discussed.",
    "stance":0.8
  },
  {
    "title":"Green Synthesis of Hierarchical Metal–Organic Framework\/Wood Functional Composites with Superior Mechanical Properties",
    "abstract":"The applicability of advanced composite materials with hierarchical structure that conjugate metal-organic frameworks (MOFs) with macroporous materials is commonly limited by their inferior mechanical properties. Here, a universal green synthesis method for the in situ growth of MOF nanocrystals within wood substrates is introduced. Nucleation sites for different types of MOFs are readily created by a sodium hydroxide treatment, which is demonstrated to be broadly applicable to different wood species. The resulting MOF\/wood composite exhibits hierarchical porosity with 130 times larger specific surface area compared to native wood. Assessment of the CO2 adsorption capacity demonstrates the efficient utilization of the MOF loading along with similar adsorption ability to that of pure MOF. Compression and tensile tests reveal superior mechanical properties, which surpass those obtained for polymer substrates. The functionalization strategy offers a stable, sustainable, and scalable platform for the fabrication of multifunctional MOF\/wood-derived composites with potential applications in environmental- and energy-related fields.",
    "stance":0.7
  },
  {
    "title":"Synergistic Effects in N,O‐Comodified Carbon Nanotubes Boost Highly Selective Electrochemical Oxygen Reduction to H<sub>2<\/sub>O<sub>2<\/sub>",
    "abstract":"Abstract Electrochemical 2‐electron oxygen reduction reaction (ORR) is a promising route for renewable and on‐site H 2 O 2 production. Oxygen‐rich carbon nanotubes have been demonstrated their high selectivity (≈80%), yet tailoring the composition and structure of carbon nanotubes to further enhance the selectivity and widen working voltage range remains a challenge. Herein, combining formamide condensation coating and mild temperature calcination, a nitrogen and oxygen comodified carbon nanotubes (N,O‐CNTs) electrocatalyst is synthesized, which shows excellent selective (&gt;95%) H 2 O 2 selectivity in a wide voltage range (from 0 to 0.65 V versus reversible hydrogen electrode). It is significantly superior to the corresponding selectivity values of CNTs (≈50% in 0–0.65 V vs RHE) and O‐CNTs (≈80% in 0.3–0.65 V vs RHE). Density functional theory calculations revealed that the C neighbouring to N is the active site. Introducing O‐related species can strengthen the adsorption of intermediates *OOH, while N‐doping can weaken the adsorption of in situ generated *O and optimize the *OOH adsorption energy, thus improving the 2‐electron pathway. With optimized N,O‐CNTs catalysts, a Janus electrode is designed by adjusting the asymmetric wettability to achieve H 2 O 2 productivity of 264.8 mol kg cat –1 h –1 .",
    "stance":0.6
  },
  {
    "title":"Electronic Structure Engineering of Highly‐Scalable Earth‐Abundant Multi‐Synergized Electrocatalyst for Exceptional Overall Water Splitting in Neutral Medium",
    "abstract":"Efficient neutral water splitting may represent in future a sustainable solution to unconstrained energy requirements, but yet necessitates the development of innovative avenues for achieving the currently unmet required performances. Herein, a novel paradigm based on the combination of electronic structure engineering and surface morphology tuning of earth-abundant 3D-hierarchical binder-free electrocatalysts is demonstrated, via a scalable single-step thermal transformation of nickel substrates under sulfur environment. A temporal-evolution of the resulting 3D-nanostructured substrates is performed for the intentional enhancement of non-abundant highly-catalytic Ni",
    "stance":0.7
  },
  {
    "title":"Lactate Efflux Inhibition by Syrosingopine\/LOD Co‐Loaded Nanozyme for Synergetic Self‐Replenishing Catalytic Cancer Therapy and Immune Microenvironment Remodeling",
    "abstract":"Abstract An effective systemic mechanism regulates tumor development and progression; thus, a rational design in a one‐stone‐two‐birds strategy is meant for cancer treatment. Herein, a hollow Fe 3 O 4 catalytic nanozyme carrier co‐loading lactate oxidase (LOD) and a clinically‐used hypotensor syrosingopine (Syr) are developed and delivered for synergetic cancer treatment by augmented self‐replenishing nanocatalytic reaction, integrated starvation therapy, and reactivating anti‐tumor immune microenvironment. The synergetic bio‐effects of this nanoplatform stemmed from the effective inhibition of lactate efflux through blocking the monocarboxylate transporters MCT1\/MCT4 functions by the loaded Syr as a trigger. Sustainable production of hydrogen peroxide by catalyzation of the increasingly residual intracellular lactic acid by the co‐delivered LOD and intracellular acidification enabled the augmented self‐replenishing nanocatalytic reaction. Large amounts of produced reactive oxygen species (ROS) damaged mitochondria to inhibit oxidative phosphorylation as the substituted energy supply upon the hampered glycolysis pathway of tumor cells. Meanwhile, remodeling anti‐tumor immune microenvironment is implemented by pH gradient reversal, promoting the release of proinflammatory cytokines, restored effector T and NK cells, increased M1‐polarize tumor‐associated macrophages, and restriction of regulatory T cells. Thus, the biocompatible nanozyme platform achieved the synergy of chemodynamic\/immuno\/starvation therapies. This proof‐of‐concept study represents a promising candidate nanoplatform for synergetic cancer treatment.",
    "stance":0.0
  },
  {
    "title":"Coordinative Stabilization of Single Bismuth Sites in a Carbon–Nitrogen Matrix to Generate Atom‐Efficient Catalysts for Electrochemical Nitrate Reduction to Ammonia",
    "abstract":"Electrochemical nitrate reduction to ammonia powered by renewable electricity is not only a promising alternative to the established energy-intense and non-ecofriendly Haber-Bosch reaction for ammonia generation but also a future contributor to the ever-more important denitrification schemes. Nevertheless, this reaction is still impeded by the lack of understanding for the underlying reaction mechanism on the molecular scale which is necessary for the rational design of active, selective, and stable electrocatalysts. Herein, a novel single-site bismuth catalyst (Bi-N-C) for nitrate electroreduction is reported to produce ammonia with maximum Faradaic efficiency of 88.7% and at a high rate of 1.38 mg h",
    "stance":0.7
  },
  {
    "title":"Fabrication of Interface Engineered S‐Scheme Heterojunction Nanocatalyst for Ultrasound‐Triggered Sustainable Cancer Therapy",
    "abstract":"Abstract In order to establish a set of perfect heterojunction designs and characterization schemes, step‐scheme (S‐scheme) BiOBr@Bi 2 S 3 nanoheterojunctions that enable the charge separation and expand the scope of catalytic reactions, aiming to promote the development and improvement of heterojunction engineering is developed. In this kind of heterojunction system, the Fermi levels mediate the formation of the internal electric field at the interface and guide the recombination of the weak redox carriers, while the strong redox carriers are retained. Thus, these high‐energy electrons and holes are able to catalyze a variety of substrates in the tumor microenvironment, such as the reduction of oxygen and carbon dioxide to superoxide radicals and carbon monoxide (CO), and the oxidation of H 2 O to hydroxyl radicals, thus achieving sonodynamic therapy and CO combined therapy. Mechanistically, the generated reactive oxygen species and CO damage DNA and inhibit cancer cell energy levels, respectively, to synergistically induce tumor cell apoptosis. This study provides new insights into the realization of high efficiency and low toxicity in catalytic therapy from a unique perspective of materials design. It is anticipated that this catalytic therapeutic method will garner significant interest in the sonocatalytic nanomedicine field.",
    "stance":0.0
  },
  {
    "title":"Materials Containing Single‐, Di‐, Tri‐, and Multi‐Metal Atoms Bonded to C, N, S, P, B, and O Species as Advanced Catalysts for Energy, Sensor, and Biomedical Applications",
    "abstract":"Abstract Modifying the coordination or local environments of single‐, di‐, tri‐, and multi‐metal atom (SMA\/DMA\/TMA\/MMA)‐based materials is one of the best strategies for increasing the catalytic activities, selectivity, and long‐term durability of these materials. Advanced sheet materials supported by metal atom‐based materials have become a critical topic in the fields of renewable energy conversion systems, storage devices, sensors, and biomedicine owing to the maximum atom utilization efficiency, precisely located metal centers, specific electron configurations, unique reactivity, and precise chemical tunability. Several sheet materials offer excellent support for metal atom‐based materials and are attractive for applications in energy, sensors, and medical research, such as in oxygen reduction, oxygen production, hydrogen generation, fuel production, selective chemical detection, and enzymatic reactions. The strong metal–metal and metal–carbon with metal–heteroatom (i.e., N, S, P, B, and O) bonds stabilize and optimize the electronic structures of the metal atoms due to strong interfacial interactions, yielding excellent catalytic activities. These materials provide excellent models for understanding the fundamental problems with multistep chemical reactions. This review summarizes the substrate structure‐activity relationship of metal atom‐based materials with different active sites based on experimental and theoretical data. Additionally, the new synthesis procedures, physicochemical characterizations, and energy and biomedical applications are discussed. Finally, the remaining challenges in developing efficient SMA\/DMA\/TMA\/MMA‐based materials are presented.",
    "stance":0.4
  },
  {
    "title":"Fully Floatable Mortise‐and‐Tenon Architecture for Synergistically Photo\/Sono‐Driven Evaporation Desalination and Plastic‐Enabled Value‐Added Co‐Conversion of H<sub>2<\/sub>O and CO<sub>2<\/sub>",
    "abstract":"Abstract Establishing an advanced ecosystem incorporating freshwater harvesting, plastic utilization, and clean fuel acquisition is profoundly significant. However, low‐efficiency evaporation, single energy utilization, and catalyst leakage severely hinder sustainable development. Herein, a nanofiber‐based mortise‐and‐tenon structural Janus aerogel (MTSJA) is strategically designed in the first attempt and supports Z‐scheme catalysts. By harnessing of the upper hydrophilic layer with hydrophilic channels embedding into the hydrophobic bottom layer to achieve tailoring bottom wettability states. MTSJA is capable of a fully‐floating function for lower heat loss, water supply, and high‐efficiency solar‐to‐vapor conversion. Benefiting from the ultrasonic cavitation effect and high sensitivity of materials to mechanical forces, this is also the first demonstration of synergistic solar and ultrasound fields to power simultaneous evaporation desalination and waste plastics as reusable substrates generating fuel energy. The system enables persistent desalination with an exceptional evaporation rate of 3.1 kg m −2 h −1 and 82.3% efficiency (21 wt.% NaCl solution and 1 sun), and realizes H 2 , CO, and CH 4 yields with 16.1, 9.5, and 3 µmol h −1 g −1 , respectively. This strategy holds great potential for desalination and plastics value‐added transformation toward clean energy and carbon neutrality.",
    "stance":0.9
  },
  {
    "title":"Visible‐Light‐Induced Hydrogen GeneratioG from Mixtures of Hydrogen Boride Nanosheets and Phenanthroline Molecules",
    "abstract":"Abstract Hydrogen boride (HB) nanosheets are recognized as a safe and lightweight hydrogen carrier, yet their hydrogen (H 2 ) generation technique has been limited. In the present study, nitrogen‐containing organic heterocycles are mixed with HB nanosheets in acetonitrile solution for visible‐light‐driven H 2 generation. After exploring various nitrogen‐containing heterocycles, the mixture of 1,10‐phenanthroline molecules (Phens) and HB nanosheets exhibited significant H 2 generation even under visible light irradiation. The quantum efficiency for H 2 generation of the mixture of HB nanosheets and Phens is 0.6%. Based on spectroscopic and electrochemical analyses and density functional theory (DFT) calculations, it is determined that radical species generated from Phens with electrons and protons donated by HB nanosheets are responsive to visible light for H 2 generation. The HB nanosheets\/Phens mixture presented in this study can generate H 2 using renewable energy sources such as sunlight without the need for complex electrochemical systems or heating mechanisms and is expected to serve as a lightweight hydrogen storage\/release system.",
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
