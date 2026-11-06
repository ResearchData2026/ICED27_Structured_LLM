import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

GOOGLE_API_KEY = "" #paid to use, keep an eye on the cost

GENERATION_MODEL = "gemini-3.8-flash"
THINKING_LEVEL = "medium"          # "low", "medium" or "high" (Gemini 3.x)
MAX_OUTPUT_TOKENS = 8192
# Temperature is left at the model default, as Google advises for Gemini 3.x.

SCRIPT_DIR = Path(__file__).resolve().parent
TRIZ_MATRIX_FILE = "ICED27/Structured LLM/TRIZ_Contradiction_Matrix.xls"   # .xls, .xlsx or .csv; a full path also works
# A bare file name is looked for next to this script, then in the folder above it.

PILOT = False                       # True = 2 briefs (1 adaptive, 1 original) x 2 runs, in results/pilot
RUN_NAME = "run_01"                # folder for the full run; same name = resume
MOCK = False                       # True = fake responses, no API calls, no cost

N_CONCEPTS = 4
N_RUNS = 5
BASELINE = "generic"
CONDITIONS = ["generic", "scamper", "triz", "analogy", "morphological"]
RANDOM_SEED = 2027                 # run order and source-domain draws

MAX_WORKERS = 4                    # parallel sessions; lower it if you hit rate limits
MAX_RETRIES = 6

RESULTS_DIR = SCRIPT_DIR / "results"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DESIGN BRIEFS                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# DESIGN BRIEF FORMAT (keep to this when editing)
#   "id"      short unique label, no spaces (used in file names)
#   "type"    "adaptive"  redesign of an existing product type (Pahl et al., 2007)
#             "original"  a new product, defined by what it must do, no product named
#   "title"   short name for reports and the paper; the model never sees it
#   "product" adaptive briefs only: a generic product type, no brand names
#   "goal"    one sentence: what to achieve, plus one "without ...", "while ..." or
#             "using ..." condition that creates the trade-off. Original briefs start
#             "Design a product that ..." (or similar) so the answer stays a product.
#             Do not suggest how to achieve the goal.
#   "source"  optional: the product in the 40-product set the brief comes from
#             (for the record only; the model never sees it)
# The model sees only:  DESIGN BRIEF (kind) / Product: ... / Goal: ...
#
# The six products are taken from the 40-product set of the author's earlier study:
# three as adaptive (redesign) briefs and three re-framed as original-design briefs
# from the product's function. Chosen to spread across product categories and across
# the life-cycle stage where most of the environmental impact sits.
BRIEFS = [
    {
        "id": "A1_kettle",
        "type": "adaptive",
        "title": "Electric kettle",
        "product": "Electric kettle",
        "goal": "Redesign it to use much less energy per use, without making boiling water slower or less convenient.",
        "source": "Electric kettle",
    },
    {
        "id": "A2_screwdriver",
        "type": "adaptive",
        "title": "Cordless screwdriver",
        "product": "Cordless screwdriver",
        "goal": "Redesign it so that it lasts much longer and uses fewer resources over its life, without "
                "reducing its power or ease of use.",
        "source": "Cordless screwdriver",
    },
    {
        "id": "A3_toothbrush",
        "type": "adaptive",
        "title": "Electric toothbrush",
        "product": "Electric toothbrush",
        "goal": "Redesign it to create far less waste over its life, while cleaning teeth as effectively as today.",
        "source": "Electric toothbrush",
    },
    {
        "id": "O1_laptopwork",
        "type": "original",
        "title": "Portable laptop workspace",
        "goal": "Design a product that lets people work comfortably on a laptop wherever they are, using as "
                "little material as possible.",
        "source": "Portable laptop stand",
    },
    {
        "id": "O2_citytransport",
        "type": "original",
        "title": "Personal urban transport",
        "goal": "Design a personal transport product for short city journeys of a few kilometres, using far "
                "fewer resources than a car.",
        "source": "Folding bike",
    },
    {
        "id": "O3_irrigation",
        "type": "original",
        "title": "Off-grid irrigation pump",
        "goal": "Design a product that lets smallholder farmers without grid electricity lift water for "
                "irrigation, without using fossil fuel.",
        "source": "Solar-powered water pump",
    },
]


def check_briefs(briefs: list) -> list:
    """Return a list of problems with the BRIEFS format (empty if all is well)."""
    problems, seen = [], set()
    for n, b in enumerate(briefs, 1):
        label = b.get("id", f"brief {n}")
        for key in ("id", "type", "title", "goal"):
            if not str(b.get(key, "")).strip():
                problems.append(f"{label}: missing '{key}'")
        if b.get("type") not in ("adaptive", "original"):
            problems.append(f"{label}: 'type' must be \"adaptive\" or \"original\"")
        if b.get("type") == "adaptive" and not str(b.get("product", "")).strip():
            problems.append(f"{label}: adaptive briefs need a 'product'")
        if b.get("type") == "original" and str(b.get("product", "")).strip():
            problems.append(f"{label}: original briefs should not name a 'product'")
        if re.search(r"\s", str(b.get("id", ""))):
            problems.append(f"{label}: 'id' must not contain spaces")
        if b.get("id") in seen:
            problems.append(f"{label}: duplicate 'id'")
        seen.add(b.get("id"))
        extra = set(b) - {"id", "type", "title", "product", "goal", "source"}
        if extra:
            problems.append(f"{label}: unknown field(s) {sorted(extra)}, the model would not see them")
    return problems

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  METHOD CONTENT                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SCAMPER_OPERATORS = [
    ("Substitute", "replace a material, component, process, energy source or user action"),
    ("Combine", "merge the product with another product, function or service"),
    ("Adapt", "borrow and adjust a solution that already works in another context"),
    ("Modify", "change size, shape, form, attributes or magnify or minimise a feature"),
    ("Put to other uses", "use the product, its parts or its waste for a different purpose"),
    ("Eliminate", "remove a part, material, step or function"),
    ("Reverse", "invert, rearrange or change the order, orientation or roles"),
]

TRIZ_PARAMETERS = [
    "Weight of moving object", "Weight of stationary object", "Length of moving object",
    "Length of stationary object", "Area of moving object", "Area of stationary object",
    "Volume of moving object", "Volume of stationary object", "Speed", "Force",
    "Stress or pressure", "Shape", "Stability of the object's composition", "Strength",
    "Duration of action of moving object", "Duration of action of stationary object",
    "Temperature", "Illumination intensity", "Use of energy by moving object",
    "Use of energy by stationary object", "Power", "Loss of energy", "Loss of substance",
    "Loss of information", "Loss of time", "Quantity of substance", "Reliability",
    "Measurement accuracy", "Manufacturing precision", "External harm affects the object",
    "Object-generated harmful factors", "Ease of manufacture", "Ease of operation",
    "Ease of repair", "Adaptability or versatility", "Device complexity",
    "Difficulty of detecting and measuring", "Extent of automation", "Productivity",
]

TRIZ_PRINCIPLES = [
    "Segmentation", "Taking out", "Local quality", "Asymmetry", "Merging",
    "Universality", "Nested doll", "Anti-weight", "Preliminary anti-action",
    "Preliminary action", "Beforehand cushioning", "Equipotentiality",
    "The other way round", "Spheroidality and curvature", "Dynamics",
    "Partial or excessive actions", "Another dimension", "Mechanical vibration",
    "Periodic action", "Continuity of useful action", "Skipping", "Blessing in disguise",
    "Feedback", "Intermediary", "Self-service", "Copying",
    "Cheap short-living objects", "Mechanics substitution", "Pneumatics and hydraulics",
    "Flexible shells and thin films", "Porous materials", "Colour changes",
    "Homogeneity", "Discarding and recovering", "Parameter changes",
    "Phase transitions", "Thermal expansion", "Strong oxidants", "Inert atmosphere",
    "Composite materials",
]

# Source-domain categories for design-by-analogy. N_CONCEPTS (4) are drawn at random
# for each session, so the model cannot fall back on its favourite sources.
SOURCE_DOMAINS = [
    "animals", "plants", "microorganisms and cells", "the human body",
    "natural phenomena (weather, geology, water)", "architecture and construction",
    "transport and vehicles", "textiles and clothing", "food and cooking",
    "sport and games", "medical devices", "aerospace and marine engineering",
    "musical instruments", "traditional crafts and tools",
]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PROMPTS                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = (
    "You are an experienced engineering designer working on the conceptual "
    "design stage of a product. You follow the design procedure you are given "
    "step by step. Every concept must be a physical product that an engineering "
    "team could develop further. Write in plain English. Return only valid JSON "
    "in the structure requested."
)

FINAL_FORMAT = f"""
Return ONLY a JSON object with this structure:
{{
  "concepts": [
    {{
      "id": 1,
      "title": "<short concept name, max 8 words>",
      "description": "<60 to 100 words describing what the concept is, how it works and how it meets the brief>",
      "method_trace": "<the method element that produced this concept>"
    }}
  ]
}}
The "concepts" list must contain exactly {N_CONCEPTS} concepts with ids 1 to {N_CONCEPTS}.
Each description must be between 60 and 100 words and must stand on its own,
without referring to the method or to other concepts.
"""


def brief_text(brief: dict) -> str:
    kind = "Redesign of an existing product" if brief["type"] == "adaptive" else "New product design"
    lines = [f"DESIGN BRIEF ({kind})"]
    if brief.get("product"):
        lines.append(f"Product: {brief['product']}")
    lines.append(f"Goal: {brief['goal']}")
    return "\n".join(lines) + "\n"


def as_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("No JSON object found in response")


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))


# Words that would tell the judge which method produced a concept. The judge
# sees only the title and description, so these must not appear there.
METHOD_TERMS = re.compile(
    r"\bscamper\b|\btriz\b|inventive principle|\bprinciple\s+\d+|morpholog|contradiction matrix"
    r"|source domain|by analogy|design-by-analogy|\banalog(?:y|ies)\b", re.I)
WORD_RANGE = (50, 120)   # the prompt asks for 60-100 words; this allows a small margin


def check_concepts(d: dict) -> None:
    """Same check for every condition, so all final outputs are comparable."""
    cs = d.get("concepts")
    if not isinstance(cs, list) or len(cs) != N_CONCEPTS:
        raise ValueError(f"expected {N_CONCEPTS} concepts, got {len(cs) if isinstance(cs, list) else 'none'}")
    for c in cs:
        if not c.get("title") or not c.get("description"):
            raise ValueError("a concept is missing its title or description")
        n = word_count(c["description"])
        if not WORD_RANGE[0] <= n <= WORD_RANGE[1]:
            raise ValueError(f"description of {n} words, outside {WORD_RANGE[0]}-{WORD_RANGE[1]}")
        leak = METHOD_TERMS.search(f"{c['title']} {c['description']}")
        if leak:
            raise ValueError(f"title or description names the method ('{leak.group(0)}')")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TRIZ CONTRADICTION MATRIX                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def triz_path() -> Path:
    p = Path(TRIZ_MATRIX_FILE)
    if p.is_absolute():
        return p
    for folder in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if (folder / p).exists():
            return folder / p
    return SCRIPT_DIR / p


def read_grids(path: Path) -> list:
    """Every sheet of an Excel file (.xls or .xlsx), or the single grid of a CSV file."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        sheets = pd.read_excel(path, header=None, dtype=str, sheet_name=None)
        return [df.fillna("").values.tolist() for df in sheets.values()]
    import csv
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return [list(csv.reader(f))]
        except UnicodeDecodeError:
            continue
    return []


def _num(cell):
    m = re.match(r"^\s*(\d{1,2})\b", str(cell))
    return int(m.group(1)) if m else None


def _pad(grid: list) -> list:
    width = max((len(r) for r in grid), default=0)
    return [[str(x) for x in r] + [""] * (width - len(r)) for r in grid]


def _find_run(seq: list, n: int):
    """Index where 1..n appear in order in seq, or None."""
    target = list(range(1, n + 1))
    for start in range(len(seq) - n + 1):
        if seq[start:start + n] == target:
            return start
    return None


def load_triz(path: Path) -> dict:
    """
    Load the contradiction matrix, and if present the 39 parameter names and
    the 40 principles with descriptions, from one file (any sheet).
    Returns {"matrix": {(improving, worsening): [principles]},
             "parameters": [39 names], "principles": {n: {"name", "description"}}}
    """
    if MOCK and not path.exists():
        rng = random.Random(0)
        return {"matrix": {(i, j): rng.sample(range(1, 41), rng.randint(2, 4))
                           for i in range(1, 40) for j in range(1, 40) if i != j},
                "parameters": TRIZ_PARAMETERS,
                "principles": {n + 1: {"name": nm, "description": ""} for n, nm in enumerate(TRIZ_PRINCIPLES)}}
    if not path.exists():
        raise FileNotFoundError(
            f"TRIZ matrix not found: {path}\n  Put your contradiction matrix next to this script "
            "and set TRIZ_MATRIX_FILE at the top of the file.")
    grids = [_pad(g) for g in read_grids(path)]

    out, errors = None, []
    for g in grids:
        try:
            out = parse_triz_matrix(g)
            break
        except ValueError as e:
            errors.append(str(e))
    if out is None:
        raise ValueError(errors[-1] if errors else "the file is empty")

    out["principles"] = None
    for g in grids:
        pr = parse_triz_principles(g)
        if pr:
            out["principles"] = pr
            break
    if out["principles"] is None:   # fall back to the built-in names
        out["principles"] = {n + 1: {"name": nm, "description": ""} for n, nm in enumerate(TRIZ_PRINCIPLES)}
    return out


def parse_triz_matrix(grid: list) -> dict:
    # worsening parameters: a row where 1..39 appear across consecutive cells
    header_row = col_start = None
    for r, row in enumerate(grid):
        col_start = _find_run([_num(x) for x in row], 39)
        if col_start is not None:
            header_row = r
            break
    # improving parameters: a column where 1..39 appear down consecutive rows
    imp_col = row_start = None
    for c in range(len(grid[0]) if grid else 0):
        row_start = _find_run([_num(row[c]) for row in grid], 39)
        if row_start is not None:
            imp_col = c
            break
    if header_row is None or imp_col is None:
        raise ValueError("Could not find parameters 1-39 along a row and a column of the TRIZ matrix.")

    matrix = {}
    for i in range(1, 40):
        for j in range(1, 40):
            if i == j:
                continue
            cell = grid[row_start + i - 1][col_start + j - 1]
            ps = [int(x) for x in re.findall(r"\d+", cell) if 1 <= int(x) <= 40]
            if ps:
                matrix[(i, j)] = list(dict.fromkeys(ps))
    if len(matrix) < 800:
        raise ValueError(f"Only {len(matrix)} filled cells found in the TRIZ matrix. Check the file layout.")

    # parameter names, if the column next to the improving numbers holds them
    names = [" ".join(grid[row_start + k][imp_col + 1].split()) if imp_col + 1 < len(grid[0]) else ""
             for k in range(39)]
    parameters = names if all(n and _num(n) is None for n in names) else TRIZ_PARAMETERS
    return {"matrix": matrix, "parameters": parameters}


def parse_triz_principles(grid: list):
    """Rows numbered 1..40 down a column, name in the next column, description after it."""
    for c in range(max(0, len(grid[0]) - 1) if grid else 0):
        start = _find_run([_num(row[c]) if str(row[c]).strip().isdigit() else None for row in grid], 40)
        if start is None:
            continue
        pr = {}
        for k in range(40):
            row = grid[start + k]
            name = " ".join(row[c + 1].split()).rstrip(":;,. ")
            desc = row[c + 2].strip() if c + 2 < len(row) else ""
            if not name or _num(name) is not None:
                break
            pr[k + 1] = {"name": name, "description": desc}
        if len(pr) == 40:
            return pr
    return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GEMINI CALL                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Generator:
    def __init__(self):
        if MOCK:
            return
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def call(self, prompt: str, step: str) -> tuple:
        if MOCK:
            return json.dumps(mock_step(step, prompt)), {}
        t = self.types
        cfg = dict(system_instruction=SYSTEM_PROMPT, response_mime_type="application/json",
                   max_output_tokens=MAX_OUTPUT_TOKENS)
        try:
            config = t.GenerateContentConfig(**cfg, thinking_config=t.ThinkingConfig(thinking_level=THINKING_LEVEL))
        except Exception:
            config = t.GenerateContentConfig(**cfg)
        resp = self.client.models.generate_content(model=GENERATION_MODEL, contents=prompt, config=config)
        usage = {}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {k: getattr(um, k, None) for k in
                     ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")}
        return resp.text or "", usage


class Chain:
    """Runs one session's steps, retrying a step until its output passes the check."""

    def __init__(self, gen: Generator, sid: str):
        self.gen, self.sid, self.steps = gen, sid, []

    def step(self, name: str, prompt: str, check=None) -> dict:
        feedback = ""   # after a rejected answer, the next attempt is told why
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw, usage = self.gen.call(prompt + feedback, name)
                data = extract_json(raw)
                if check:
                    check(data)
                self.steps.append({"step": name, "prompt": prompt + feedback, "raw_response": raw,
                                   "parsed": data, "usage": usage, "attempts": attempt, "timestamp": now()})
                return data
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                if isinstance(e, ValueError):   # the answer arrived but failed a check
                    feedback = (f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED: {e}. "
                                "Answer again in full, correcting this.")
                wait = min(60, 2 ** attempt) + random.random()
                log(f"  {self.sid} [{name}] attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]}), retrying in {wait:.0f}s")
                time.sleep(wait)

    def code_step(self, name: str, result: dict) -> None:
        """Record a step done by this script (lookup or random draw), not by the model."""
        self.steps.append({"step": name, "by": "script", "result": result, "timestamp": now()})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  THE FIVE PROCEDURES                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_generic(ch: Chain, brief: dict, rng: random.Random) -> tuple:
    def chk(d):
        for k in ("requirements", "constraints", "key_challenges"):
            if not isinstance(d.get(k), list) or not d[k]:
                raise ValueError(f"missing {k}")
    a = ch.step("1_analyse_brief", (
        f"{brief_text(brief)}\n"
        "STEP 1 OF 2. Analyse the design brief before generating any ideas.\n"
        "List the requirements the design must meet, the constraints it must respect, and the key "
        "design challenges that a good concept has to resolve.\n"
        'Return JSON: {"requirements": ["..."], "constraints": ["..."], "key_challenges": ["..."]}'
    ), chk)
    final = ch.step("2_generate_concepts", (
        f"{brief_text(brief)}\n"
        f"ANALYSIS FROM STEP 1\n{as_json(a)}\n\n"
        f"STEP 2 OF 2. Generate {N_CONCEPTS} distinct design concepts that meet the requirements and "
        "constraints and resolve the key challenges identified above.\n"
        'In "method_trace" name the key challenge the concept mainly addresses.\n'
        f"{FINAL_FORMAT}"
    ), check_concepts)
    workings = "Key challenges: " + "; ".join(a["key_challenges"])
    return final, workings


def run_scamper(ch: Chain, brief: dict, rng: random.Random) -> tuple:
    def chk(d):
        for k in ("components", "materials", "functions", "use"):
            if k not in d:
                raise ValueError(f"missing {k}")
    a = ch.step("1_describe_existing_product", (
        f"{brief_text(brief)}\n"
        "STEP 1 OF 3. SCAMPER starts from an existing product. "
        + ("Describe the product named in the brief as it is today"
           if brief.get("product") else
           "Identify the product or solution most commonly used for this purpose today, and describe it")
        + ": its main components, materials, functions, and how it is used.\n"
        'Include its name in "product". '

        'Return JSON: {"product": "...", "components": ["..."], "materials": ["..."], "functions": ["..."], '
        '"use": "..."}'
    ), chk)

    names = [n for n, _ in SCAMPER_OPERATORS]
    ops = "\n".join(f"  {i+1}. {n}: {d}" for i, (n, d) in enumerate(SCAMPER_OPERATORS))

    def chk2(d):
        sel = d.get("selected")
        if not isinstance(sel, list) or len(sel) != N_CONCEPTS:
            raise ValueError(f"expected {N_CONCEPTS} selected operators")
        ops_chosen = [x.get("operator") for x in sel]
        if any(o not in names for o in ops_chosen) or len(set(ops_chosen)) != N_CONCEPTS:
            raise ValueError(f"operators must be {N_CONCEPTS} different names from the list")
    b = ch.step("2_select_operators", (
        f"{brief_text(brief)}\n"
        f"EXISTING PRODUCT FROM STEP 1\n{as_json(a)}\n\n"
        f"STEP 2 OF 3. From the seven SCAMPER operators below, select the {N_CONCEPTS} most promising for "
        "this product and brief. For each, name the specific component, material, function or aspect of "
        "use of the existing product you will apply it to, and say in one sentence why.\n"
        f"{ops}\n"
        'Return JSON: {"selected": [{"operator": "<name exactly as listed>", "aspect": "...", "reason": "..."}]}'
    ), chk2)

    plan = "\n".join(f"  Concept {n+1}: {x['operator']} applied to {x['aspect']}" for n, x in enumerate(b["selected"]))
    final = ch.step("3_apply_operators", (
        f"{brief_text(brief)}\n"
        f"EXISTING PRODUCT FROM STEP 1\n{as_json(a)}\n\n"
        "STEP 3 OF 3. Apply each selected SCAMPER operator to its aspect of the existing product and "
        "develop the result into a design concept that meets the brief. One concept per operator, in this order:\n"
        f"{plan}\n"
        'In "method_trace" give the operator and the aspect it was applied to.\n'
        f"{FINAL_FORMAT}"
    ), check_concepts)
    return final, f"Starting product: {a.get('product', '')} | Operators: " + "; ".join(f"{x['operator']} ({x['aspect']})" for x in b["selected"])


def run_triz(ch: Chain, brief: dict, rng: random.Random, triz: dict) -> tuple:
    matrix, pnames, principles = triz["matrix"], triz["parameters"], triz["principles"]
    params = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(pnames))

    def chk(d):
        cs = d.get("contradictions")
        if not isinstance(cs, list) or len(cs) != 3:
            raise ValueError("give exactly 3 contradictions")
        empty = []
        for c in cs:
            i, j = int(c["improving"]), int(c["worsening"])
            if not (1 <= i <= 39 and 1 <= j <= 39) or i == j:
                raise ValueError("parameters must be two different numbers from 1 to 39")
            if (i, j) not in matrix:
                empty.append(f"improving {i} vs worsening {j}")
        if empty:   # as in TRIZ practice, restate the contradiction with neighbouring parameters
            raise ValueError("the contradiction matrix gives no principles for " + "; ".join(empty)
                             + ". Restate that contradiction using other, closely related parameters")

    a = ch.step("1_identify_contradictions", (
        f"{brief_text(brief)}\n"
        "STEP 1 OF 4. Identify the technical contradictions in this design problem. A technical "
        "contradiction is a pair of engineering parameters where improving one makes the other worse. "
        "Identify the 3 most important contradictions, each expressed with the 39 TRIZ engineering "
        "parameters below, and explain each in one sentence in terms of this product.\n"
        f"{params}\n"
        'Return JSON: {"contradictions": [{"improving": <number>, "worsening": <number>, '
        '"explanation": "..."}]}'
    ), chk)

    # Step 2, done in code: look up the contradiction matrix
    counts, first_seen, lookups = {}, {}, []
    for k, c in enumerate(a["contradictions"]):
        i, j = int(c["improving"]), int(c["worsening"])
        ps = matrix.get((i, j), [])
        lookups.append({"improving": i, "improving_name": pnames[i - 1],
                        "worsening": j, "worsening_name": pnames[j - 1], "principles": ps,
                        "explanation": c.get("explanation", "")})
        for pos, p in enumerate(ps):
            counts[p] = counts.get(p, 0) + 1
            first_seen.setdefault(p, (k, pos))
    recommended = sorted(counts, key=lambda p: (-counts[p], first_seen[p]))   # most frequent first
    ch.code_step("2_matrix_lookup", {"lookups": lookups, "recommended_principles": recommended,
                                     "recommended_names": [principles[p]["name"] for p in recommended]})

    contra_text = "\n".join(
        f"  - Improving {l['improving']} {l['improving_name']} vs worsening {l['worsening']} "
        f"{l['worsening_name']}: {l['explanation']} -> matrix recommends {l['principles']}" for l in lookups)

    def describe(ps):
        return "\n\n".join(f"Principle {p}. {principles[p]['name']}\n{principles[p]['description']}".rstrip()
                            for p in ps)

    # Step 3: the model selects the principles to use from those the matrix recommends
    if len(recommended) > N_CONCEPTS:
        def chk3(d):
            sel = [int(x.get("principle", 0)) for x in d.get("selected", [])]
            if len(sel) != N_CONCEPTS or len(set(sel)) != N_CONCEPTS or any(p not in recommended for p in sel):
                raise ValueError(f"select {N_CONCEPTS} different principles from the recommended list")
        b = ch.step("3_select_principles", (
            f"{brief_text(brief)}\n"
            f"CONTRADICTIONS AND MATRIX RECOMMENDATIONS\n{contra_text}\n\n"
            f"STEP 3 OF 4. From the principles the matrix recommends, select the {N_CONCEPTS} that are most "
            "applicable to this product, and say which contradiction each one resolves.\n\n"
            f"RECOMMENDED PRINCIPLES\n{describe(recommended)}\n\n"
            'Return JSON: {"selected": [{"principle": <number>, "contradiction": "...", "reason": "..."}]}'
        ), chk3)
        chosen = [int(x["principle"]) for x in b["selected"]]
    else:
        chosen = [recommended[n % len(recommended)] for n in range(N_CONCEPTS)]   # too few to choose from
        ch.code_step("3_select_principles", {"note": "matrix recommended too few principles to select from; "
                                             "all were used", "chosen": chosen})

    plist = "\n".join(f"  Concept {n+1}: principle {p} {principles[p]['name']}" for n, p in enumerate(chosen))
    final = ch.step("4_apply_principles", (
        f"{brief_text(brief)}\n"
        f"CONTRADICTIONS\n{contra_text}\n\n"
        "STEP 4 OF 4. For each concept, apply the assigned inventive principle to resolve one of the "
        "contradictions, and develop the result into a design concept that meets the brief. If a "
        "principle appears twice, apply it in a different way the second time.\n"
        f"{plist}\n\n"
        f"WHAT THE PRINCIPLES MEAN (with the standard TRIZ examples)\n{describe(list(dict.fromkeys(chosen)))}\n\n"
        'In "method_trace" give the principle number and name and the contradiction it resolves.\n'
        f"{FINAL_FORMAT}"
    ), check_concepts)
    workings = ("Contradictions: " + "; ".join(f"{l['improving_name']} vs {l['worsening_name']} -> {l['principles']}"
                                               for l in lookups)
                + " | Selected: " + ", ".join(f"{p} {principles[p]['name']}" for p in chosen))
    return final, workings


def run_analogy(ch: Chain, brief: dict, rng: random.Random) -> tuple:
    def chk1(d):
        if not isinstance(d.get("key_functions"), list) or not d["key_functions"]:
            raise ValueError("missing key_functions")
    a = ch.step("1_abstract_functions", (
        f"{brief_text(brief)}\n"
        "STEP 1 OF 3. Design-by-analogy starts by abstracting the problem. State the key functions the "
        "design must perform in solution-neutral terms, as a verb and a noun (for example 'retain heat', "
        "'separate particles'), without naming any existing product or solution. Then state the core "
        "design problem in one solution-neutral sentence.\n"
        'Return JSON: {"key_functions": ["..."], "core_problem": "..."}'
    ), chk1)

    domains = rng.sample(SOURCE_DOMAINS, N_CONCEPTS)        # step 2, done in code
    ch.code_step("2_draw_source_domains", {"domains": domains})

    def chk3(d):
        s = d.get("sources")
        if not isinstance(s, list) or len(s) != N_CONCEPTS:
            raise ValueError(f"expected {N_CONCEPTS} sources")
    dlist = "\n".join(f"  {n+1}. {d}" for n, d in enumerate(domains))
    b = ch.step("3_find_sources", (
        f"{brief_text(brief)}\n"
        f"ABSTRACTED PROBLEM FROM STEP 1\n{as_json(a)}\n\n"
        "STEP 2 OF 3. For each source domain below, in this order, find one specific system in that "
        "domain that already achieves one of the key functions. Describe how it achieves the function: "
        "its structure, the mechanism or principle, and why it works.\n"
        f"{dlist}\n"
        'Return JSON: {"sources": [{"domain": "...", "source_system": "...", "function": "...", '
        '"how_it_works": "..."}]}'
    ), chk3)

    final = ch.step("4_map_and_transfer", (
        f"{brief_text(brief)}\n"
        f"ABSTRACTED PROBLEM\n{as_json(a)}\n\n"
        f"SOURCE SYSTEMS FROM STEP 2\n{as_json(b)}\n\n"
        "STEP 3 OF 3. For each source system, in order, carry out the analogical transfer. First map the "
        "source onto the target: which element of the source corresponds to which element of the product, "
        "and which relation or mechanism is being transferred. Transfer the underlying mechanism, not the "
        "surface appearance. Then adapt it to the brief's constraints and develop it into a design concept.\n"
        'In "method_trace" give the source system and the mechanism transferred.\n'
        f"{FINAL_FORMAT}"
    ), check_concepts)
    workings = "Sources: " + "; ".join(f"{s['domain']}: {s['source_system']}" for s in b["sources"])
    return final, workings


def run_morphological(ch: Chain, brief: dict, rng: random.Random) -> tuple:
    def chk1(d):
        sf = d.get("sub_functions")
        if not isinstance(sf, list) or not 4 <= len(sf) <= 6:
            raise ValueError("expected 4 to 6 sub-functions")
    a = ch.step("1_decompose_functions", (
        f"{brief_text(brief)}\n"
        "STEP 1 OF 3. Decompose the design problem into 4 to 6 sub-functions that together achieve the "
        "goal of the brief, including its condition. Express each in solution-neutral terms, as a verb and a "
        "noun, without naming a solution.\n"
        'Return JSON: {"sub_functions": [{"name": "...", "description": "..."}]}'
    ), chk1)

    def chk2(d):
        m = d.get("matrix")
        if not isinstance(m, list) or len(m) != len(a["sub_functions"]):
            raise ValueError("matrix must have one row per sub-function")
        for row in m:
            if not 3 <= len(row.get("options", [])) <= 5:
                raise ValueError("each sub-function needs 3 to 5 options")
    b = ch.step("2_brainstorm_options", (
        f"{brief_text(brief)}\n"
        f"SUB-FUNCTIONS FROM STEP 1\n{as_json(a)}\n\n"
        "STEP 2 OF 3. Build the morphological matrix. For each sub-function, brainstorm 3 to 5 alternative "
        "means of achieving it. Make the options for each sub-function genuinely different from one "
        "another (different principles, not variations of one idea).\n"
        'Return JSON: {"matrix": [{"sub_function": "...", "options": ["..."]}]}'
    ), chk2)

    final = ch.step("3_combine", (
        f"{brief_text(brief)}\n"
        f"MORPHOLOGICAL MATRIX FROM STEP 2\n{as_json(b)}\n\n"
        f"STEP 3 OF 3. Select {N_CONCEPTS} combinations from the matrix, taking exactly one option for each "
        "sub-function. The options in a combination must be compatible with one another, and the "
        f"{N_CONCEPTS} combinations must differ substantially from each other. Develop each combination "
        "into an integrated design concept.\n"
        'In "method_trace" list the option chosen for each sub-function.\n'
        f"{FINAL_FORMAT}"
    ), check_concepts)
    workings = "Sub-functions: " + "; ".join(r["sub_function"] for r in b["matrix"])
    return final, workings


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MOCK RESPONSES (MOCK = True)                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def mock_step(step: str, prompt: str) -> dict:
    time.sleep(0.003)
    if step == "1_analyse_brief":
        return {"requirements": ["r1"], "constraints": ["c1"], "key_challenges": ["k1", "k2"]}
    if step == "1_describe_existing_product":
        return {"product": "p", "components": ["body", "lid"], "materials": ["paper"], "functions": ["hold"], "use": "u"}
    if step == "1_identify_contradictions":
        return {"contradictions": [{"improving": random.randint(1, 20), "worsening": random.randint(21, 39),
                                    "explanation": "x"} for _ in range(3)]}
    if step == "2_select_operators":
        ops = random.sample([n for n, _ in SCAMPER_OPERATORS], N_CONCEPTS)
        return {"selected": [{"operator": o, "aspect": "lid", "reason": "r"} for o in ops]}
    if step == "3_select_principles":
        rec = [int(x) for x in re.findall(r"^Principle (\d+)\.", prompt, re.M)]
        return {"selected": [{"principle": p, "contradiction": "c", "reason": "r"} for p in rec[:N_CONCEPTS]]}
    if step == "1_abstract_functions":
        return {"key_functions": ["retain heat"], "core_problem": "p"}
    if step == "3_find_sources":
        return {"sources": [{"domain": f"d{n}", "source_system": f"s{n}", "function": "f", "how_it_works": "h"}
                            for n in range(N_CONCEPTS)]}
    if step == "1_decompose_functions":
        return {"sub_functions": [{"name": f"sf{n}", "description": "d"} for n in range(4)]}
    if step == "2_brainstorm_options":
        return {"matrix": [{"sub_function": f"sf{n}", "options": ["a", "b", "c"]} for n in range(4)]}
    words = " ".join(["word"] * random.randint(60, 100))
    return {"concepts": [{"id": i + 1, "title": f"Mock concept {i+1}", "description": f"Concept {random.random():.4f} {words}",
                          "method_trace": "mock"} for i in range(N_CONCEPTS)]}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RUN                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_dir() -> Path:
    return RESULTS_DIR / ("pilot" if PILOT else RUN_NAME)


def briefs_in_design() -> list:
    if not PILOT:
        return BRIEFS
    first = {}
    for b in BRIEFS:                       # pilot: the first adaptive and the first original brief
        first.setdefault(b["type"], b)
    return list(first.values())


def runs_in_design() -> int:
    return 2 if PILOT else N_RUNS


def session_file(sid: str) -> Path:
    return run_dir() / "sessions" / f"{sid}.json"


def run_session(gen: Generator, triz: dict, brief: dict, condition: str, run: int) -> str:
    sid = f"{brief['id']}__{condition}__r{run}"
    if session_file(sid).exists():
        return f"{sid} (already done)"
    rng = random.Random(f"{RANDOM_SEED}-{sid}")     # reproducible draws for this session
    ch = Chain(gen, sid)
    t0 = time.time()
    if condition == "generic":
        final, workings = run_generic(ch, brief, rng)
    elif condition == "scamper":
        final, workings = run_scamper(ch, brief, rng)
    elif condition == "triz":
        final, workings = run_triz(ch, brief, rng, triz)
    elif condition == "analogy":
        final, workings = run_analogy(ch, brief, rng)
    elif condition == "morphological":
        final, workings = run_morphological(ch, brief, rng)
    else:
        raise ValueError(condition)
    save_json(session_file(sid), {
        "session_id": sid, "brief_id": brief["id"], "condition": condition, "run": run,
        "model": "MOCK" if MOCK else GENERATION_MODEL, "thinking_level": THINKING_LEVEL,
        "timestamp": now(), "seconds": round(time.time() - t0, 1),
        "n_model_calls": sum(1 for s in ch.steps if s.get("by") != "script"),
        "system_prompt": SYSTEM_PROMPT, "method_workings": workings, "steps": ch.steps,
        "concepts": [{"id": c.get("id", i + 1), "title": c["title"], "description": c["description"],
                      "method_trace": c.get("method_trace", ""), "word_count": word_count(c["description"])}
                     for i, c in enumerate(final["concepts"])],
    })
    return sid


def compile_outputs(expected: int) -> Path:
    sessions = [json.loads(p.read_text()) for p in sorted((run_dir() / "sessions").glob("*.json"))]
    out = run_dir() / "generation_results.json"
    save_json(out, {
        "metadata": {
            "generation_model": "MOCK" if MOCK else GENERATION_MODEL, "thinking_level": THINKING_LEVEL,
            "pilot": PILOT, "mock": MOCK, "conditions": CONDITIONS, "baseline_condition": BASELINE,
            "n_concepts": N_CONCEPTS, "n_runs": runs_in_design(),
            "n_sessions_expected": expected, "n_sessions_complete": len(sessions),
            "triz_matrix_file": str(TRIZ_MATRIX_FILE), "source_domains": SOURCE_DOMAINS,
            "compiled": now(), "system_prompt": SYSTEM_PROMPT,
        },
        "briefs": briefs_in_design(),
        "sessions": sessions,
    })

    lines = ["# Idea Generation Results", "",
             f"**Model**: {'MOCK' if MOCK else GENERATION_MODEL} (thinking: {THINKING_LEVEL})  ",
             f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
             f"**Sessions**: {len(sessions)} of {expected}", "",
             "| Condition | Sessions | Model calls per session | Mean words per concept |", "|---|:---:|:---:|:---:|"]
    for c in CONDITIONS:
        ss = [s for s in sessions if s["condition"] == c]
        if ss:
            w = [x["word_count"] for s in ss for x in s["concepts"]]
            calls = sum(x["n_model_calls"] for x in ss) / len(ss)
            lines.append(f"| {c} | {len(ss)} | {calls:.1f} | {sum(w) / len(w):.0f} |")
    lines.append("")
    for b in briefs_in_design():
        lines += [f"## {b['id']} ({b['type']}): {b['title']}", ""]
        for s in [s for s in sessions if s["brief_id"] == b["id"]]:
            lines += [f"### {s['condition']} (run {s['run']})", ""]
            for st in s["steps"][:-1]:     # every step except the final concepts
                body = st["result"] if st.get("by") == "script" else st["parsed"]
                tag = " *(script)*" if st.get("by") == "script" else ""
                lines += [f"<details><summary>{st['step']}{tag}</summary>", "", "```json",
                          json.dumps(body, indent=1, ensure_ascii=False), "```", "</details>", ""]
            for c in s["concepts"]:
                lines.append(f"{c['id']}. **{c['title']}** *[{c['method_trace']}]*: {c['description']}")
            lines.append("")
    (run_dir() / "generation_summary.md").write_text("\n".join(lines))
    return out


def main():
    problems = check_briefs(BRIEFS)
    if problems:
        print("=" * 60)
        print("ERROR: please fix the design briefs (BRIEFS near the top of this file):")
        for p in problems:
            print(f"  - {p}")
        print("=" * 60)
        sys.exit(1)
    if not MOCK and GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("=" * 60)
        print("ERROR: Please set your Gemini API key!")
        print("  Open this file and paste it where it says YOUR_API_KEY_HERE")
        print("=" * 60)
        sys.exit(1)

    triz = None
    if "triz" in CONDITIONS:
        try:
            triz = load_triz(triz_path())
        except Exception as e:
            print("=" * 60)
            print(f"ERROR loading the TRIZ matrix: {e}")
            if "xlrd" in str(e):
                print("  Reading .xls files needs one more package: pip install xlrd")
            print("=" * 60)
            sys.exit(1)
        m, pn, pr = triz["matrix"], triz["parameters"], triz["principles"]
        print(f"TRIZ matrix: {len(m)} filled cells, {len(pn)} parameters, {len(pr)} principles"
              f"{' with descriptions' if any(v['description'] for v in pr.values()) else ' (names only)'}.")
        print("  Spot-check against your table:")
        for i, j in [(1, 3), (9, 14), (27, 36)]:
            print(f"    improving {i} {pn[i-1]} vs worsening {j} {pn[j-1]}: {m.get((i, j), '(empty)')}")

    gen = Generator()
    jobs = [(b, c, r) for b in briefs_in_design() for c in CONDITIONS for r in range(1, runs_in_design() + 1)]
    random.Random(RANDOM_SEED).shuffle(jobs)
    todo = [j for j in jobs if not session_file(f"{j[0]['id']}__{j[1]}__r{j[2]}").exists()]

    print("=" * 60)
    print(f"  Idea generation   model: {'MOCK' if MOCK else GENERATION_MODEL}")
    print(f"  Saving to: {run_dir()}")
    print(f"  Sessions: {len(jobs)} in design, {len(todo)} still to run")
    print("=" * 60)

    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(run_session, gen, triz, *j): j for j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                log(f"[{i}/{len(todo)}] {f.result()}")
            except Exception as e:
                failures += 1
                b, c, r = futs[f]
                log(f"[{i}/{len(todo)}] FAILED {b['id']} {c} r{r}: {e}")

    out = compile_outputs(len(jobs))
    print()
    print("=" * 60)
    print(f"  {failures} sessions failed. Run this file again to retry them." if failures
          else "  DONE! All sessions generated.")
    print(f"  Results: {out}")
    print(f"  Readable version: {out.parent / 'generation_summary.md'}")
    print("  Next: run the judge script on generation_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
