import hashlib
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

OPENAI_API_KEY =
JUDGE_MODEL = "gpt-5.6-luna"            # the model both pilot rounds used; keep it fixed
EMBEDDING_MODEL = "text-embedding-3-large"
RANK_EFFORT = "low"                     # reasoning effort for ranking a group
RATE_EFFORT = "low"                     # life-cycle rating passes
RESCORE_EFFORT = "high"                 # life-cycle rating when the two passes disagree
CODING_EFFORT = "low"                   # describing principles
GROUPING_EFFORT = "medium"              # building the principle tree for each brief
MAX_COMPLETION_TOKENS = 16000

GROUP_SIZE = 5                          # concepts ranked together in one call
GROUPS_PER_CONCEPT = 4                  # groups each concept appears in, per criterion (pilot used 6;
                                        # 4 predicts split-half ~0.82-0.89; raise it later if C1 is low)
REVERSE_CHECK_SHARE = 0.10              # share of groups ranked again with the order reversed
RATE_REPEATS = 2                        # life-cycle rating passes per concept
RESCORE_GAP = 2                         # third pass if the two passes differ by >= this
SHAH_WEIGHTS = {"physical": 10, "working": 6}
HUMAN_CHECK_PAIRS = 30                  # pairs in the optional expert sheet
BOOTSTRAP = 2000                        # resamples for confidence intervals

SCOPE = "iced"                          # "iced" or "journal" (see the top of this file)

INPUT_FILE = Path(__file__).resolve().parent / "results" / "run_01" / "generation_results.json"
OUTPUT_FOLDER = "evaluation_v2"         # created next to the input file
MOCK = False                            # True = simulated answers, no API calls, no cost

MAX_WORKERS = 4                         # parallel calls; lower it if you hit rate limits
MAX_RETRIES = 6
RANDOM_SEED = 2027

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WHAT THE JUDGE IS ASKED                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

JUDGE_SYSTEM = (
    "You are an expert engineering design researcher who evaluates early-stage "
    "design concepts. You follow the evaluation procedure exactly as written and "
    "judge the concepts on their merits against the brief. Return only valid JSON."
)

CRITERIA = {
    "novelty": {
        "adjective": "novel",
        "definition": (
            "Novelty means how different a concept is from the products and solutions that already "
            "exist for this purpose today. A new working principle, a new structure or layout, or a new "
            "way for users to carry out the task all make a concept more novel. A familiar solution with "
            "parts added, removed or restyled is not very novel."),
        "exclude": (
            "Judge novelty only. Do not consider whether the concept would work, how useful it is, or how "
            "well it is written. A longer or more detailed description is not more novel."),
    },
    "usefulness": {
        "adjective": "useful",
        "definition": (
            "Usefulness means how well a concept would achieve the goal of the brief, including the "
            "condition the goal states, while being technically feasible and practical for its users "
            "(no major added burden in use, maintenance or cost)."),
        "exclude": (
            "Judge whether the mechanism would plausibly work. Do not reward figures or benefits that the "
            "description claims but does not make credible. Judge usefulness only, not novelty or writing "
            "quality. A longer or more detailed description is not more useful."),
    },
}

# Han, Jiang & Childs (2021), Energies 14(12), 3469: four life-cycle metrics.
LIFECYCLE_STAGES = ["material", "production", "use", "end_of_life"]
LIFECYCLE_TEXT = {
    "material": "Material: the type, amount and origin of materials (e.g. less material, renewable, recycled, non-toxic, fewer material types).",
    "production": "Production: manufacturing and assembly (e.g. fewer processes, less energy, less waste, simpler assembly).",
    "use": "Use: the use phase (e.g. energy, water or consumables used in operation, durability, maintenance, repair, upgrade).",
    "end_of_life": "End of life: what happens afterwards (e.g. reuse, remanufacture, recycling, composting, ease of disassembly).",
}
SPECIFICITY_TEXT = """Specificity: how much engineering detail (materials, mechanisms, geometry, interactions) the description gives.
  1 = a vague aspiration or slogan
  2 = a general idea with one concrete element
  3 = the main working principle is clear but key details are missing
  4 = working principle and most key details are given
  5 = detailed enough to start sketching and prototyping"""

LETTERS = "ABCDEFGHIJ"


def rank_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {"reasoning": {"type": "string"},
                       "ranking": {"type": "array", "items": {"type": "string", "enum": list(LETTERS[:n])}}},
        "required": ["reasoning", "ranking"],
        "additionalProperties": False,
    }
STAGE = {"type": "integer", "enum": [-1, 0, 1, 2]}
RATE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"},
                   **{f"sus_{s}": STAGE for s in LIFECYCLE_STAGES},
                   "specificity": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
    "required": ["rationale", *[f"sus_{s}" for s in LIFECYCLE_STAGES], "specificity"],
    "additionalProperties": False,
}
CODING_SCHEMA = {
    "type": "object",
    "properties": {"physical_principle": {"type": "string"}, "working_principle": {"type": "string"}},
    "required": ["physical_principle", "working_principle"],
    "additionalProperties": False,
}
GROUPING_SCHEMA = {
    "type": "object",
    "properties": {"physical_principles": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "working_principles": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "concept_labels": {"type": "array", "items": {"type": "string"}}},
                "required": ["name", "concept_labels"], "additionalProperties": False}},
        },
        "required": ["name", "working_principles"], "additionalProperties": False}}},
    "required": ["physical_principles"],
    "additionalProperties": False,
}

RATINGS = [*[f"sus_{s}" for s in LIFECYCLE_STAGES], "specificity"]
RATING_RANGES = {"specificity": (1, 5), **{f"sus_{s}": (-1, 2) for s in LIFECYCLE_STAGES}}


def brief_text(brief: dict) -> str:
    kind = "Redesign of an existing product" if brief["type"] == "adaptive" else "New product design"
    lines = [f"DESIGN BRIEF ({kind})"]
    if brief.get("product"):
        lines.append(f"Product: {brief['product']}")
    lines.append(f"Goal: {brief['goal']}")
    return "\n".join(lines) + "\n"


def rank_prompt(brief: dict, members: list, criterion: str) -> str:
    c = CRITERIA[criterion]
    n = len(members)
    blocks = "\n\n".join(f"CONCEPT {LETTERS[i]}\nTitle: {m['title']}\nDescription: {m['description']}"
                          for i, m in enumerate(members))
    return (
        f"{brief_text(brief)}\n"
        f"{n} design concepts for this brief are shown below, in random order.\n\n"
        f"{blocks}\n\n"
        f"QUESTION: Rank all {n} concepts from most {c['adjective']} to least {c['adjective']}.\n"
        f"{c['definition']}\n"
        f"{c['exclude']}\n"
        "The order in which the concepts are shown means nothing. If two are very close, still put one above the other.\n\n"
        f"First write a short reasoning (max 100 words) comparing the concepts, then give the ranking as a list of "
        f"all {n} letters, most {c['adjective']} first, each letter exactly once."
    )


def rate_prompt(brief: dict, concept: dict, careful: bool = False) -> str:
    stages = "\n".join(f"  {LIFECYCLE_TEXT[s]}" for s in LIFECYCLE_STAGES)
    extra = ("\nThis concept received inconsistent ratings before. Reason carefully "
             "through every stage before answering.\n") if careful else ""
    return (
        f"{brief_text(brief)}\n"
        "CONCEPT TO RATE\n"
        f"Title: {concept['title']}\n"
        f"Description: {concept['description']}\n\n"
        "PART A. SUSTAINABILITY BY LIFE-CYCLE STAGE\n"
        "Compare the concept with existing products or solutions that serve the same purpose today. "
        "For each stage give\n"
        "  -1 = worse than existing products or solutions\n"
        "   0 = no meaningful difference\n"
        "   1 = a moderate improvement\n"
        "   2 = a substantial improvement through a concrete, credible mechanism\n"
        f"{stages}\n\n"
        "PART B. SPECIFICITY\n"
        f"{SPECIFICITY_TEXT}\n"
        f"{extra}\n"
        "First write a short rationale (max 80 words) covering parts A and B, then give the answers."
    )


def coding_prompt(brief: dict, concept: dict) -> str:
    return (
        f"{brief_text(brief)}\n"
        "CONCEPT\n"
        f"Title: {concept['title']}\n"
        f"Description: {concept['description']}\n\n"
        "Describe this concept, do not evaluate it.\n"
        "1. physical_principle: the main physical, chemical or biological principle the concept relies on "
        "to achieve the goal, in a few words (e.g. 'thermal insulation', 'gravity-driven flow').\n"
        "2. working_principle: how that principle is put to work in this concept, in one short phrase "
        "(e.g. 'vacuum double wall around the water chamber')."
    )


def grouping_prompt(brief: dict, items: list) -> str:
    lines = "\n".join(f"[{lab}] {c['title']} | physical principle: {code['physical_principle']} | "
                      f"working principle: {code['working_principle']}" for lab, c, code in items)
    return (
        f"{brief_text(brief)}\n"
        "Below is a pool of design concepts for this brief. Build a genealogy tree of the pool, "
        "following Shah et al. (2003).\n"
        "Level 1: group the concepts by the physical principle they rely on. Concepts relying on "
        "the same underlying principle belong together even if worded differently.\n"
        "Level 2: within each physical principle, group the concepts by working principle, meaning "
        "the way that principle is put to work. Concepts that differ only in details, styling or "
        "materials of the same kind share a working principle.\n"
        "Use as many groups as the concepts really need, no more and no fewer. "
        "Every concept label must appear exactly once in the tree.\n\n"
        f"{lines}\n"
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS AND API CALLS                                                  ║
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


def load_json(path: Path):
    return json.loads(path.read_text())


def retry(fn, what: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = min(60, 2 ** attempt) + random.random()
            log(f"  {what}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:150]}), retrying in {wait:.0f}s")
            time.sleep(wait)


def seeded(*parts) -> random.Random:
    """A random generator that always gives the same numbers for the same inputs."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


class Judge:
    def __init__(self):
        if MOCK:
            return
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def ask(self, prompt: str, schema: dict, name: str, effort: str, mock) -> tuple:
        if MOCK:
            time.sleep(0.002)
            return mock(), {}
        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}],
            reasoning_effort=effort,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "schema": schema, "strict": True}},
        )
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {"prompt_tokens": response.usage.prompt_tokens,
                     "completion_tokens": response.usage.completion_tokens}
        return json.loads(response.choices[0].message.content), usage

    def embed(self, texts: list) -> list:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in response.data]


# Simulated answers for MOCK mode. They contain built-in differences between
# conditions, so a mock run also checks that the analysis recovers known effects.
MOCK_EFFECTS = {"novelty": {"generic": 0.0, "scamper": -0.2, "triz": 0.4, "analogy": 0.8, "morphological": 0.4},
                "usefulness": {"generic": 0.0, "scamper": 0.2, "triz": 0.0, "analogy": -0.4, "morphological": -0.2}}


def mock_latent(c: dict, criterion: str) -> float:
    return MOCK_EFFECTS[criterion].get(c["condition"], 0.0) + seeded(c["concept_id"], criterion).gauss(0, 1)


def mock_rank(members: list, criterion: str, key: str) -> dict:
    """Plackett-Luce sampling (Gumbel noise) with a slight bonus for the first position shown."""
    r = seeded(key, criterion)
    noisy = [(mock_latent(m, criterion) + (0.15 if i == 0 else 0) - math.log(-math.log(r.random() or 1e-12)), LETTERS[i])
             for i, m in enumerate(members)]
    return {"reasoning": "mock", "ranking": [lab for _, lab in sorted(noisy, reverse=True)]}


def mock_rate(c: dict, rep: str) -> dict:
    r = seeded(c["concept_id"], rep)
    base = {"generic": 0, "scamper": 0.3, "triz": 0.0, "analogy": -0.2, "morphological": 0.1}.get(c["condition"], 0)
    clip = lambda v: int(max(-1, min(2, round(v))))  # noqa: E731
    return {"rationale": "mock", **{f"sus_{s}": clip(base + r.gauss(0.3, 0.8)) for s in LIFECYCLE_STAGES},
            "specificity": int(max(1, min(5, round(r.gauss(3.5, 0.7)))))}


def mock_code(c: dict) -> dict:
    r = seeded(c["concept_id"], "code")
    n = {"generic": 3, "scamper": 4, "triz": 5, "analogy": 7, "morphological": 5}.get(c["condition"], 4)
    p = f"principle {r.randint(1, n)}"
    return {"physical_principle": p, "working_principle": f"{p} variant {r.randint(1, 3)}"}


def mock_embedding(c: dict, dim: int = 64) -> list:
    spread = {"generic": 0.6, "scamper": 0.7, "triz": 0.8, "analogy": 1.0, "morphological": 0.8}.get(c["condition"], 0.7)
    rb, rc = seeded(c["brief_id"], "emb"), seeded(c["concept_id"], "emb")
    centre = [rb.gauss(0, 1) for _ in range(dim)]
    return [x + spread * rc.gauss(0, 1) for x in centre]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  THE EXPERIMENT DATA                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Experiment:
    def __init__(self, input_file: Path):
        data = load_json(input_file)
        self.meta = data["metadata"]
        self.briefs = {b["id"]: b for b in data["briefs"]}
        self.out = input_file.parent / OUTPUT_FOLDER
        self.cache = self.out / "judgements"
        self.concepts = []
        for s in data["sessions"]:
            for i, c in enumerate(s["concepts"], 1):
                self.concepts.append({
                    "concept_id": f"{s['session_id']}__c{i}", "session_id": s["session_id"],
                    "brief_id": s["brief_id"], "brief_type": self.briefs[s["brief_id"]].get("type", "brief"),
                    "condition": s["condition"], "run": s["run"],
                    "title": c["title"], "description": c["description"],
                    "word_count": c.get("word_count") or len(c["description"].split()),
                })
        self.by_id = {c["concept_id"]: c for c in self.concepts}
        self.conditions = list(self.meta["conditions"])
        self.baseline = self.meta.get("baseline_condition", self.conditions[0])

    def complete_briefs(self) -> dict:
        """Briefs whose sessions are all generated. Comparisons and trees need the full pool."""
        expected = len(self.conditions) * self.meta["n_runs"]
        groups = {}
        for c in self.concepts:
            groups.setdefault(c["brief_id"], []).append(c)
        done = {}
        for bid, g in groups.items():
            if len({c["session_id"] for c in g}) < expected:
                log(f"  {bid}: skipped, only {len({c['session_id'] for c in g})} of {expected} sessions generated")
            else:
                done[bid] = g
        return done

    def rank_file(self, criterion, group_id):
        return self.cache / "rankings" / criterion / f"{group_id}.json"

    def rate_file(self, cid, rep):
        return self.cache / "ratings" / f"{cid}__{rep}.json"

    def coding_file(self, cid):
        return self.cache / "coding" / f"{cid}.json"

    def grouping_file(self, bid):
        return self.cache / "grouping" / f"{bid}.json"

    def plan_file(self, bid):
        return self.cache / "group_plan" / f"{bid}.json"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: PAIR PLAN                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def make_group_plan(ex: Experiment, bid: str, concepts: list) -> list:
    """Each concept appears in GROUPS_PER_CONCEPT groups of about GROUP_SIZE concepts from the
    same brief, drawn at random across all conditions and runs, in random order. A share of
    groups is ranked again with the order reversed, to measure position bias."""
    ids = sorted(c["concept_id"] for c in concepts)
    path = ex.plan_file(bid)
    settings = {"group_size": GROUP_SIZE, "groups_per_concept": GROUPS_PER_CONCEPT,
                "reverse_check_share": REVERSE_CHECK_SHARE}
    if path.exists():
        old = load_json(path)
        if old["concept_ids"] == ids and old["settings"] == settings:
            return old["groups"]
    # One generator per round, so the plan for rounds 1..k is the same whatever
    # GROUPS_PER_CONCEPT is: raising it later only adds rounds, and every ranking
    # already made stays valid.
    groups, rev_list = [], []
    for rnd in range(1, GROUPS_PER_CONCEPT + 1):
        rng = random.Random(f"{RANDOM_SEED}-{bid}-round{rnd}")
        order = ids[:]
        rng.shuffle(order)
        chunks = [order[i:i + GROUP_SIZE] for i in range(0, len(order), GROUP_SIZE)]
        if len(chunks) > 1 and len(chunks[-1]) < 3:      # spread a small remainder over other groups
            for i, cid in enumerate(chunks.pop()):
                chunks[i].append(cid)
        this_round = [{"group_id": f"{bid}__r{rnd}g{k:02d}", "members": ch, "round": rnd, "kind": "main"}
                      for k, ch in enumerate(chunks, 1)]
        groups += this_round
        reversed_groups = rng.sample(this_round, round(REVERSE_CHECK_SHARE * len(this_round)))
        rev_list.extend(reversed_groups)
    for g in rev_list:
        groups.append({"group_id": g["group_id"] + "_rev", "members": g["members"][::-1], "round": g["round"],
                       "kind": "reverse", "reverse_of": g["group_id"]})
    save_json(path, {"brief_id": bid, "concept_ids": ids, "settings": settings, "groups": groups})
    return groups


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2-5: JUDGE CALLS AND EMBEDDINGS                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def rank_once(ex, judge, criterion, group):
    path = ex.rank_file(criterion, group["group_id"])
    if path.exists():
        return
    members = [ex.by_id[c] for c in group["members"]]
    n = len(members)
    base_prompt = rank_prompt(ex.briefs[members[0]["brief_id"]], members, criterion)
    feedback = {"text": ""}

    def call():
        prompt = base_prompt + feedback["text"]
        raw, usage = judge.ask(prompt, rank_schema(n), "group_ranking", RANK_EFFORT,
                               lambda: mock_rank(members, criterion, group["group_id"]))
        if sorted(raw["ranking"]) != list(LETTERS[:n]):
            feedback["text"] = (f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED: the ranking must contain each of the letters "
                                f"{', '.join(LETTERS[:n])} exactly once. Answer again in full.")
            raise ValueError(f"invalid ranking {raw['ranking']}")
        return raw, usage, prompt

    raw, usage, prompt = retry(call, f"rank {criterion} {group['group_id']}")
    ranked = [group["members"][LETTERS.index(lab)] for lab in raw["ranking"]]
    save_json(path, {"criterion": criterion, "brief_id": members[0]["brief_id"], **group, "ranked": ranked,
                     "model": "MOCK" if MOCK else JUDGE_MODEL, "effort": RANK_EFFORT,
                     "timestamp": now(), "usage": usage, "prompt": prompt, "answers": raw})


def rate_once(ex, judge, c, rep, effort, careful):
    path = ex.rate_file(c["concept_id"], rep)
    if path.exists():
        return
    prompt = rate_prompt(ex.briefs[c["brief_id"]], c, careful)
    raw, usage = retry(lambda: judge.ask(prompt, RATE_SCHEMA, "lifecycle_rating", effort, lambda: mock_rate(c, rep)),
                       f"rate {c['concept_id']} {rep}")
    save_json(path, {"concept_id": c["concept_id"], "repeat": rep, "effort": effort,
                     "model": "MOCK" if MOCK else JUDGE_MODEL, "timestamp": now(), "usage": usage,
                     "prompt": prompt, "answers": raw, "scores": {k: raw[k] for k in RATINGS}})


def code_once(ex, judge, c):
    path = ex.coding_file(c["concept_id"])
    if path.exists():
        return
    prompt = coding_prompt(ex.briefs[c["brief_id"]], c)
    raw, usage = retry(lambda: judge.ask(prompt, CODING_SCHEMA, "concept_coding", CODING_EFFORT, lambda: mock_code(c)),
                       f"code {c['concept_id']}")
    save_json(path, {"concept_id": c["concept_id"], "model": "MOCK" if MOCK else JUDGE_MODEL,
                     "timestamp": now(), "usage": usage, "prompt": prompt, "answers": raw})


def mock_tree(items):
    tree = {}
    for lab, c, code in items:
        tree.setdefault(code["physical_principle"], {}).setdefault(code["working_principle"], []).append(lab)
    return {"physical_principles": [{"name": p, "working_principles": [
        {"name": w, "concept_labels": ls} for w, ls in ws.items()]} for p, ws in tree.items()]}


def group_once(ex, judge, bid, concepts):
    path = ex.grouping_file(bid)
    ids = {c["concept_id"] for c in concepts}
    if path.exists() and set(load_json(path)["label_map"].values()) == ids:
        return
    items = [(c, load_json(ex.coding_file(c["concept_id"]))["answers"]) for c in concepts]
    random.Random(bid).shuffle(items)        # hide session and condition order
    labelled = [(f"K{i+1:03d}", c, code) for i, (c, code) in enumerate(items)]
    label_map = {lab: c["concept_id"] for lab, c, _ in labelled}
    prompt = grouping_prompt(ex.briefs[bid], labelled)

    def call():
        raw, usage = judge.ask(prompt, GROUPING_SCHEMA, "genealogy", GROUPING_EFFORT, lambda: mock_tree(labelled))
        seen = [lab for p in raw["physical_principles"] for w in p["working_principles"]
                for lab in w["concept_labels"]]
        if sorted(seen) != sorted(label_map):
            raise ValueError("the tree does not contain every concept exactly once")
        return raw, usage

    raw, usage = retry(call, f"grouping {bid}")
    assignment = {}
    for i, p in enumerate(raw["physical_principles"], 1):
        for j, w in enumerate(p["working_principles"], 1):
            for lab in w["concept_labels"]:
                assignment[label_map[lab]] = {"physical_id": f"P{i}", "physical_name": p["name"],
                                              "working_id": f"P{i}.W{j}", "working_name": w["name"]}
    save_json(path, {"brief_id": bid, "model": "MOCK" if MOCK else JUDGE_MODEL, "timestamp": now(),
                     "usage": usage, "label_map": label_map, "prompt": prompt,
                     "answers": raw, "assignment": assignment})


def embed_all(ex, judge):
    path = ex.cache / "embeddings.json"
    model = "MOCK" if MOCK else EMBEDDING_MODEL
    store = load_json(path) if path.exists() else {"model": model, "vectors": {}}
    if store["model"] != model:
        store = {"model": model, "vectors": {}}
    todo = [c for c in ex.concepts if c["concept_id"] not in store["vectors"]]
    if not todo:
        log("  embeddings: nothing to do")
        return
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        if MOCK:
            vecs = [mock_embedding(c) for c in batch]
        else:
            texts = [f"{c['title']}. {c['description']}" for c in batch]
            vecs = retry(lambda: judge.embed(texts), "embeddings")
        for c, v in zip(batch, vecs):
            store["vectors"][c["concept_id"]] = v
        save_json(path, store)
    log(f"  embeddings: {len(todo)} concepts embedded")


def run_parallel(tasks, label):
    if not tasks:
        log(f"{label}: nothing to do")
        return
    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(fn, *args) for fn, args in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                failures += 1
                log(f"{label} FAILED: {e}")
            if i % 100 == 0 or i == len(futs):
                log(f"{label}: {i}/{len(futs)}")
    if failures:
        log(f"{label}: {failures} failures. Run this file again to retry them.")


def run_judging(ex: Experiment):
    judge = Judge()
    briefs = ex.complete_briefs()
    cs = [c for g in briefs.values() for c in g]
    plans = {bid: make_group_plan(ex, bid, g) for bid, g in briefs.items()}

    rank_tasks = [(rank_once, (ex, judge, crit, g)) for crit in CRITERIA for plan in plans.values()
                  for g in plan if not ex.rank_file(crit, g["group_id"]).exists()]
    journal = SCOPE == "journal"
    rate_tasks = [] if not journal else [(rate_once, (ex, judge, c, f"r{r}", RATE_EFFORT, False))
                  for r in range(1, RATE_REPEATS + 1) for c in cs if not ex.rate_file(c["concept_id"], f"r{r}").exists()]
    code_tasks = [] if not journal else [(code_once, (ex, judge, c)) for c in cs
                                         if not ex.coding_file(c["concept_id"]).exists()]
    if journal:
        print(f"  Calls still to make: {len(rank_tasks)} group rankings, {len(rate_tasks)} ratings, "
              f"{len(code_tasks)} codings, up to {len(briefs)} trees, plus embeddings")
    else:
        print(f"  Scope: ICED. Calls still to make: {len(rank_tasks)} group rankings, plus embeddings "
              "(life-cycle ratings, coding and trees are left for the journal scope)")

    print("STEP 1: Embeddings (no judge)")
    embed_all(ex, judge)

    print(f"STEP 2: Ranking groups of {GROUP_SIZE} (novelty and usefulness, separate calls)")
    random.Random(RANDOM_SEED).shuffle(rank_tasks)
    run_parallel(rank_tasks, "  rankings")

    if not journal:
        return plans

    print("STEP 3: Life-cycle and specificity ratings")
    random.Random(RANDOM_SEED).shuffle(rate_tasks)
    run_parallel(rate_tasks, "  ratings")
    tasks = []
    for c in cs:
        reps = [ex.rate_file(c["concept_id"], f"r{r}") for r in range(1, RATE_REPEATS + 1)]
        if RATE_REPEATS < 2 or not all(p.exists() for p in reps) or ex.rate_file(c["concept_id"], "rescore").exists():
            continue
        res = [load_json(p)["scores"] for p in reps]
        if max(max(r[k] for r in res) - min(r[k] for r in res) for k in RATINGS) >= RESCORE_GAP:
            tasks.append((rate_once, (ex, judge, c, "rescore", RESCORE_EFFORT, True)))
    run_parallel(tasks, "  rescoring disagreements")

    print("STEP 4: Coding principles")
    run_parallel(code_tasks, "  coding")

    print("STEP 5: Principle tree for each brief")
    run_parallel([(group_once, (ex, judge, bid, g)) for bid, g in briefs.items()
                  if all(ex.coding_file(c["concept_id"]).exists() for c in g)], "  grouping")
    return plans


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICS                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fit_plackett_luce(items: list, rankings: list, prior: float = 0.5, iters: int = 3000) -> dict:
    """Plackett-Luce strengths by the MM algorithm (Hunter, 2004). A ranking is read as a
    sequence of choices: the top concept is chosen from all, the second from the rest, and
    so on. Each concept also plays `prior` won and `prior` lost games against a fixed
    average opponent, which keeps estimates finite for concepts that always came first or
    last. Returns log-strengths."""
    import numpy as np
    idx = {c: i for i, c in enumerate(items)}
    n = len(items)
    stages, winners = [], []
    for r in rankings:
        ids = [idx[c] for c in r]
        for t in range(len(ids) - 1):
            stages.append(ids[t:])
            winners.append(ids[t])
    if not stages:
        return dict(zip(items, np.zeros(n)))
    width = max(map(len, stages))
    M = np.array([st + [-1] * (width - len(st)) for st in stages])
    mask = M >= 0
    Mi = np.where(mask, M, 0)
    won = np.full(n, prior)
    np.add.at(won, np.array(winners), 1.0)
    p = np.ones(n)
    for _ in range(iters):
        tot = (p[Mi] * mask).sum(axis=1)
        denom = 2 * prior / (p + 1.0)
        np.add.at(denom, M[mask], np.broadcast_to((1.0 / tot)[:, None], M.shape)[mask])
        new = won / denom
        done = np.max(np.abs(np.log(new) - np.log(p))) < 1e-9
        p = new
        if done:
            break
    return dict(zip(items, np.log(p)))


def spearman(x, y) -> float:
    import pandas as pd
    d = pd.DataFrame({"x": list(x), "y": list(y)}).dropna()
    if len(d) < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return float("nan")
    return float(d["x"].rank().corr(d["y"].rank()))


def split_half(items, rankings, n_splits=20, seed=RANDOM_SEED) -> float:
    """Split the group rankings in two at random, score each half, correlate, and step up
    with Spearman-Brown. Averaged over n_splits random splits."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_splits):
        w = rankings[:]
        rng.shuffle(w)
        h1, h2 = w[: len(w) // 2], w[len(w) // 2:]
        s1, s2 = fit_plackett_luce(items, h1), fit_plackett_luce(items, h2)
        r = spearman([s1[i] for i in items], [s2[i] for i in items])
        if r == r:
            vals.append(2 * r / (1 + r) if r > -1 else float("nan"))
    return sum(vals) / len(vals) if vals else float("nan")


def weighted_kappa(a, b, lo, hi):
    import numpy as np
    k = hi - lo + 1
    a, b = np.asarray(a, int) - lo, np.asarray(b, int) - lo
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    return float(1 - (w * obs).sum() / (w * exp).sum()) if (w * exp).sum() else float("nan")


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    cats = set(a) | set(b)
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def holm(pvals):
    import numpy as np
    p = np.asarray(pvals, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mean_pairwise_distance(vectors) -> float:
    """Mean cosine distance between all pairs (0 = identical texts, larger = more diverse)."""
    import numpy as np
    v = np.asarray(vectors, float)
    if len(v) < 2:
        return float("nan")
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    iu = np.triu_indices(len(v), 1)
    return float(np.mean(1 - sim[iu]))


class Bootstrap:
    """Confidence intervals that respect the design: sessions are resampled with
    replacement within each brief and condition, so the four concepts from one
    session always move together."""

    def __init__(self, df, n=BOOTSTRAP, seed=RANDOM_SEED):
        import numpy as np
        self.np = np
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.cells = {}
        for (bid, cond, sid), g in df.groupby(["brief_id", "condition", "session_id"]):
            self.cells.setdefault((bid, cond), []).append(g)
        self.briefs = sorted(df["brief_id"].unique())

    def _values(self, groups, col, idx=None):
        np = self.np
        chosen = groups if idx is None else [groups[i] for i in idx]
        return np.concatenate([g[col].values for g in chosen]) if chosen else np.array([])

    def stat(self, col, method, base, fn, resample=False):
        np = self.np
        vals = []
        for bid in self.briefs:
            gm, gb = self.cells.get((bid, method), []), self.cells.get((bid, base), [])
            if not gm or not gb:
                continue
            im = self.rng.integers(0, len(gm), len(gm)) if resample else None
            ib = self.rng.integers(0, len(gb), len(gb)) if resample else None
            x, y = self._values(gm, col, im), self._values(gb, col, ib)
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            if len(x) and len(y):
                vals.append(fn(x, y))
        return float(np.mean(vals)) if vals else float("nan")

    def ci(self, col, method, base, fn):
        np = self.np
        point = self.stat(col, method, base, fn)
        reps = np.array([self.stat(col, method, base, fn, resample=True) for _ in range(self.n)])
        reps = reps[~np.isnan(reps)]
        if not len(reps):
            return point, float("nan"), float("nan")
        return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))

    def mean_ci(self, col, cond):
        """Mean of one condition with a session-bootstrap interval."""
        np = self.np

        def one(resample):
            vals = []
            for bid in self.briefs:
                g = self.cells.get((bid, cond), [])
                if g:
                    i = self.rng.integers(0, len(g), len(g)) if resample else None
                    x = self._values(g, col, i)
                    vals.append(np.nanmean(x))
            return float(np.mean(vals)) if vals else float("nan")
        reps = np.array([one(True) for _ in range(self.n)])
        return one(False), float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))


def prob_superiority(x, y):
    """Probability that a random concept from x outranks a random concept from y (ties count half)."""
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def mean_diff(x, y):
    return float(x.mean() - y.mean())


def md_table(df, index_name="condition"):
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(map(str, cols)) + " |",
             "|---|" + "|".join(":---:" for _ in cols) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def fmt_ci(p, lo, hi, pct=False, digits=2):
    if p != p:
        return "n/a"
    if pct:
        return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"
    return f"{p:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def verdict(lo, hi, null):
    if lo != lo:
        return "n/a"
    if lo > null:
        return "higher"
    if hi < null:
        return "lower"
    return "no clear difference"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: ANALYSIS AND REPORT                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_analysis(ex: Experiment, plans: dict):
    import numpy as np
    import pandas as pd
    try:
        from scipy import stats
    except ImportError:
        stats = None
        print("  scipy not installed: pip install scipy (some tests skipped)")
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        smf = None
        print("  statsmodels not installed: pip install statsmodels (regression checks skipped)")

    print("STEP 6: Analysis")
    out = ex.out / f"analysis_{SCOPE}"
    out.mkdir(parents=True, exist_ok=True)
    base = ex.baseline
    stage_cols = [f"sus_{s}" for s in LIFECYCLE_STAGES]
    df = pd.DataFrame([c for c in ex.concepts if c["brief_id"] in plans])
    if df.empty:
        print("  Nothing to analyse yet.")
        return
    order = [c for c in ex.conditions if c in set(df["condition"])]
    methods = [c for c in order if c != base]

    # ---- Group rankings -> Plackett-Luce scores per brief, standardised within brief
    ranks = [load_json(p) for p in (ex.cache / "rankings").glob("*/*.json")]
    planned = {g["group_id"] for plan in plans.values() for g in plan}
    ranks = [d for d in ranks if d["brief_id"] in plans and d["group_id"] in planned]
    rank_df = pd.DataFrame([{"criterion": d["criterion"], "brief_id": d["brief_id"], "group_id": d["group_id"],
                             "kind": d["kind"], "reverse_of": d.get("reverse_of"), "members": d["members"],
                             "ranked": d["ranked"]} for d in ranks])
    pairs = []      # every ranking implies a winner for each pair in its group
    for d in ranks:
        if d["kind"] != "main":
            continue
        pos = {c: i for i, c in enumerate(d["members"])}
        for i, w in enumerate(d["ranked"]):
            for lo in d["ranked"][i + 1:]:
                a, b = sorted((w, lo), key=pos.get)
                pairs.append({"criterion": d["criterion"], "brief_id": d["brief_id"], "group_id": d["group_id"],
                              "kind": "main", "a": a, "b": b, "winner": w, "loser": lo})
    cmp_df = pd.DataFrame(pairs)
    if not rank_df.empty:
        rank_df.assign(members=rank_df["members"].map(" > ".join).str.replace(" > ", " | "),
                       ranked=rank_df["ranked"].map(" > ".join)).to_csv(out / "rankings_all.csv", index=False)
        cmp_df.to_csv(out / "implied_pairs.csv", index=False)
    reliab = []
    for crit in CRITERIA:
        scores = {}
        for bid in plans:
            sub = rank_df[(rank_df["criterion"] == crit) & (rank_df["brief_id"] == bid) & (rank_df["kind"] == "main")] \
                if not rank_df.empty else rank_df
            if sub.empty:
                continue
            items = sorted(df.loc[df["brief_id"] == bid, "concept_id"])
            rankings = list(sub["ranked"])
            s = fit_plackett_luce(items, rankings)
            v = np.array(list(s.values()))
            z = (v - v.mean()) / (v.std() if v.std() > 0 else 1)
            scores.update(dict(zip(s.keys(), z)))
            n_cmp = sum(len(r) * (len(r) - 1) // 2 for r in rankings)
            reliab.append({"criterion": crit, "brief_id": bid, "n_rankings": len(sub),
                           "comparisons_per_concept": round(2 * n_cmp / len(items), 1),
                           "split_half": round(split_half(items, rankings), 3)})
        df[crit] = df["concept_id"].map(scores)

    # ---- Ratings: median of passes (with the careful rescore where it exists)
    journal = SCOPE == "journal"
    rows = [{"concept_id": d["concept_id"], "repeat": d["repeat"], **d["scores"]}
            for d in map(load_json, (ex.cache / "ratings").glob("*.json"))] if journal else []
    long = pd.DataFrame(rows)
    if not long.empty:
        long.to_csv(out / "ratings_all_passes.csv", index=False)
        df = df.merge(long.groupby("concept_id")[RATINGS].median().reset_index(), on="concept_id", how="left")
    else:
        for k in RATINGS:
            df[k] = np.nan

    # ---- Embeddings: distinctiveness of each concept and diversity of each pool
    emb_path = ex.cache / "embeddings.json"
    vecs = load_json(emb_path)["vectors"] if emb_path.exists() else {}
    has_emb = all(cid in vecs for cid in df["concept_id"])
    if has_emb:
        dist = {}
        for bid, g in df.groupby("brief_id"):
            v = np.asarray([vecs[c] for c in g["concept_id"]], float)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            sim = v @ v.T
            np.fill_diagonal(sim, np.nan)
            dist.update(dict(zip(g["concept_id"], 1 - np.nanmean(sim, axis=1))))
        df["distinctiveness"] = df["concept_id"].map(dist)

    # ---- Principle tree
    groups = [{"concept_id": cid, **a} for d in map(load_json, (ex.cache / "grouping").glob("*.json"))
              for cid, a in d["assignment"].items()] if journal else []
    codes = [{"concept_id": d["concept_id"], **d["answers"]}
             for d in map(load_json, (ex.cache / "coding").glob("*.json"))] if journal else []
    if codes:
        df = df.merge(pd.DataFrame(codes), on="concept_id", how="left")
    has_tree = bool(groups)
    if has_tree:
        df = df.merge(pd.DataFrame(groups), on="concept_id", how="left")
    df.to_csv(out / "concepts.csv", index=False)

    # ---- Session table
    agg = {k: "mean" for k in ["novelty", "usefulness", "word_count", *RATINGS] if k in df}
    sessions = df.groupby(["session_id", "brief_id", "brief_type", "condition", "run"]).agg(agg).reset_index()
    if has_emb:
        sessions["diversity_within_session"] = sessions["session_id"].map(
            {sid: mean_pairwise_distance([vecs[c] for c in g["concept_id"]]) for sid, g in df.groupby("session_id")})
    sessions.to_csv(out / "sessions.csv", index=False)
    n_sess = sessions.groupby("condition").size().min()
    pilot = n_sess < 10

    # ---- Pool table: one row per brief x condition (all runs pooled)
    pools = []
    for (bid, btype, cond), g in df.groupby(["brief_id", "brief_type", "condition"]):
        row = {"brief_id": bid, "brief_type": btype, "condition": cond, "n_concepts": len(g)}
        if has_emb:
            row["diversity_pooled"] = mean_pairwise_distance([vecs[c] for c in g["concept_id"]])
            row["diversity_within_session"] = sessions.loc[(sessions["brief_id"] == bid) & (sessions["condition"] == cond),
                                                           "diversity_within_session"].mean()
        if has_tree and g["working_id"].notna().all():
            n = len(g)
            raw = (SHAH_WEIGHTS["physical"] * g["physical_id"].nunique()
                   + SHAH_WEIGHTS["working"] * g["working_id"].nunique()) / n
            vmin, vmax = sum(SHAH_WEIGHTS.values()) / n, sum(SHAH_WEIGHTS.values())
            others = set(df.loc[(df["brief_id"] == bid) & (df["condition"] != cond), "working_id"])
            row.update({"n_physical": g["physical_id"].nunique(), "n_working": g["working_id"].nunique(),
                        "variety_shah": (raw - vmin) / (vmax - vmin),
                        "unique_working_principles": len(set(g["working_id"]) - others)})
        pools.append(row)
    pools = pd.DataFrame(pools)
    pools.to_csv(out / "pools_by_brief_condition.csv", index=False)

    boot = Bootstrap(df)
    insights = []
    rep = [f"# Evaluation report ({'ICED paper scope' if not journal else 'full journal scope'})", "",
           f"**Judge model:** {'MOCK' if MOCK else JUDGE_MODEL}   **Embedding model:** {'MOCK' if MOCK else EMBEDDING_MODEL}  ",
           f"**Generation model:** {ex.meta.get('generation_model')}  ",
           f"**Evaluated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
           f"**Concepts:** {len(df)}   **Sessions:** {len(sessions)}   **Briefs:** {df['brief_id'].nunique()}   "
           f"**Baseline:** {base}", ""]
    if pilot:
        rep += [f"> **Pilot-sized data** ({n_sess} sessions per condition). Use this report to check that the "
                "measures behave (see Checks), not to draw conclusions about the methods.", ""]
    rep += ["## Key insights", "", "INSIGHTS_PLACEHOLDER", ""]

    # ---- R1. Better? Novelty and usefulness
    r1_rows, r1_csv = {}, []
    for crit in CRITERIA:
        if crit not in df or df[crit].isna().all():
            continue
        for cond in order:
            m, lo, hi = boot.mean_ci(crit, cond)
            r1_rows.setdefault(cond, {})[f"{crit} (z)"] = f"{m:+.2f}"
            row = {"criterion": crit, "condition": cond, "mean_z": m, "mean_lo": lo, "mean_hi": hi}
            if cond != base:
                ps, plo, phi = boot.ci(crit, cond, base, prob_superiority)
                r1_rows[cond][f"{crit}: P(beats {base})"] = fmt_ci(ps, plo, phi, pct=True)
                row.update({"p_superiority": ps, "ps_lo": plo, "ps_hi": phi, "verdict": verdict(plo, phi, 0.5)})
                # direct head-to-head comparisons, as a transparent check on the model
                if not cmp_df.empty:
                    cm = cmp_df[(cmp_df["criterion"] == crit) & (cmp_df["kind"] == "main")]
                    cond_of = df.set_index("concept_id")["condition"]
                    ca, cb = cm["a"].map(cond_of), cm["b"].map(cond_of)
                    h2h = cm[((ca == cond) & (cb == base)) | ((ca == base) & (cb == cond))]
                    row.update({"head_to_head_n": len(h2h),
                                "head_to_head_win_rate": (h2h["winner"].map(cond_of) == cond).mean() if len(h2h) else np.nan})
            else:
                r1_rows[cond][f"{crit}: P(beats {base})"] = "(baseline)"
            r1_csv.append(row)
    r1 = pd.DataFrame(r1_rows).T.reindex(order)
    r1_df = pd.DataFrame(r1_csv)
    r1_df.to_csv(out / "R1_novelty_usefulness.csv", index=False)
    rep += ["## R1. Better? Novelty and usefulness", "",
            f"Scores come from rankings of groups of {GROUP_SIZE} concepts within each brief (Plackett-Luce), standardised within each "
            "brief, so 0 is the average concept for that brief and 1 is one standard deviation above it. "
            f"**P(beats {base})** is the probability that a concept from the method is judged better than a concept "
            f"from the {base} baseline for the same brief: 50% means no difference. Brackets are 95% intervals from "
            "resampling sessions.", "", md_table(r1), ""]
    for crit in CRITERIA:
        sub = r1_df[(r1_df["criterion"] == crit) & (r1_df["condition"] != base)] if not r1_df.empty else r1_df
        if sub.empty:
            continue
        sub = sub.sort_values("p_superiority", ascending=False)
        up = [f"{r.condition} ({100 * r.p_superiority:.0f}%)" for r in sub.itertuples() if r.verdict == "higher"]
        down = [f"{r.condition} ({100 * r.p_superiority:.0f}%)" for r in sub.itertuples() if r.verdict == "lower"]
        same = [r.condition for r in sub.itertuples() if r.verdict == "no clear difference"]
        parts = []
        if up:
            parts.append(f"more {CRITERIA[crit]['adjective']} than the baseline: " + ", ".join(up))
        if down:
            parts.append(f"less {CRITERIA[crit]['adjective']}: " + ", ".join(down))
        if same:
            parts.append("no clear difference: " + ", ".join(same))
        insights.append(f"**{crit.capitalize()}** (chance that a method's concept beats a baseline concept): "
                        + "; ".join(parts) + ".")
    if smf is not None and not pilot and "novelty" in sessions:
        rep += [f"**Regression check** (session means, brief as a fixed effect, Holm-adjusted p for each method vs {base}):", ""]
        for crit in CRITERIA:
            fit = smf.ols(f"{crit} ~ C(condition, Treatment('{base}')) + C(brief_id)", data=sessions).fit()
            names = [i for i in fit.params.index if "condition" in i]
            padj = holm([fit.pvalues[i] for i in names])
            rep.append(f"- {crit}: " + ", ".join(f"{i.split('T.')[-1].rstrip(']')} {fit.params[i]:+.2f} (p = {p:.3f})"
                                                  for i, p in zip(names, padj)))
        rep.append("")
    if "novelty" in df and "usefulness" in df:
        rho_all = spearman(df["novelty"], df["usefulness"])
        per = {c: spearman(g["novelty"], g["usefulness"]) for c, g in df.groupby("condition")}
        rep += [f"**Novelty-usefulness trade-off:** Spearman rho between a concept's novelty and usefulness = "
                f"{rho_all:.2f} (by condition: " + ", ".join(f"{c} {per[c]:.2f}" for c in order) + "). "
                "Negative means more novel concepts tend to be judged less useful.", ""]
        insights.append(f"**Trade-off.** Across all concepts, novelty and usefulness correlate at rho = {rho_all:.2f}"
                        + (" (more novel concepts tend to be judged less useful)." if rho_all < -0.1 else
                           " (more novel concepts also tend to be judged more useful)." if rho_all > 0.1 else
                           " (largely independent)."))

    # ---- R2. Wider? Spread
    spread_cols = [c for c in ["diversity_pooled", "diversity_within_session", "variety_shah", "unique_working_principles"]
                   if c in pools]
    if spread_cols:
        r2 = pools.groupby("condition")[spread_cols].mean().reindex(order)
        r2_out = r2.copy().astype(object)
        tests = []
        for col in spread_cols:
            piv = pools.pivot(index="brief_id", columns="condition", values=col)
            for cond in methods:
                if cond not in piv or base not in piv:
                    continue
                d = (piv[cond] - piv[base]).dropna()
                p = float("nan")
                if stats is not None and len(d) >= 5 and (d != 0).any():
                    p = float(stats.wilcoxon(d).pvalue)
                tests.append({"measure": col, "method": cond, "mean_diff_vs_baseline": d.mean(),
                              "briefs_higher": int((d > 0).sum()), "briefs": len(d), "wilcoxon_p": p})
        tests = pd.DataFrame(tests)
        tests.to_csv(out / "R2_spread_vs_baseline.csv", index=False)
        for col in spread_cols:
            for cond in order:
                v = r2.loc[cond, col]
                txt = f"{v:.3f}" if col != "unique_working_principles" else f"{v:.1f}"
                t = tests[(tests["measure"] == col) & (tests["method"] == cond)] if not tests.empty else tests
                if len(t):
                    t = t.iloc[0]
                    txt += f" ({t.briefs_higher}/{t.briefs})"
                r2_out.loc[cond, col] = txt
        r2.to_csv(out / "R2_spread.csv")
        n_pool = int(pools["n_concepts"].median())
        rep += ["## R2. Wider? Spread of ideas", "",
                f"Each value is the mean over briefs; in brackets, the number of briefs where the method is higher "
                f"than the {base} baseline.", "",
                f"- **diversity_pooled**: mean semantic distance between all {n_pool} concepts a method produced for a "
                "brief over all runs (embeddings; no judge). This is the key measure of whether a method stops the model "
                "returning to the same ideas.",
                "- **diversity_within_session**: the same, within one session of 4 concepts. If pooled diversity is "
                "much lower than this, the method gives varied ideas in one session but repeats them across runs.",
                *(["- **variety_shah**: Shah et al. (2003) variety from the principle tree, 0 = all pooled concepts share one "
                   "principle, 1 = all differ.",
                   "- **unique_working_principles**: working principles that only this method reached for that brief."]
                  if "variety_shah" in spread_cols else []), "",
                md_table(r2_out), ""]
        if "diversity_pooled" in spread_cols and not tests.empty:
            t = tests[tests["measure"] == "diversity_pooled"].sort_values("mean_diff_vs_baseline", ascending=False)
            nb = int(t["briefs"].max())
            wider = [f"{r.method} ({r.briefs_higher}/{r.briefs} briefs)" for r in t.itertuples()
                     if r.briefs_higher == r.briefs and r.briefs >= 2]
            narrower = [f"{r.method} ({r.briefs - r.briefs_higher}/{r.briefs})" for r in t.itertuples()
                        if r.briefs_higher == 0 and r.briefs >= 2]
            msg = "**Spread.** Pooled over runs, "
            msg += ("the concepts were more diverse than the baseline's in every brief for " + ", ".join(wider)) if wider \
                else f"no method was more diverse than the baseline in all {nb} briefs"
            if narrower:
                msg += "; less diverse in every brief for " + ", ".join(narrower)
            insights.append(msg + ".")

    # ---- R3. Where? Life-cycle profile
    if df[stage_cols].notna().any().any():
        prof = df.groupby("condition")[stage_cols].mean().reindex(order)
        prof.columns = LIFECYCLE_STAGES
        improving = df.assign(**{s: df[s] >= 1 for s in stage_cols}).groupby("condition")[stage_cols].mean().reindex(order)
        improving.columns = LIFECYCLE_STAGES
        diffs, diff_rows = {}, []
        for cond in methods:
            for s, col in zip(LIFECYCLE_STAGES, stage_cols):
                d, lo, hi = boot.ci(col, cond, base, mean_diff)
                diffs.setdefault(cond, {})[s] = fmt_ci(d, lo, hi)
                diff_rows.append({"method": cond, "stage": s, "diff": d, "lo": lo, "hi": hi, "verdict": verdict(lo, hi, 0)})
        diff_df = pd.DataFrame(diff_rows)
        prof.round(2).to_csv(out / "R3_lifecycle_profile.csv")
        improving.round(3).to_csv(out / "R3_lifecycle_share_improving.csv")
        diff_df.to_csv(out / "R3_lifecycle_vs_baseline.csv", index=False)
        df.groupby(["brief_id", "condition"])[stage_cols].mean().round(2).to_csv(out / "R3_lifecycle_by_brief.csv")
        rep += ["## R3. Where? Life-cycle profile", "",
                "Mean rating against existing products, -1 (worse) to 2 (substantial improvement) (Han et al., 2021):", "",
                md_table(prof.round(2)), "",
                "Share of concepts with at least a moderate improvement (rating 1 or 2):", "",
                md_table((100 * improving).round(0).astype(int).astype(str) + "%"), "",
                f"Difference from the {base} baseline [95% interval]:", "",
                md_table(pd.DataFrame(diffs).T.reindex(methods)), ""]
        sig = diff_df[diff_df["verdict"] != "no clear difference"].copy()
        if len(sig):
            sig["abs"] = sig["diff"].abs()
            sig = sig.sort_values("abs", ascending=False).head(4)
            insights.append("**Life cycle.** Clearest shifts from the baseline: " + "; ".join(
                f"{r.method} {'more' if r.diff > 0 else 'less'} improvement at {r.stage.replace('_', ' ')} ({r.diff:+.2f})"
                for r in sig.itertuples()) + ".")
        else:
            insights.append("**Life cycle.** No method shifted the life-cycle profile clearly away from the baseline.")
        weak = prof.mean().sort_values().index[0]
        insights.append(f"**Life-cycle blind spot.** Across all conditions, the weakest stage is "
                        f"{weak.replace('_', ' ')} (mean {prof[weak].mean():+.2f}).")

    # ---- R4. Brief type
    types = sorted(df["brief_type"].unique())
    if journal and len(types) > 1:
        rows = {}
        for crit in CRITERIA:
            if crit not in df:
                continue
            for bt in types:
                bsub = Bootstrap(df[df["brief_type"] == bt], n=min(BOOTSTRAP, 1000))
                for cond in methods:
                    ps, lo, hi = bsub.ci(crit, cond, base, prob_superiority)
                    rows.setdefault(cond, {})[f"{crit} ({bt})"] = fmt_ci(ps, lo, hi, pct=True)
        r4 = pd.DataFrame(rows).T.reindex(methods)
        r4.to_csv(out / "R4_by_brief_type.csv")
        rep += ["## R4. Which method for which design intent?", "",
                f"P(beats {base}) separately for adaptive briefs (redesign of an existing product) and original briefs "
                "(a new product), after Pahl et al. (2007).", "", md_table(r4), ""]
        if spread_cols and "diversity_pooled" in pools:
            t = pools.pivot_table(index="condition", columns="brief_type", values="diversity_pooled").reindex(order).round(3)
            rep += ["Pooled semantic diversity by brief type:", "", md_table(t), ""]
        if smf is not None and not pilot:
            rep.append("**Interaction test** (session means; a small p means methods rank differently on the two brief types):")
            rep.append("")
            import statsmodels.api as sm
            for crit in CRITERIA:
                fit = smf.ols(f"{crit} ~ C(condition) * C(brief_type)", data=sessions).fit()
                row = sm.stats.anova_lm(fit, typ=2).loc["C(condition):C(brief_type)"]
                rep.append(f"- {crit}: F = {row['F']:.2f}, p = {row['PR(>F)']:.3f}")
            rep.append("")

    # ---- Checks
    rep += ["## Checks (for the Method section)", ""]
    if reliab:
        rel = pd.DataFrame(reliab)
        rel.to_csv(out / "C1_split_half_reliability.csv", index=False)
        main_r = rank_df[rank_df["kind"] == "main"]
        # position bias: where in the list was the concept that came out on top?
        first_top = float(np.mean([r.ranked[0] == r.members[0] for r in main_r.itertuples()]))
        last_top = float(np.mean([r.ranked[0] == r.members[-1] for r in main_r.itertuples()]))
        expected = float(np.mean([1 / len(r.members) for r in main_r.itertuples()]))
        pos_rank = {}
        for r in main_r.itertuples():
            for i, c in enumerate(r.members):
                pos_rank.setdefault(i + 1, []).append(r.ranked.index(c) + 1)
        pos_txt = ", ".join(f"position {k}: {np.mean(v):.2f}" for k, v in sorted(pos_rank.items()) if len(v) >= 5)
        # reversed-order consistency: same group ranked again with the order reversed
        rev_rho, rev_pairs = [], []
        for r in rank_df[rank_df["kind"] == "reverse"].itertuples():
            orig = main_r[(main_r["criterion"] == r.criterion) & (main_r["group_id"] == r.reverse_of)]
            if not len(orig):
                continue
            o = orig["ranked"].iloc[0]
            ro = {c: i for i, c in enumerate(o)}
            rr = {c: i for i, c in enumerate(r.ranked)}
            rev_rho.append(spearman([ro[c] for c in o], [rr[c] for c in o]))
            rev_pairs += [(ro[x] < ro[y]) == (rr[x] < rr[y]) for i, x in enumerate(o) for y in o[i + 1:]]
        rev_mean = float(np.nanmean(rev_rho)) if rev_rho else float("nan")
        rev_pair = float(np.mean(rev_pairs)) if rev_pairs else float("nan")
        rel_mean = rel.groupby("criterion")["split_half"].mean()
        rep += ["**C1. Ranking reliability.** Split-half reliability of the concept scores, splitting the group "
                "rankings in two (Spearman-Brown; about 0.7 or more is the usual standard for comparative judgement, "
                "Verhavert et al., 2019):", "",
                md_table(rel.set_index("criterion"), "criterion"), "",
                f"- Position bias: the concept shown first was ranked top in {100 * first_top:.0f}% of groups, the one shown "
                f"last in {100 * last_top:.0f}% ({100 * expected:.0f}% = no bias). Mean rank by position shown ({pos_txt}; "
                "equal values = no bias).",
                f"- Reversed-order consistency: {len(rev_rho)} groups were ranked again with the order reversed. Rankings "
                f"correlated at rho = {rev_mean:.2f}, and {100 * rev_pair:.0f}% of pairs kept the same order. "
                "The main analysis uses the first ranking only.", ""]
        insights.append("**Judge reliability.** Split-half reliability "
                        + ", ".join(f"{k} {v:.2f}" for k, v in rel_mean.items())
                        + f"; reversed-order consistency rho = {rev_mean:.2f}.")
        if (rel["split_half"] < 0.7).any():
            low = rel[rel["split_half"] < 0.7]
            msg = ("Split-half reliability is below 0.7 for " + ", ".join(f"{r.criterion} {r.brief_id}" for r in low.itertuples())
                   + f". Raise GROUPS_PER_CONCEPT (now {GROUPS_PER_CONCEPT}) and run again: rankings already made are kept, "
                   "only the new rounds are added.")
            rep += [f"> {msg}", ""]
            insights.append("**Action.** " + msg)
    if not long.empty and {"r1", "r2"} <= set(long["repeat"]):
        wide = long[long["repeat"].isin(["r1", "r2"])].pivot_table(index="concept_id", columns="repeat", values=RATINGS)
        rep += ["**C2. Life-cycle rating reliability**, pass 1 vs pass 2 (quadratic weighted kappa; > 0.6 substantial):", ""]
        for k in RATINGS:
            pair = wide[k].dropna()
            lo, hi = RATING_RANGES[k]
            rep.append(f"- {k}: kappa = {weighted_kappa(pair['r1'], pair['r2'], lo, hi):.2f}, "
                       f"exact agreement = {(pair['r1'] == pair['r2']).mean():.2f}")
        rep += [f"- concepts rescored after disagreement: {(long['repeat'] == 'rescore').sum()} of {len(df)}", ""]
    has_spec = "specificity" in df and df["specificity"].notna().any()
    rep += ["**C3. Length and detail bias.** Correlation of each score with word count"
            + (" and specificity" if has_spec else "")
            + " (close to 0 means the judge is not rewarding longer or more detailed descriptions):", ""]
    for crit in CRITERIA:
        if crit in df:
            rep.append(f"- {crit}: rho with word count = {spearman(df[crit], df['word_count']):.2f}"
                       + (f", with specificity = {spearman(df[crit], df['specificity']):.2f}" if has_spec else ""))
    wc = df.groupby("condition")["word_count"].mean().reindex(order).round(1)
    rep += ["- mean words per concept: " + ", ".join(f"{c} {v}" for c, v in wc.items()), ""]
    if smf is not None and not pilot:
        control = "specificity" if has_spec else "word_count"
        rep.append(f"- method effects vs {base} after adjusting for {control.replace('_', ' ')} "
                   "(session means, brief fixed effect):")
        for crit in CRITERIA:
            fit = smf.ols(f"{crit} ~ C(condition, Treatment('{base}')) + C(brief_id) + {control}", data=sessions).fit()
            rep.append(f"  - {crit}: " + ", ".join(f"{i.split('T.')[-1].rstrip(']')} {v:+.2f} (p = {fit.pvalues[i]:.3f})"
                                                    for i, v in fit.params.filter(like="condition").items()))
        rep.append("")
    if journal and has_emb and "novelty" in df:
        rep += [f"**Two views of novelty.** The judge rates novelty against existing products; distinctiveness "
                f"(embeddings) measures how far a concept is from the other concepts the model produced for the same "
                f"brief. They are different things and need not agree: rho = {spearman(df['novelty'], df['distinctiveness']):.2f}. "
                "A low value means ideas that look new to the market are common in the model's output.", ""]

    # ---- C4 human check
    human_section(ex, df, cmp_df, rep, insights, pd)

    # ---- Methodology note
    rep += ["## Methodology note (for the paper)", "",
            f"Concepts were evaluated by {JUDGE_MODEL}, a model from a different family to the generator "
            f"({ex.meta.get('generation_model')}), so that no model evaluated its own output. The judge saw the brief and "
            "the concepts' titles and descriptions only, never the condition or the method workings. Novelty and "
            "usefulness were assessed by comparative judgement (Thurstone, 1927; Pollitt, 2012), which avoids the ceiling "
            "effects of absolute ratings and suits LLM judges (Liusie et al., 2024). The judge was shown groups of "
            f"{GROUP_SIZE} concepts for the same brief and ranked them from most to least novel, and in a separate call "
            f"from most to least useful. Each concept appeared in {GROUPS_PER_CONCEPT} groups drawn at random from all "
            "conditions and runs for its brief, in random order; "
            f"{int(100 * REVERSE_CHECK_SHARE)}% of groups were ranked again in reverse order to measure position bias "
            "(Zheng et al., 2023). Rankings were converted to a score per concept with the Plackett-Luce model "
            "(Luce, 1959; Plackett, 1975; Hunter, 2004), standardised within each brief, and reliability was estimated by "
            "split-half correlation (Verhavert et al., 2019). Method effects are reported as the probability that a method's concept "
            "outranks a baseline concept for the same brief, with 95% intervals from resampling sessions. Spread was "
            "measured without the judge, as the mean cosine distance between text embeddings of the concepts a method "
            f"produced for a brief over all runs ({EMBEDDING_MODEL}; Doshi & Hauser, 2024; Meincke et al., 2024)"
            + (", and as Shah et al.'s (2003) variety from a genealogy tree of physical and working principles built once "
               "per brief with condition labels removed. Each concept was also rated against existing products at the four "
               "life-cycle stages of Han et al. (2021), in two independent passes with the median taken and a third careful "
               "pass where they differed by two or more points." if journal else "."), ""]

    if pilot:
        insights.insert(0, "**Pilot data: these lines check that the pipeline works. With so few sessions the intervals "
                           "are unreliable, so do not read them as findings.**")
    text = "\n".join(rep).replace("INSIGHTS_PLACEHOLDER", "\n".join(f"- {s}" for s in insights) or "- (none yet)")
    (ex.out / f"evaluation_report_{SCOPE}.md").write_text(text)
    save_json(ex.out / f"evaluation_scores_{SCOPE}.json", {
        "metadata": {"scope": SCOPE, "judge_model": "MOCK" if MOCK else JUDGE_MODEL, "embedding_model": EMBEDDING_MODEL,
                     "group_size": GROUP_SIZE, "groups_per_concept": GROUPS_PER_CONCEPT,
                     "reverse_check_share": REVERSE_CHECK_SHARE,
                     "rate_repeats": RATE_REPEATS, "rescore_gap": RESCORE_GAP, "shah_weights": SHAH_WEIGHTS,
                     "bootstrap": BOOTSTRAP, "evaluated": now(), "generation": ex.meta},
        "concepts": json.loads(df.to_json(orient="records")),
        "sessions": json.loads(sessions.to_json(orient="records")),
        "pools": json.loads(pools.to_json(orient="records")),
    })
    make_figures(ex, df, pools, r1_df, order, base)

    print()
    print("=" * 60)
    print("  KEY INSIGHTS")
    print("=" * 60)
    for s in insights:
        print("  - " + s.replace("**", ""))
    print("=" * 60)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  OPTIONAL HUMAN CHECK (C4)                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

HUMAN_COLUMNS = ["item", "brief", "concept_A_title", "concept_A_description",
                 "concept_B_title", "concept_B_description", "more_novel (A/B)", "more_useful (A/B)"]


def human_section(ex, df, cmp_df, rep, insights, pd):
    folder = ex.out / "human_check"
    template = folder / "human_check_TEMPLATE.csv"
    key_file = folder / "key.json"
    if cmp_df.empty:
        return
    main = cmp_df[cmp_df["kind"] == "main"]
    both = main.groupby(["a", "b"])["criterion"].nunique()
    both = both[both == len(CRITERIA)].reset_index()
    if not key_file.exists() and len(both):
        rng = random.Random(RANDOM_SEED)
        both["brief_id"] = both["a"].map(df.set_index("concept_id")["brief_id"])
        by_brief = {bid: g.sample(frac=1, random_state=RANDOM_SEED).to_dict("records") for bid, g in both.groupby("brief_id")}
        chosen = []
        while len(chosen) < HUMAN_CHECK_PAIRS and any(by_brief.values()):
            for bid in sorted(by_brief):
                if by_brief[bid] and len(chosen) < HUMAN_CHECK_PAIRS:
                    chosen.append(by_brief[bid].pop())
        rng.shuffle(chosen)
        rows, key = [], []
        for i, p in enumerate(chosen, 1):
            a, b = ex.by_id[p["a"]], ex.by_id[p["b"]]
            brief = ex.briefs[a["brief_id"]]
            rows.append({"item": i, "brief": (f"{brief.get('product')}: " if brief.get("product") else "") + brief["goal"],
                         "concept_A_title": a["title"], "concept_A_description": a["description"],
                         "concept_B_title": b["title"], "concept_B_description": b["description"],
                         "more_novel (A/B)": "", "more_useful (A/B)": ""})
            key.append({"item": i, "a": p["a"], "b": p["b"]})
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=HUMAN_COLUMNS).to_csv(template, index=False, encoding="utf-8-sig")
        save_json(key_file, {"note": "Do not show this file to raters.", "pairs": key})
        (folder / "INSTRUCTIONS.txt").write_text(
            "Optional expert check (about 30 minutes per person).\n\n"
            "1. Copy human_check_TEMPLATE.csv to human_check_<yourname>.csv in this folder.\n"
            "2. For each row, read the brief and the two concepts, then write A or B in both answer columns:\n"
            "   more_novel: which concept is more different from the products and solutions that exist for this\n"
            "     purpose today (new working principle, structure or way of use). Ignore whether it would work.\n"
            "   more_useful: which concept would better achieve the goal of the brief, including its condition,\n"
            "     while being feasible and practical. Ignore novelty and writing quality.\n"
            "   If they are very close, pick the slightly better one. Work alone; do not discuss answers.\n"
            "3. Run the evaluation script again. It adds the agreement with the LLM judge to the report (C4).\n")
        log(f"  Human check sheet written to {folder}")
    if not key_file.exists():
        return
    key = {p["item"]: p for p in load_json(key_file)["pairs"]}
    files = [f for f in [*folder.glob("human_check_*.csv"), *folder.glob("human_check_*.xlsx")] if f.name != template.name]
    col = {"novelty": "more_novel (A/B)", "usefulness": "more_useful (A/B)"}
    results, raters = [], {}
    for f in files:
        if f.suffix == ".xlsx":     # rater workbook (sheet "Ratings"); needs openpyxl
            h = pd.read_excel(f, sheet_name="Ratings").rename(columns={"Item": "item", "More novel (A/B)": "more_novel (A/B)",
                                                                       "More useful (A/B)": "more_useful (A/B)"})
        else:
            h = pd.read_csv(f, encoding="utf-8-sig")
        name = f.stem.replace("human_check_", "")
        for crit in CRITERIA:
            hum, jud = [], []
            for _, r in h.iterrows():
                item = int(r["item"])
                ans = str(r.get(col[crit], "")).strip().upper()
                if ans not in ("A", "B") or item not in key:
                    continue
                p = key[item]
                j = main[(main["criterion"] == crit) & (((main["a"] == p["a"]) & (main["b"] == p["b"]))
                                                        | ((main["a"] == p["b"]) & (main["b"] == p["a"])))]
                if not len(j):
                    continue
                hum.append(ans)
                jud.append("A" if j["winner"].iloc[0] == p["a"] else "B")
                raters.setdefault((name, crit), {})[item] = ans
            if hum:
                results.append({"rater": name, "criterion": crit, "n": len(hum),
                                "agreement_with_judge": sum(x == y for x, y in zip(hum, jud)) / len(hum),
                                "kappa_with_judge": cohen_kappa(hum, jud)})
    if not results:
        rep += ["**C4. Agreement with human experts.** Not done yet. A sheet with "
                f"{HUMAN_CHECK_PAIRS} pairs is in {OUTPUT_FOLDER}/human_check/ (see INSTRUCTIONS.txt).", ""]
        return
    res = pd.DataFrame(results)
    names = sorted({r for r, _ in raters})
    between = []
    for crit in CRITERIA:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = raters.get((names[i], crit), {}), raters.get((names[j], crit), {})
                common = sorted(set(x) & set(y))
                if common:
                    between.append(f"{crit} {names[i]} vs {names[j]}: "
                                   f"{100 * sum(x[c] == y[c] for c in common) / len(common):.0f}% agreement")
    res.to_csv(ex.out / f"analysis_{SCOPE}" / "C4_human_agreement.csv", index=False)
    rep += ["**C4. Agreement with human experts** (same pairs, judged independently; 50% = chance):", "",
            md_table(res.round(2).set_index("rater"), "rater"), ""]
    if between:
        rep += ["Agreement between the human raters: " + "; ".join(between) + ". The judge is as good as an expert "
                "if its agreement with each person is close to the agreement between people.", ""]
    m = res.groupby("criterion")["agreement_with_judge"].mean()
    insights.append("**Human check.** Judge agreed with the experts on " +
                    ", ".join(f"{100 * v:.0f}% of {k} choices" for k, v in m.items()) + ".")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURES                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Fixed colour per condition, never reassigned. Baseline is neutral grey.
COLOURS = {"generic": "#8a8984", "scamper": "#2a78d6", "triz": "#eb6834",
           "analogy": "#1baf7a", "morphological": "#4a3aa7"}
MARKERS = {"generic": "o", "scamper": "s", "triz": "^", "analogy": "D", "morphological": "v"}


def make_figures(ex, df, pools, r1_df, order, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("  matplotlib not installed: pip install matplotlib (figures skipped)")
        return
    import numpy as np
    fig_dir = ex.out / f"figures_{SCOPE}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})

    # Figure 1: novelty vs usefulness, condition means with 95% intervals
    if not r1_df.empty and {"novelty", "usefulness"} <= set(r1_df["criterion"]):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.axhline(0, color="#d4d3cd", lw=0.8, zorder=0)
        ax.axvline(0, color="#d4d3cd", lw=0.8, zorder=0)
        for cond in order:
            n = r1_df[(r1_df["criterion"] == "novelty") & (r1_df["condition"] == cond)].iloc[0]
            u = r1_df[(r1_df["criterion"] == "usefulness") & (r1_df["condition"] == cond)].iloc[0]
            col = COLOURS.get(cond, "#52514e")
            ax.errorbar(n.mean_z, u.mean_z, xerr=[[n.mean_z - n.mean_lo], [n.mean_hi - n.mean_z]],
                        yerr=[[u.mean_z - u.mean_lo], [u.mean_hi - u.mean_z]], fmt=MARKERS.get(cond, "o"),
                        color=col, ms=8, lw=1.5, capsize=0, mec="white", mew=1.5)
            ax.annotate(cond, (n.mean_z, u.mean_z), xytext=(7, 5), textcoords="offset points", color="#0b0b0b")
        ax.set_xlabel("Novelty (z, mean and 95% interval)")
        ax.set_ylabel("Usefulness (z, mean and 95% interval)")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig1_novelty_vs_usefulness.png", dpi=300)
        plt.close(fig)

    # Figure 2: life-cycle shift from the baseline (diverging, grey midpoint)
    stage_cols = [f"sus_{s}" for s in LIFECYCLE_STAGES]
    if df[stage_cols].notna().any().any():
        prof = df.groupby("condition")[stage_cols].mean().reindex(order)
        diff = (prof - prof.loc[base]).drop(index=base)
        lim = max(0.25, float(np.nanmax(np.abs(diff.values))))
        cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e9e8e4", "#2a78d6"])
        fig, ax = plt.subplots(figsize=(4.6, 0.55 * len(diff) + 1.0))
        im = ax.imshow(diff.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(LIFECYCLE_STAGES)), [s.replace("_", " ") for s in LIFECYCLE_STAGES])
        ax.set_yticks(range(len(diff)), diff.index)
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                ax.text(j, i, f"{diff.values[i, j]:+.2f}", ha="center", va="center", color="#0b0b0b", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.set_label(f"difference from {base}")
        cb.outline.set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_lifecycle_shift.png", dpi=300)
        plt.close(fig)

    # Figure 3: pooled semantic diversity per condition, one point per brief
    if "diversity_pooled" in pools:
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        briefs = sorted(pools["brief_id"].unique())
        for i, cond in enumerate(order):
            g = pools[pools["condition"] == cond].set_index("brief_id").reindex(briefs)
            col = COLOURS.get(cond, "#52514e")
            jitter = np.linspace(-0.12, 0.12, len(briefs))
            ax.scatter(i + jitter, g["diversity_pooled"], s=18, color=col, alpha=0.45, lw=0)
            ax.scatter([i], [g["diversity_pooled"].mean()], s=70, color=col, marker=MARKERS.get(cond, "o"),
                       edgecolor="white", lw=1.5, zorder=3)
        ax.set_xticks(range(len(order)), order)
        ax.set_ylabel("Pooled semantic diversity\n(mean cosine distance)")
        ax.grid(axis="y", color="#e9e8e4", lw=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_spread.png", dpi=300)
        plt.close(fig)
    log(f"  figures saved in {fig_dir}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RUN                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    input_file = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else INPUT_FILE
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        print("  Run ICED27_Structure_Creativity_idea_generation.py first, or set INPUT_FILE at the top of this file.")
        sys.exit(1)
    if not MOCK and OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        print("=" * 60)
        print("ERROR: Please set your OpenAI API key!")
        print("  Open this file and paste it where it says YOUR_API_KEY_HERE")
        print("=" * 60)
        sys.exit(1)

    random.seed(RANDOM_SEED)
    ex = Experiment(input_file)
    print("=" * 60)
    if SCOPE not in ("iced", "journal"):
        print('ERROR: SCOPE must be "iced" or "journal"')
        sys.exit(1)
    print(f"  Evaluation: comparative judgement + embeddings   Scope: {SCOPE}")
    print(f"  Judge: {'MOCK' if MOCK else JUDGE_MODEL}   Embeddings: {'MOCK' if MOCK else EMBEDDING_MODEL}")
    print(f"  Input: {input_file}")
    print(f"  Output: {ex.out}")
    print(f"  Concepts: {len(ex.concepts)}   Briefs: {len(ex.briefs)}   Baseline: {ex.baseline}")
    print("=" * 60)
    plans = run_judging(ex)
    run_analysis(ex, plans)
    print(f"\n  Report: {ex.out / f'evaluation_report_{SCOPE}.md'}")


if __name__ == "__main__":
    main()
