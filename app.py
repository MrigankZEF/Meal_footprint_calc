# app.py
import sqlite3
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple

DB_PATH = "rivm_food.db"
app = FastAPI()

# -------------------------
# Text normalization helpers
# -------------------------

WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "to", "for", "in", "on", "with",
    "average", "nl", "economic", "consumed", "consumer",
}

def norm_basic(s: str) -> str:
    """Lowercase and collapse whitespace; keep alnum only."""
    if not s:
        return ""
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    """Tokenize into alnum words, apply very light stemming/plural normalization."""
    s = norm_basic(s)
    tokens = WORD_RE.findall(s)
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        out.append(t)
        # plural normalization: onions -> onion, tomatoes -> tomato
        # also handle simple 'es' endings: tomatoes -> tomato
        if len(t) > 3:
            if t.endswith("es"):
                out.append(t[:-2])
            elif t.endswith("s"):
                out.append(t[:-1])
    # unique while preserving order
    seen = set()
    uniq = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def whole_word_contains(text: str, word: str) -> bool:
    """Check if 'word' appears as a whole word in 'text' (both already normalized)."""
    if not text or not word:
        return False
    # \b doesn't play perfectly with numbers; we already tokenize on [a-z0-9]+ so do manual
    tokens = set(WORD_RE.findall(text))
    return word in tokens

# -------------------------
# Scoring
# -------------------------

def score_query_against_field(q_tokens: List[str], field_text: str, query_norm: str) -> float:
    if not field_text:
        return 0.0

    field_norm = norm_basic(field_text)
    f_tokens_list = tokenize(field_norm)
    f_tokens = set(f_tokens_list)
    if not q_tokens or not f_tokens:
        return 0.0

    q_set = set(q_tokens)

    # recall-ish overlap: how many query tokens matched
    inter = q_set & f_tokens
    overlap = len(inter) / max(1, len(q_set))

    # exact match bonus
    exact = 1.0 if field_norm == query_norm else 0.0

    # contains bonuses
    contains_frac = sum(1.0 for qt in q_set if whole_word_contains(field_norm, qt)) / max(1, len(q_set))

    # IMPORTANT FIX:
    # Only use "contains_all" bonus when query has >=2 tokens.
    contains_all = 0.0
    if len(q_set) >= 2:
        contains_all = 1.0 if contains_frac == 1.0 else 0.0

    # Start-of-string bonus (very useful for food_name_base like "Tomato")
    starts = 1.0 if field_norm.startswith(query_norm) else 0.0

    # Penalty for being much longer than query (prevents "herring ... tomato sauce" beating "Tomato")
    # For single-token queries, this matters most.
    extra_tokens = max(0, len(f_tokens_list) - len(q_set))
    length_penalty = 1.0 / (1.0 + 0.20 * extra_tokens)

    score = (
        0.55 * overlap +
        0.20 * contains_frac +
        0.20 * starts +
        0.50 * exact +
        0.10 * contains_all
    )

    return score * length_penalty


def final_score(query: str, row: sqlite3.Row) -> float:
    """
    Compute a robust score for a dataset row using multiple fields and weights.
    Returns score in [0, 1].
    """
    query_norm = norm_basic(query)
    q_tokens = tokenize(query_norm)

    if not q_tokens:
        return 0.0

    # Field weights: put most trust in food_name_base and NEVO names
    fields = [
        ("food_name_base", 1.00),
        ("nevo_name_en",   0.90),
        ("nevo_name_nl",   0.85),
        ("variant_name_raw", 0.55),
    ]

    best = 0.0
    for fname, w in fields:
        txt = row[fname] if fname in row.keys() else None
        s = score_query_against_field(q_tokens, txt or "", query_norm)
        best = max(best, w * s)

    # Clamp to [0,1]
    return float(max(0.0, min(1.0, best)))

def expand_query_terms(text: str) -> List[str]:
    """
    Deterministic query expansion for very short queries.
    Keeps system consistent for all users.
    """
    t = norm_basic(text)
    toks = tokenize(t)

    terms = [text]

    # If single token query, try a couple deterministic expansions
    if len(toks) == 1:
        w = toks[0]
        # generic expansions that often exist in datasets
        if w == "chicken":
            terms += ["chicken meat", "chicken fillet"]
        elif w == "cheese":
            terms += ["hard cheese", "soft cheese"]
        elif w == "onion":
            terms += ["onions"]
        elif w == "tomato":
            terms += ["tomatoes"]

    # unique, preserve order
    seen = set()
    out = []
    for x in terms:
        nx = norm_basic(x)
        if nx and nx not in seen:
            seen.add(nx)
            out.append(x)
    return out

# -------------------------
# API models
# -------------------------

class MatchRequest(BaseModel):
    text: str
    boundary: str = "consumption"  # default
    top_k: int = 5

class IngredientIn(BaseModel):
    name: str
    amount: float
    unit: str = "g"  # g, kg
    boundary: str = "consumption"

class CalcRequest(BaseModel):
    items: List[IngredientIn]

# -------------------------
# DB helpers
# -------------------------

def get_connection():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def fetch_candidate_rows(cur, boundary: str, term: str, limit: int) -> List[sqlite3.Row]:
    """
    Candidate pool based on LIKE search across key fields.
    """
    q = f"%{term.strip()}%"
    cur.execute("""
        SELECT * FROM food_scenarios
        WHERE boundary = ?
          AND (
                food_name_base LIKE ?
             OR variant_name_raw LIKE ?
             OR nevo_name_en LIKE ?
             OR nevo_name_nl LIKE ?
          )
        LIMIT ?
    """, (boundary, q, q, q, q, limit))
    return cur.fetchall()

# -------------------------
# /match endpoint
# -------------------------

@app.post("/match")
def match(req: MatchRequest):
    con = get_connection()
    cur = con.cursor()

    # Expand query deterministically (helps "chicken" vs "eggs chicken", etc.)
    terms = expand_query_terms(req.text)

    # Fetch candidates for each term; union by scenario_id to avoid duplicates
    cand_by_id: Dict[str, sqlite3.Row] = {}

    for term in terms:
        rows = fetch_candidate_rows(cur, req.boundary, term, limit=800)
        for r in rows:
            sid = r["scenario_id"]
            if sid not in cand_by_id:
                cand_by_id[sid] = r

        # If nothing found with phrase LIKE, fall back to token-based LIKE (deterministic)
    if not cand_by_id:
        # try each token separately (OR), but still scoped to boundary
        toks = tokenize(req.text)
        # if query is multi-token like "cottage cheese", this will try "%cottage%" and "%cheese%"
        for tok in toks:
            rows = fetch_candidate_rows(cur, req.boundary, tok, limit=800)
            for r in rows:
                sid = r["scenario_id"]
                if sid not in cand_by_id:
                    cand_by_id[sid] = r

    # If still nothing, return empty (no random fallback)
    if not cand_by_id:
        con.close()
        return {"query": req.text, "boundary": req.boundary, "matches": []}


    scored: List[Tuple[float, str, sqlite3.Row]] = []
    for sid, r in cand_by_id.items():
        # score against original query only (so behavior stays consistent)
        conf = final_score(req.text, r)

        # deterministic tie-breaker: prefer matches where food_name_base contains query token(s)
        # and then sort by food_name_base alphabetically for stable ordering
        fnb = norm_basic(r["food_name_base"] or "")
        tie = 1 if any(whole_word_contains(fnb, t) for t in tokenize(req.text)) else 0

        scored.append((conf, str(tie), r))

    # Sort: confidence desc, tie desc, then food_name_base asc for stability
    scored.sort(key=lambda x: (-x[0], -int(x[1]), norm_basic(x[2]["food_name_base"] or ""), x[2]["scenario_id"]))

    top = scored[: max(1, req.top_k)]
    con.close()

    results: List[Dict[str, Any]] = []
    for conf, _, r in top:
        results.append({
            "scenario_id": r["scenario_id"],
            "food_name_base": r["food_name_base"],
            "variant_name_raw": r["variant_name_raw"],
            "cooking_method": r["cooking_method"],
            "storage": r["storage"],
            "packaging": r["packaging"],
            "co2_kg_per_kg": r["co2_kg_per_kg"],
            "land_m2a_per_kg": r["land_m2a_per_kg"],
            "water_m3_per_kg": r["water_m3_per_kg"],
            "confidence": float(conf),
        })

    return {"query": req.text, "boundary": req.boundary, "matches": results}

# -------------------------
# /calc endpoint
# -------------------------

@app.post("/calc")
def calc(req: CalcRequest):
    rows_out = []
    totals = {"co2": 0.0, "land": 0.0, "water": 0.0}

    for item in req.items:
        kg = item.amount / 1000.0 if item.unit.lower() == "g" else item.amount

        m = match(MatchRequest(text=item.name, boundary=item.boundary, top_k=1))

        if not m["matches"]:
            rows_out.append({
                "ingredient": item.name,
                "kg": kg,
                "mapped_to": None,
                "confidence": 0.0,
                "co2": None, "land": None, "water": None,
                "note": "no match"
            })
            continue

        best = m["matches"][0]
        co2 = kg * best["co2_kg_per_kg"]
        land = kg * best["land_m2a_per_kg"]
        water = kg * best["water_m3_per_kg"]

        totals["co2"] += co2
        totals["land"] += land
        totals["water"] += water

        rows_out.append({
            "ingredient": item.name,
            "kg": kg,
            "mapped_to": best["food_name_base"],
            "variant": best["variant_name_raw"],
            "confidence": best["confidence"],
            "co2": co2,
            "land": land,
            "water": water,
            "note": "ok" if best["confidence"] >= 0.75 else "needs confirmation"
        })

    return {"rows": rows_out, "totals": totals}
