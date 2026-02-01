# app.py
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re

DB_PATH = "rivm_food.db"
app = FastAPI()

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def simple_score(query: str, candidate: str) -> float:
    # cheap MVP score: token overlap
    q = set(norm(query).split())
    c = set(norm(candidate).split())
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)

class MatchRequest(BaseModel):
    text: str
    boundary: str = "consumption"  # default
    top_k: int = 5

class MatchResult(BaseModel):
    scenario_id: str
    food_name_base: str
    variant_name_raw: str
    cooking_method: Optional[str]
    storage: Optional[str]
    packaging: Optional[str]
    co2_kg_per_kg: float
    land_m2a_per_kg: float
    water_m3_per_kg: float
    confidence: float

@app.post("/match")
def match(req: MatchRequest):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Candidate pool: same boundary, search in food_name_base and variant_name_raw
    q = f"%{req.text.strip()}%"
    cur.execute("""
        SELECT * FROM food_scenarios
        WHERE boundary = ?
          AND (food_name_base LIKE ? OR variant_name_raw LIKE ? OR nevo_name_en LIKE ? OR nevo_name_nl LIKE ?)
        LIMIT 500
    """, (req.boundary, q, q, q, q))
    rows = cur.fetchall()

    # If nothing found, widen the net
    if not rows:
        cur.execute("""
            SELECT * FROM food_scenarios
            WHERE boundary = ?
            LIMIT 1000
        """, (req.boundary,))
        rows = cur.fetchall()

    scored = []
    for r in rows:
        cand = f"{r['food_name_base']} {r['variant_name_raw']} {r['nevo_name_en']} {r['nevo_name_nl']}"
        conf = max(
            simple_score(req.text, r["food_name_base"] or ""),
            simple_score(req.text, r["nevo_name_en"] or ""),
            simple_score(req.text, r["nevo_name_nl"] or ""),
            simple_score(req.text, r["variant_name_raw"] or ""),
        )
        scored.append((conf, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:req.top_k]
    con.close()

    results = []
    for conf, r in top:
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

class IngredientIn(BaseModel):
    name: str
    amount: float
    unit: str = "g"  # g, kg
    boundary: str = "consumption"

class CalcRequest(BaseModel):
    items: List[IngredientIn]

@app.post("/calc")
def calc(req: CalcRequest):
    rows_out = []
    totals = {"co2": 0.0, "land": 0.0, "water": 0.0}

    for item in req.items:
        kg = item.amount / 1000.0 if item.unit.lower() == "g" else item.amount
        # match best
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
