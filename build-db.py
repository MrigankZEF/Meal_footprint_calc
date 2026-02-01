# build_db.py
import sqlite3
import pandas as pd

CSV_PATH = "food_scenarios_normalized.csv"
DB_PATH = "rivm_food.db"

df = pd.read_csv(CSV_PATH)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.execute("DROP TABLE IF EXISTS food_scenarios")

cur.execute("""
CREATE TABLE food_scenarios (
  scenario_id TEXT PRIMARY KEY,
  boundary TEXT,
  variant_name_raw TEXT,
  food_name_base TEXT,
  storage TEXT,
  packaging TEXT,
  cooking_method TEXT,
  stage_tag TEXT,
  nevo_code TEXT,
  nevo_name_nl TEXT,
  nevo_product_group_nl TEXT,
  nevo_name_en TEXT,
  nevo_product_group_en TEXT,
  co2_kg_per_kg REAL,
  land_m2a_per_kg REAL,
  water_m3_per_kg REAL,
  source_version TEXT
)
""")

df.to_sql("food_scenarios", con, if_exists="append", index=False)

# Indexes for fast search
cur.execute("CREATE INDEX IF NOT EXISTS idx_food_base ON food_scenarios(food_name_base)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_variant_raw ON food_scenarios(variant_name_raw)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_boundary ON food_scenarios(boundary)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_nevo_code ON food_scenarios(nevo_code)")

con.commit()
con.close()

print("Built DB:", DB_PATH, "rows:", len(df))
