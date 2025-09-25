import os
import sqlite3
import pandas as pd

# paths to DBs (assuming in same directory)
legacy_db = "../legacy_npbench.db"
exp_db = "../exp_npbench.db"

# open connections
conn_legacy = sqlite3.connect(legacy_db)
conn_exp = sqlite3.connect(exp_db)

# load results table from both
legacy = pd.read_sql_query("SELECT * FROM results", conn_legacy)
exp = pd.read_sql_query("SELECT * FROM results", conn_exp)

# Drop numpy results in BOTH tables
legacy = legacy[legacy["framework"] != "numpy"].reset_index(drop=True)
exp = exp[exp["framework"] != "numpy"].reset_index(drop=True)

# rename frameworks accordingly to legacy and experimental
legacy["framework"] = legacy["framework"].replace({"dace_gpu": "legacy"})
exp["framework"] = exp["framework"].replace({"dace_gpu": "experimental"})

# combine results and store them in new database
combined = pd.concat([legacy, exp], ignore_index=True)

# remove old file if it exists
if os.path.exists("npbench.db"):
    os.remove("npbench.db")

conn_out = sqlite3.connect("npbench.db")
combined.to_sql("results", conn_out, if_exists="replace", index=False)

# close all connections
conn_legacy.close()
conn_exp.close()
conn_out.close()