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

# Drop numpy results in 
legacy = legacy[legacy["framework"] != "numpy"].reset_index(drop=True)

# rename in legacy
legacy["framework"] = legacy["framework"].replace({"dace_gpu": "legacy"})

# rename in experimental
exp["framework"] = exp["framework"].replace({"dace_gpu": "experimental"})

combined = pd.concat([legacy, exp], ignore_index=True)

# remove old file if it exists
if os.path.exists("npbench.db"):
    os.remove("npbench.db")

# create new connection
conn_out = sqlite3.connect("npbench.db")

# save combined table into it
combined.to_sql("results", conn_out, if_exists="replace", index=False)

conn_legacy.close()
conn_exp.close()
conn_out.close()