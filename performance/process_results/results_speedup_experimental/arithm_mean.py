import sqlite3
import pandas as pd

# Connect
conn = sqlite3.connect("npbench.db")

# Load the full dataset
df = pd.read_sql_query("SELECT benchmark, details, framework, time FROM results;", conn)

# Compute median time per (benchmark, details, framework)
grouped = df.groupby(["benchmark", "details", "framework"])["time"].median().reset_index()

# For each (benchmark, framework), pick the row with the minimum median time
best = grouped.loc[grouped.groupby(["benchmark", "framework"])["time"].idxmin()]


experimental = best[best["framework"] == "experimental"]
median_time_exp = experimental["time"].mean()
print("Mean experimental time:", median_time_exp)

legacy = best[best["framework"] == "legacy"]
median_time_leg= legacy["time"].mean()
print("Mean legacy time:", median_time_leg)