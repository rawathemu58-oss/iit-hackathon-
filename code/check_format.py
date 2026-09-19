"""
The Corporate Heist - submission FORMAT checker.
This only checks that your submission.csv is well-formed. It does NOT score it.

Usage:  python check_format.py submission.csv test.csv
"""
import sys
import pandas as pd

K = 500
sub_path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
test_path = sys.argv[2] if len(sys.argv) > 2 else "test.csv"
errors, warnings = [], []
try:
    s = pd.read_csv(sub_path, dtype=str)
except Exception as e:
    print(f"FAIL: cannot read {sub_path}: {e}"); sys.exit(1)
test_ids = set(pd.read_csv(test_path, usecols=["candidate_id"], dtype=str).candidate_id)
if list(s.columns) != ["rank", "candidate_id"]:
    errors.append(f"columns must be exactly: rank,candidate_id  (found: {','.join(s.columns)})")
else:
    ids = s.candidate_id.str.strip()
    r = pd.to_numeric(s["rank"], errors="coerce")
    if len(s) != K: errors.append(f"must contain exactly {K} rows (found {len(s)})")
    if r.isna().any() or sorted(r.astype(int).tolist()) != list(range(1, len(s) + 1)):
        errors.append("rank must be the integers 1..N with no gaps or repeats")
    if ids.duplicated().any(): errors.append(f"{int(ids.duplicated().sum())} candidate_id values are repeated")
    unknown = sorted(set(ids) - test_ids)
    if unknown: errors.append(f"{len(unknown)} candidate_id values are not in test.csv, e.g. {unknown[:3]}")
if errors:
    print("FAIL"); [print("  -", e) for e in errors]; sys.exit(1)
print("OK - submission.csv is correctly formatted.")
