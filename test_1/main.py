"""
The Corporate Heist - Innov8 4.0 preliminary round.

Pipeline: robust parse -> merit-only feature build -> seed-averaged rank-blended
ensemble trained on the Archive -> validate on the Ledger -> apply this-cycle
rules (exclusions + code-contribution fast-track + never-hired-from-college +
old-boys) on the Vault -> rank top 500 -> submission.csv

Run: python main.py   (expects train.csv, dev.csv, dev_winners.csv, test.csv in cwd)
"""
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import feats

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")
dev_winners = pd.read_csv("dev_winners.csv")
test = pd.read_csv("test.csv")

# ----------------------------------------------------------------------
# 2. PARSE + FEATURE BUILD  (see feats.py)
#
#    Every free-typed column is parsed to one canonical unit. This matters
#    more than the model: technical_assessment alone arrives as a raw 0-100
#    int, as "45/100", as "40.0 %" and as a 0-1 fraction, and naive
#    pd.to_numeric silently drops the first two forms and reads 0.45 as a
#    near-zero score instead of 45/100. Same story for kpi_met (Y/Yes/1/TRUE
#    are all "met") and awards ("0" and "-" are zero, not one).
# ----------------------------------------------------------------------
train_c = feats.build(train)
dev_c = feats.build(dev)
test_c = feats.build(test)

# role-normalised (statistics always taken from the Archive) + within-pool rank
train_c = feats.add_pool_rank(feats.add_role_z(train_c, train_c))
dev_c = feats.add_pool_rank(feats.add_role_z(train_c, dev_c))
test_c = feats.add_pool_rank(feats.add_role_z(train_c, test_c))

# ----------------------------------------------------------------------
# 3. WHICH FEATURES THE MODEL IS ALLOWED TO SEE
#
#    The debrief is explicit that the new panel ignores big-name colleges,
#    big-metro candidates, glowing referrals, big-brand employers, "proper"
#    degrees and gap-free CVs -- the old panel quietly inflated all of them.
#    Those columns are therefore never fed to the model, even though target-
#    encoding them lifts dev precision@150 from 0.487 to 0.640. That lift is
#    the old regime's bias being memorised, and the Vault is the new cycle.
#    The two old-cycle effects the debrief says survive (the old-boys colleges
#    and nothing else) are applied as an explicit, auditable rule in section 6
#    instead of being laundered through the model.
# ----------------------------------------------------------------------
# applied_role is NOT on the debrief's banned list -- it is what the candidate
# applied for, not a pedigree signal -- and the roles have genuinely different
# top-5% base rates in the Archive (Data Scientist 8.3%, QA Automation 3.1%).
# The role-z features deliberately remove that level difference, so it is fed
# back explicitly as the role's prior probability of landing in the top 5%.
_thr95 = train_c["post_hire_score"].quantile(0.95)
ROLE_PRIOR = (train_c.assign(_t=(train_c["post_hire_score"] >= _thr95))
              .groupby("applied_role")["_t"].mean())
for _f in (train_c, dev_c, test_c):
    _f["role_prior"] = _f["applied_role"].map(ROLE_PRIOR)

FEATURES = feats.FEATURES + ["role_prior"]

X_train = train_c[FEATURES]
y_train = train_c["post_hire_score"]
y_bin = (y_train >= y_train.quantile(0.95)).astype(int)

SEEDS = [1, 2, 3, 42, 100]
REG_W, CLF_W, RIDGE_W = 0.40, 0.40, 0.20

_imp = SimpleImputer(strategy="median")
_sc = StandardScaler()
_X_scaled = _sc.fit_transform(_imp.fit_transform(X_train))

_models = []
for s in SEEDS:
    reg = HistGradientBoostingRegressor(random_state=s, max_iter=300)
    reg.fit(X_train, y_train)
    clf = HistGradientBoostingClassifier(random_state=s, max_iter=300,
                                         class_weight="balanced")
    clf.fit(X_train, y_bin)
    ridge = Ridge(alpha=10.0)
    ridge.fit(_X_scaled, y_train)
    _models.append((reg, clf, ridge))


def predict_ensemble(df):
    """Rank-percentile blend of a regressor, a top-5% classifier and a linear
    model. Blending by rank, not raw score, because the three live on
    different scales."""
    X = df[FEATURES]
    Xs = _sc.transform(_imp.transform(X))
    r = np.mean([m[0].predict(X) for m in _models], axis=0)
    c = np.mean([m[1].predict_proba(X)[:, 1] for m in _models], axis=0)
    g = np.mean([m[2].predict(Xs) for m in _models], axis=0)
    rp = pd.Series(r).rank(pct=True).values
    cp = pd.Series(c).rank(pct=True).values
    gp = pd.Series(g).rank(pct=True).values
    return REG_W * rp + CLF_W * cp + RIDGE_W * gp


dev_c["pred_score"] = predict_ensemble(dev_c)
test_c["pred_score"] = predict_ensemble(test_c)

# ----------------------------------------------------------------------
# 4. VALIDATE on the Ledger
#    dev.csv is an OLD-cycle pool, so this measures the model core only; the
#    section-6 rules are this-cycle-only and cannot be validated here.
# ----------------------------------------------------------------------
top150 = set(dev_c.nlargest(150, "pred_score")["candidate_id"])
winners = set(dev_winners["candidate_id"])
print(f"[validation] precision@150 on dev: {len(top150 & winners) / 150:.4f} "
      f"(random baseline ~ {150 / len(dev_c):.3f})")

# ----------------------------------------------------------------------
# 5. DEDUPE + FABRICATION CHECKS on the Vault
#    Duplicates are the same person re-entered by a second recruiter, so the
#    differences are cosmetic: phone punctuation, email dots and "+jobs" tags,
#    name casing. Keys are normalised hard enough to survive all three, then
#    joined with union-find so a chain of partial matches collapses to one
#    person. O(n), so it fits the 5-minute budget on 10k rows.
# ----------------------------------------------------------------------
def norm_name(s):
    s = re.sub(r"[^a-z ]", " ", str(s).lower())
    return " ".join(sorted(s.split()))


def norm_email(s):
    local = str(s).lower().split("@")[0]
    local = local.split("+")[0]
    return re.sub(r"[^a-z]", "", local)


def norm_phone(s):
    return re.sub(r"\D", "", str(s))[-10:]


parent = list(range(len(test_c)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


keys = {
    "phone": test_c["phone"].map(norm_phone),
    "email": test_c["email"].map(norm_email),
    "name_grad": test_c["full_name"].map(norm_name) + "|"
                 + test_c["graduation_year"].astype(str),
}
for series in keys.values():
    for _, idx in series[series.ne("") & series.notna()].groupby(series).groups.items():
        pos = [test_c.index.get_loc(i) for i in idx]
        for p in pos[1:]:
            union(pos[0], p)

groups = pd.Series([find(i) for i in range(len(test_c))], index=test_c.index)
test_c["person_group"] = groups
# keep the highest-scoring row per person; the rest are duplicate entries
best_row = test_c.groupby("person_group")["pred_score"].idxmax()
test_c["is_duplicate"] = ~test_c.index.isin(best_row)


def flag_fabricated(df):
    age = df["age"]
    grad_age = age - (feats.CUR_YEAR - df["graduation_year"])
    bad_age = age.notna() & ~age.between(18, 65)
    bad_grad = grad_age.notna() & ~grad_age.between(15, 60)
    # more experience than a working life can hold
    bad_exp = df["exp_years"].notna() & age.notna() & (df["exp_years"] > age - 17)
    # career history claims far more time than the person has been out of college
    bad_path = df["career_months"].notna() & df["yrs_since_grad"].notna() & \
        (df["career_months"] / 12.0 > df["yrs_since_grad"] + 2)
    # shiny on one test, impossible on the other
    bad_gap = (df["tech_100"] > 90) & (df["aptitude_10"] < 2)
    return bad_age | bad_grad | bad_exp | bad_path | bad_gap


test_c["is_fabricated"] = flag_fabricated(test_c)

# ----------------------------------------------------------------------
# 6. THIS-CYCLE RULES
# ----------------------------------------------------------------------
# 6a. title inflation: titles climbing faster than the years behind them.
#     "'Head of' something three years out of college" -> straight to the bin.
test_c["title_inflation"] = (
    ((test_c["title_seniority"] >= 4) & (test_c["exp_years"] < 6))
    | ((test_c["title_seniority"] >= 3) & (test_c["exp_years"] < 4))
).fillna(False)

# 6b. cannot join inside two months.
test_c["too_slow"] = (test_c["notice_days"] > 60).fillna(False)

# 6c. sustained public code contributions. Parsed from "~31", "40+",
#     "28 merged PRs" and plain integers; "not tracked" and blanks are
#     genuinely unknown, not zero, so they are neither boosted nor punished.
test_c["contrib"] = test_c["public_code_contributions"].map(feats.parse_code_contrib)
test_c["sustained"] = (test_c["contrib"] >= 24).fillna(False)

# 6d. colleges Nightingale has never hired from, where the candidate also
#     tops the technical assessment for the role they applied to.
hired_before = set(train_c["inst_norm"]) | set(dev_c["inst_norm"])
test_c["new_college"] = ~test_c["inst_norm"].isin(hired_before)
test_c["tops_assessment"] = (test_c["tech_100_prank"] >= 0.90).fillna(False)
test_c["new_blood"] = test_c["new_college"] & test_c["tops_assessment"]

# 6e. the one old-cycle preference that survived the reorg. Identified from the
#     Archive as institutes with a large post_hire_score boost that the merit
#     features do not explain, while carrying none of the prestige that the old
#     panel inflated broadly -- i.e. not IIT/NIT/IIIT/BITS/IISc/DTU/NSIT.
OLD_BOYS = {
    "university of calicut",
    "maharshi dayanand sarswati university ajmer",
    "tilak maharashtra vidyapeeth",
    "utkal university",
    "smk fomra institute of technology",
}
test_c["old_boys"] = test_c["inst_norm"].isin(OLD_BOYS)

# ----------------------------------------------------------------------
# 7. SCORE ADJUSTMENT AND FINAL RANKING
#    pred_score is a rank percentile in [0, 1], so a boost of 0.30 moves a
#    candidate roughly 30 percentiles up the pool -- large enough to fast-track
#    a merely good profile, small enough that a weak one still does not make
#    the top 500.
# ----------------------------------------------------------------------
excluded = test_c["is_fabricated"] | test_c["is_duplicate"] | \
    test_c["too_slow"] | test_c["title_inflation"]
pool = test_c[~excluded].copy()

pool["final_score"] = (
    pool["pred_score"]
    + 0.30 * pool["sustained"]
    + 0.20 * pool["new_blood"]
    + 0.08 * pool["old_boys"]
)

shortlist = pool.nlargest(500, "final_score").copy()
shortlist["rank"] = range(1, len(shortlist) + 1)
shortlist[["rank", "candidate_id"]].to_csv("submission.csv", index=False)

print(f"[excluded] fabricated={int(test_c['is_fabricated'].sum())} "
      f"duplicate={int(test_c['is_duplicate'].sum())} "
      f"notice>60d={int(test_c['too_slow'].sum())} "
      f"title_inflation={int(test_c['title_inflation'].sum())} "
      f"| pool={len(pool)}")
print(f"[boosts in shortlist] sustained={int(shortlist['sustained'].sum())} "
      f"new_blood={int(shortlist['new_blood'].sum())} "
      f"old_boys={int(shortlist['old_boys'].sum())}")
print(f"[done] wrote submission.csv with {len(shortlist)} rows")