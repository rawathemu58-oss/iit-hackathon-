"""
The Corporate Heist - Innov8 4.0 preliminary round
Improved pipeline: clean -> dedupe/fake-filter -> ensemble model -> validate on dev ->
predict on test -> apply insider-rule adjustments -> rank top 500 -> submission.csv

Run: python main.py   (expects train.csv, dev.csv, dev_winners.csv, test.csv in cwd)
"""
import re
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")
dev_winners = pd.read_csv("dev_winners.csv")
test = pd.read_csv("test.csv")

# ----------------------------------------------------------------------
# 2. CLEANING HELPERS  (columns are typed inconsistently on purpose)
# ----------------------------------------------------------------------

def parse_experience(x):
    """'17.6 years' / '8+ yrs' / '4.6' / '>20' / '19 months' -> years (float)"""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(">", "").replace("+", "").strip()
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "month" in s:
        val = val / 12.0
    return val


def parse_ctc(x):
    """Normalize every currency format to LPA (lakhs per annum, INR)."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("₹", "").replace(",", "").replace("inr", "").strip()
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "cr" in s:
        return val * 100.0          # crore -> lakh
    if "lpa" in s or "lakh" in s or s.endswith("l"):
        return val
    if val > 10000:
        return val / 100000.0
    return val


def parse_aptitude(x):
    """'7.3' / '5.7/10' / '86%' -> 0-10 scale"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if "%" in s:
        return float(s.replace("%", "")) / 10.0
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den) * 10.0
    return float(s)


RATING_MAP = {
    "outstanding": 5, "5": 5, "5.0": 5, "5/5": 5,
    "exceeds expectations": 4, "4": 4, "4.0": 4, "4/5": 4,
    "meets expectations": 3, "3": 3, "3.0": 3, "3/5": 3,
    "needs improvement": 2, "2": 2, "2.0": 2, "2/5": 2,
    "unsatisfactory": 1, "1": 1, "1.0": 1, "1/5": 1,
}


def parse_rating(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in RATING_MAP:
        return RATING_MAP[s]
    if "not rated" in s or "new joiner" in s:
        return np.nan
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else np.nan


def parse_notice(x):
    """-> days. 'Immediate'/'Available now' -> 0. 'Serving notice - 45 days' -> 45."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if "immediate" in s or "available now" in s:
        return 0
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "month" in s:
        val *= 30
    return val


def parse_last_job_change(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    if s == "never":
        return 0
    if ">" in s:
        return float(s.replace(">", "")) + 1
    try:
        return float(s)
    except ValueError:
        return np.nan


def count_career_steps(x):
    if pd.isna(x):
        return 0
    return len(re.split(r"[>|]|->|→", str(x)))


def title_seniority(title):
    """Rough seniority score from a job title string."""
    if pd.isna(title):
        return 0
    t = str(title).lower()
    if any(k in t for k in ["head of", "chief", "vp ", "vice president", "director"]):
        return 4
    if any(k in t for k in ["principal", "staff", "lead"]):
        return 3
    if any(k in t for k in ["senior", "sr.", "sr "]):
        return 2
    if any(k in t for k in ["junior", "jr.", "trainee", "intern", "apprentice", "associate"]):
        return 0
    return 1


# recruiter_note is free text; a handful of recurring phrases carry real signal
# (impact, ownership, mentorship) vs noise (hobbies, logistics) vs red flags.
POS_NOTE_WORDS = [
    "shipped a feature used by", "primary on-call owner", "mentored",
    "led the migration", "wrote most runbooks", "strong communicator", "owns",
]
NEG_NOTE_WORDS = [
    "needed frequent guidance", "missed deadline", "struggled",
    "performance concern", "frequent guidance",
]


def note_score(x):
    if pd.isna(x):
        return 0
    s = str(x).lower()
    score = 0
    for w in POS_NOTE_WORDS:
        if w in s:
            score += 1
    for w in NEG_NOTE_WORDS:
        if w in s:
            score -= 1
    return score


def clean_frame(df):
    df = df.copy()
    df["exp_years"] = df["total_experience"].apply(parse_experience)
    df["current_ctc_lpa"] = df["current_ctc"].apply(parse_ctc)
    df["expected_ctc_lpa"] = df["expected_ctc"].apply(parse_ctc)
    df["aptitude_10"] = df["aptitude_score"].apply(parse_aptitude)
    df["rating_5"] = df["last_rating"].apply(parse_rating)
    df["notice_days"] = df["notice_period"].apply(parse_notice)
    df["last_change_yrs"] = df["last_job_change"].apply(parse_last_job_change)
    df["career_steps"] = df["career_path"].apply(count_career_steps)
    df["title_seniority"] = df["current_title"].apply(title_seniority)
    df["kpi_met_bin"] = (df["kpi_met"].astype(str).str.upper() == "Y").astype(int)
    df["currently_enrolled_bin"] = (
        df["currently_enrolled"].astype(str).str.lower() != "no_enrollment"
    ).astype(int)
    df["overtime_bin"] = (df["overtime_history"].astype(str).str.upper() == "YES").astype(int)
    df["technical_assessment"] = pd.to_numeric(df["technical_assessment"], errors="coerce")
    df["training_hours"] = pd.to_numeric(df["training_hours"], errors="coerce")
    df["trainings_last_year"] = pd.to_numeric(df["trainings_last_year"], errors="coerce")
    df["num_employers"] = pd.to_numeric(df["num_employers"], errors="coerce")
    df["note_score"] = df["recruiter_note"].apply(note_score)
    df["award_count"] = df["awards"].apply(lambda x: 0 if pd.isna(x) else len(str(x).split(",")))

    # --- insider clue #1: title inflation (senior title, too little real experience) ---
    df["title_inflation_flag"] = (
        (df["title_seniority"] >= 3) & (df["exp_years"].fillna(0) < 4)
    ).astype(int)

    # --- insider clue #2: can't join inside 2 months ---
    df["notice_too_long"] = (df["notice_days"].fillna(0) > 60).astype(int)

    # --- code contributions (test.csv only) ---
    if "public_code_contributions" in df.columns:
        df["code_contrib"] = pd.to_numeric(df["public_code_contributions"], errors="coerce").fillna(0)
        df["sustained_contributor"] = (df["code_contrib"] >= 24).astype(int)  # ~dozens/yr
    else:
        df["code_contrib"] = 0
        df["sustained_contributor"] = 0

    # --- old-boys network: a small set of colleges Nightingale leadership attended.
    # Not given explicitly -> proxy with a short, defensible list of elite institutes.
    # TODO: refine using patterns you find in train.csv (see documentation section 4).
    OLD_BOYS = {"iit", "indian institute of technology"}
    df["old_boys_flag"] = df["institute"].astype(str).str.lower().apply(
        lambda s: int(any(k in s for k in OLD_BOYS))
    )

    return df


train_c = clean_frame(train)
dev_c = clean_frame(dev)
test_c = clean_frame(test)

# ----------------------------------------------------------------------
# 3. DEDUPE + FAKE-PROFILE FILTER (applied to the VAULT / test set, since
#    that's what we are shortlisting from)
# ----------------------------------------------------------------------

def dedupe_key(df):
    """Cheap first pass: near-identical email local-part + phone last 8 digits."""
    email_key = df["email"].astype(str).str.lower().str.split("@").str[0]
    phone_key = df["phone"].astype(str).str.replace(r"\D", "", regex=True).str[-8:]
    return email_key + "_" + phone_key


def flag_duplicates(df):
    df = df.copy()
    df["dup_key"] = dedupe_key(df)
    df["is_duplicate"] = df.duplicated(subset="dup_key", keep="first")
    if HAVE_RAPIDFUZZ:
        # second pass: fuzzy match full_name + institute for rows not already caught
        names = df["full_name"].astype(str) + "|" + df["institute"].astype(str)
        seen = {}
        dup_flags = df["is_duplicate"].tolist()
        for i, key in enumerate(names):
            if dup_flags[i]:
                continue
            matched = False
            for seen_key in seen:
                if fuzz.ratio(key, seen_key) > 92:
                    matched = True
                    break
            if matched:
                dup_flags[i] = True
            else:
                seen[key] = i
        df["is_duplicate"] = dup_flags
    return df


def flag_fabricated(df):
    """Simple consistency checks -> flag profiles that 'don't add up'."""
    df = df.copy()
    age_ok = df["age"].between(18, 65) | df["age"].isna()
    grad_age = df["age"] - (2026 - df["graduation_year"])
    grad_ok = grad_age.between(15, 60) | grad_age.isna()
    exp_ok = df["exp_years"].fillna(0) <= (df["age"].fillna(60) - 18 + 1)
    score_gap = (df["technical_assessment"].fillna(0) > 90) & (df["aptitude_10"].fillna(10) < 2)
    df["is_fabricated"] = (~age_ok) | (~grad_ok) | (~exp_ok) | score_gap
    return df


test_c = flag_duplicates(test_c)
test_c = flag_fabricated(test_c)

# ----------------------------------------------------------------------
# 4. MODEL — two complementary views of the same problem, blended by RANK
#    (not raw score, since a regressor and a classifier live on different
#    scales):
#      (a) REGRESSION — predict the continuous post_hire_score.
#      (b) CLASSIFICATION — predict P(top 5%) directly, which optimizes
#          for the actual selection boundary instead of the whole curve.
#    Each is seed-averaged over 5 seeds to cut down model variance, then
#    the two rank-percentiles are blended (regression weighted higher —
#    it validated best on the dev set).
# ----------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

FEATURES = [
    "exp_years", "current_ctc_lpa", "expected_ctc_lpa", "aptitude_10", "rating_5",
    "notice_days", "last_change_yrs", "career_steps", "title_seniority",
    "kpi_met_bin", "currently_enrolled_bin", "overtime_bin",
    "technical_assessment", "training_hours", "trainings_last_year",
    "num_employers", "title_inflation_flag", "notice_too_long", "old_boys_flag",
    "note_score", "award_count", "age", "graduation_year",
]

X_train = train_c[FEATURES]
y_train = train_c["post_hire_score"]
TOP5_THRESHOLD = y_train.quantile(0.95)
y_train_bin = (y_train >= TOP5_THRESHOLD).astype(int)

SEEDS = [1, 2, 3, 42, 100]
REG_BLEND_WEIGHT = 0.7  # tuned on the dev set; regression view carries more weight


def predict_ensemble(df):
    X = df[FEATURES]
    reg_preds, clf_preds = [], []
    for s in SEEDS:
        reg = HistGradientBoostingRegressor(random_state=s, max_iter=300)
        reg.fit(X_train, y_train)
        reg_preds.append(reg.predict(X))

        clf = HistGradientBoostingClassifier(random_state=s, max_iter=300, class_weight="balanced")
        clf.fit(X_train, y_train_bin)
        clf_preds.append(clf.predict_proba(X)[:, 1])
    reg_avg = np.mean(reg_preds, axis=0)
    clf_avg = np.mean(clf_preds, axis=0)
    # blend by rank percentile so the two different scales combine fairly
    reg_pct = pd.Series(reg_avg).rank(pct=True).values
    clf_pct = pd.Series(clf_avg).rank(pct=True).values
    return REG_BLEND_WEIGHT * reg_pct + (1 - REG_BLEND_WEIGHT) * clf_pct


dev_c["pred_score"] = predict_ensemble(dev_c)
test_c["pred_score"] = predict_ensemble(test_c)

# ----------------------------------------------------------------------
# 5. VALIDATE on dev.csv against dev_winners.csv
#    NOTE: dev.csv was drawn from a PAST cycle under the OLD hiring standard
#    (same as train.csv). A high precision@150 here is a useful sanity check
#    that the model learned real signal, but it does not fully confirm
#    performance on the Vault, which follows the NEW panel's rules. That is
#    exactly why section 6 layers insider-rule adjustments on top of the
#    model instead of trusting the model's raw output on test.csv.
# ----------------------------------------------------------------------
dev_sorted = dev_c.sort_values("pred_score", ascending=False)
top150 = set(dev_sorted.head(150)["candidate_id"])
winners = set(dev_winners["candidate_id"])
precision_at_150 = len(top150 & winners) / 150
print(f"[validation] precision@150 on dev set: {precision_at_150:.3f}  "
      f"(random baseline ~= {150/len(dev_c):.3f})")

# ----------------------------------------------------------------------
# 6. APPLY INSIDER-RULE ADJUSTMENTS + EXCLUSIONS ON TEST, THEN RANK
# ----------------------------------------------------------------------
test_final = test_c.copy()

# hard exclusions: fabricated profiles, duplicate rows, unrealistic notice period
excluded = test_final["is_fabricated"] | test_final["is_duplicate"] | (test_final["notice_days"] > 60)
test_final = test_final[~excluded].copy()

# score adjustment: boost sustained code contributors, penalize title inflation
adj = test_final["pred_score"].copy()
adj += test_final["sustained_contributor"] * 5.0
adj -= test_final["title_inflation_flag"] * 8.0
test_final["final_score"] = adj

test_final = test_final.sort_values("final_score", ascending=False)
shortlist = test_final.head(500).copy()
shortlist["rank"] = range(1, len(shortlist) + 1)

submission = shortlist[["rank", "candidate_id"]]
submission.to_csv("submission.csv", index=False)
print(f"[done] wrote submission.csv with {len(submission)} rows "
      f"(excluded {excluded.sum()} fabricated/duplicate/long-notice profiles from test.csv)")
