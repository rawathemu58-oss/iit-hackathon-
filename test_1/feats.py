"""Robust parsing + feature engineering for the Corporate Heist problem."""
import re
import numpy as np
import pandas as pd

CUR_YEAR = 2026

# ---------------- numeric parsers ----------------

def parse_experience(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(">", "").replace("+", "").strip()
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "month" in s or "mo" == s[-2:]:
        val /= 12.0
    return val


def parse_ctc(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("₹", "").replace(",", "").replace("inr", "").strip()
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "cr" in s:
        return val * 100.0
    if "lpa" in s or "lakh" in s or "lac" in s or s.endswith("l"):
        return val
    if val > 10000:
        return val / 100000.0
    return val


def parse_score_100(x):
    """technical_assessment: handles 71, '45/100', '40.0 %', 0.45, 'absent'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if any(k in s for k in ("absent", "not taken", "n/a", "na", "-")) and not re.search(r"\d", s):
        return np.nan
    if "/" in s:
        m = re.match(r"\s*([\d.]+)\s*/\s*([\d.]+)", s)
        if m:
            den = float(m.group(2))
            return float(m.group(1)) / den * 100.0 if den else np.nan
    if "%" in s:
        m = re.search(r"([\d.]+)", s)
        return float(m.group(1)) if m else np.nan
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if val <= 1.0:            # stored as a 0-1 fraction
        return val * 100.0
    return val


def parse_aptitude(x):
    """-> 0-10 scale. Handles 7.3, '5.7/10', '86%', 0.73, 73."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "%" in s:
        m = re.search(r"([\d.]+)", s)
        return float(m.group(1)) / 10.0 if m else np.nan
    if "/" in s:
        m = re.match(r"\s*([\d.]+)\s*/\s*([\d.]+)", s)
        if m:
            den = float(m.group(2))
            return float(m.group(1)) / den * 10.0 if den else np.nan
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if val > 10:              # entered on a 0-100 scale
        return val / 10.0
    return val


RATING_WORDS = {
    "outstanding": 5.0, "exceptional": 5.0,
    "exceeds expectations": 4.0, "exceeds": 4.0, "above average": 4.0,
    "meets expectations": 3.0, "meets": 3.0, "average": 3.0,
    "needs improvement": 2.0, "below average": 2.0,
    "unsatisfactory": 1.0, "poor": 1.0,
}


def parse_rating(x):
    """-> 1-5. Handles 4, '4/5', '7.8/10', 7.8, 'Exceeds Expectations'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    for k, v in RATING_WORDS.items():
        if k in s:
            return v
    if "not rated" in s or "new joiner" in s:
        return np.nan
    if "/" in s:
        m = re.match(r"\s*([\d.]+)\s*/\s*([\d.]+)", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            return num / den * 5.0 if den else np.nan
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if val > 5:               # 0-10 scale
        return val / 2.0
    return val


def parse_notice(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if "immediate" in s or "available now" in s:
        return 0.0
    m = re.search(r"([\d.]+)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "month" in s:
        val *= 30
    if "week" in s:
        val *= 7
    return val


def parse_last_job_change(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    if s == "never":
        return np.nan
    if ">" in s:
        m = re.search(r"([\d.]+)", s)
        return float(m.group(1)) + 1 if m else np.nan
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else np.nan


TRUE_TOKENS = {"y", "yes", "1", "true", "t"}
FALSE_TOKENS = {"n", "no", "0", "false", "f"}


def parse_bool(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in TRUE_TOKENS:
        return 1.0
    if s in FALSE_TOKENS:
        return 0.0
    return np.nan


def parse_awards(x):
    """'0' / '-' / nan -> 0 ; '1 award' -> 1 ; 'Yes - Spot Award' -> 1."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip().lower()
    if s in ("", "-", "none", "nil", "0", "no"):
        return 0.0
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return 1.0               # any non-empty textual award name


def parse_code_contrib(x):
    """test.csv only. '~3', '10+', '3 merged PRs', 'not tracked', nan."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "not tracked" in s or s in ("", "-", "n/a", "na", "unknown"):
        return np.nan
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else np.nan


# ---------------- career_path ----------------
SPLIT_RE = re.compile(r"\s*(?:->|→|>|\|)\s*")
DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mo|month|y|yr|year)", re.I)


def parse_career(path):
    """-> (n_steps, total_months, mean_months, last_months, max_months)."""
    if pd.isna(path):
        return (0, np.nan, np.nan, np.nan, np.nan)
    parts = [p for p in SPLIT_RE.split(str(path)) if p.strip()]
    durs = []
    for p in parts:
        m = DUR_RE.search(p)
        if m:
            v = float(m.group(1))
            unit = m.group(2).lower()
            durs.append(v if unit.startswith("mo") else v * 12.0)
    if not durs:
        return (len(parts), np.nan, np.nan, np.nan, np.nan)
    return (len(parts), float(np.sum(durs)), float(np.mean(durs)),
            durs[-1], float(np.max(durs)))


SENIOR_4 = ("head of", "chief", "vp ", "vice president", "director", "cto", "ceo")
SENIOR_3 = ("principal", "staff", "lead", "manager", "architect")
SENIOR_2 = ("senior", "sr.", "sr ", " ii", " iii", "specialist")
SENIOR_0 = ("junior", "jr.", "trainee", "intern", "apprentice", "associate", " i ")


def title_seniority(title):
    if pd.isna(title):
        return np.nan
    t = " " + str(title).lower() + " "
    if any(k in t for k in SENIOR_4):
        return 4
    if any(k in t for k in SENIOR_3):
        return 3
    if any(k in t for k in SENIOR_2):
        return 2
    if any(k in t for k in SENIOR_0):
        return 0
    return 1


POS_NOTE = [
    "shipped", "used by", "on-call owner", "primary on-call", "mentored", "mentors",
    "led the migration", "runbooks", "strong communicator", "owns", "ownership",
    "ahead of sprint", "excellent system-design", "excellent system design",
    "raised the bar", "go-to", "proactive",
]
NEG_NOTE = [
    "needed frequent guidance", "frequent guidance", "missed deadline", "struggled",
    "performance concern", "lukewarm", "concerns about", "hard to reach",
    "declined to", "no-show", "did not complete",
]
# recruiter-referral / pedigree flattery: old-panel bias, deliberately NOT scored
NEUTRAL_NOTE = ["cricket", "hybrid", "weekend shifts", "relocat", "marathon", "chess"]


def note_pos(x):
    if pd.isna(x):
        return 0
    s = str(x).lower()
    return sum(w in s for w in POS_NOTE)


def note_neg(x):
    if pd.isna(x):
        return 0
    s = str(x).lower()
    return sum(w in s for w in NEG_NOTE)


def count_items(x):
    if pd.isna(x) or str(x).strip() in ("", "-", "none"):
        return 0
    return len([p for p in re.split(r"[;,|]", str(x)) if p.strip()])


def norm_inst(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------- main builder ----------------

def build(df):
    d = df.copy()
    d["exp_years"] = d["total_experience"].map(parse_experience)
    d["current_ctc_lpa"] = d["current_ctc"].map(parse_ctc)
    d["expected_ctc_lpa"] = d["expected_ctc"].map(parse_ctc)
    d["tech_100"] = d["technical_assessment"].map(parse_score_100)
    d["aptitude_10"] = d["aptitude_score"].map(parse_aptitude)
    d["rating_5"] = d["last_rating"].map(parse_rating)
    d["notice_days"] = d["notice_period"].map(parse_notice)
    d["last_change_yrs"] = d["last_job_change"].map(parse_last_job_change)
    d["kpi_met_bin"] = d["kpi_met"].map(parse_bool)
    d["overtime_bin"] = d["overtime_history"].map(parse_bool)
    d["award_count"] = d["awards"].map(parse_awards)
    d["enrolled_bin"] = (
        d["currently_enrolled"].astype(str).str.lower().ne("no_enrollment")
    ).astype(int)
    d["training_hours"] = pd.to_numeric(d["training_hours"], errors="coerce")
    d["trainings_last_year"] = pd.to_numeric(d["trainings_last_year"], errors="coerce")
    d["num_employers"] = pd.to_numeric(d["num_employers"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["graduation_year"] = pd.to_numeric(d["graduation_year"], errors="coerce")

    car = d["career_path"].map(parse_career)
    d["career_steps"] = [c[0] for c in car]
    d["career_months"] = [c[1] for c in car]
    d["career_mean_mo"] = [c[2] for c in car]
    d["career_last_mo"] = [c[3] for c in car]
    d["career_max_mo"] = [c[4] for c in car]

    d["title_seniority"] = d["current_title"].map(title_seniority)
    d["note_pos"] = d["recruiter_note"].map(note_pos)
    d["note_neg"] = d["recruiter_note"].map(note_neg)
    d["note_len"] = d["recruiter_note"].fillna("").astype(str).str.len()
    d["n_skills"] = d["skills"].map(count_items)
    d["n_certs"] = d["certifications"].map(count_items)

    # derived ratios
    d["ctc_hike"] = d["expected_ctc_lpa"] / d["current_ctc_lpa"].replace(0, np.nan)
    d["ctc_per_year_exp"] = d["current_ctc_lpa"] / d["exp_years"].replace(0, np.nan)
    d["exp_per_employer"] = d["exp_years"] / d["num_employers"].replace(0, np.nan)
    d["yrs_since_grad"] = CUR_YEAR - d["graduation_year"]
    # career gap: declared experience vs months actually accounted for in path
    d["career_gap_yrs"] = d["yrs_since_grad"] - d["career_months"] / 12.0
    d["exp_vs_path"] = d["exp_years"] - d["career_months"] / 12.0
    d["seniority_per_year"] = d["title_seniority"] / d["exp_years"].replace(0, np.nan)
    d["tech_x_apt"] = d["tech_100"] * d["aptitude_10"]

    d["inst_norm"] = d["institute"].map(norm_inst)
    return d


ROLE_Z_COLS = ("tech_100", "aptitude_10", "rating_5", "current_ctc_lpa", "exp_years")


def add_role_z(ref, other, cols=ROLE_Z_COLS):
    """Standardise within applied_role using ref's statistics."""
    other = other.copy()
    stats = ref.groupby("applied_role")[list(cols)].agg(["mean", "std"])
    for c in cols:
        m = other["applied_role"].map(stats[(c, "mean")])
        sd = other["applied_role"].map(stats[(c, "std")]).replace(0, np.nan)
        other[c + "_rz"] = (other[c] - m) / sd
    return other


def add_pool_rank(df, cols=("tech_100", "aptitude_10", "rating_5")):
    """Within-pool, within-role percentile: 'topped the technical assessment'
    is a relative statement about this cycle's pool, not an absolute score."""
    df = df.copy()
    for c in cols:
        df[c + "_prank"] = df.groupby("applied_role")[c].rank(pct=True)
    return df


FEATURES = [
    "exp_years", "current_ctc_lpa", "expected_ctc_lpa", "aptitude_10", "rating_5",
    "tech_100", "notice_days", "last_change_yrs", "career_steps", "career_months",
    "career_mean_mo", "career_last_mo", "career_max_mo", "title_seniority",
    "kpi_met_bin", "enrolled_bin", "overtime_bin", "training_hours",
    "trainings_last_year", "num_employers", "note_pos", "note_neg", "note_len",
    "n_skills", "n_certs", "award_count", "age", "yrs_since_grad",
    "ctc_hike", "ctc_per_year_exp", "exp_per_employer", "career_gap_yrs",
    "exp_vs_path", "seniority_per_year", "tech_x_apt",
    "tech_100_rz", "aptitude_10_rz", "rating_5_rz", "current_ctc_lpa_rz", "exp_years_rz",
    "tech_100_prank", "aptitude_10_prank", "rating_5_prank",
]