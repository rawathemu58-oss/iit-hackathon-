"""Ablation harness. Reports dev precision@150 AND a 5-fold internal CV
precision@5% on train (≈1000 positives total -> far less noisy than dev)."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import feats

SEED = 42
np.random.seed(SEED)

train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")
dev_w = set(pd.read_csv("dev_winners.csv")["candidate_id"])

tr = feats.build(train)
dv = feats.build(dev)


def prep(ref, other):
    o = feats.add_role_z(ref, other)
    return feats.add_pool_rank(o)


def fit_predict(tr_df, te_df, cols, seeds=(1, 2, 3), w=(0.4, 0.4, 0.2)):
    Xtr, ytr = tr_df[cols], tr_df["post_hire_score"]
    ybin = (ytr >= ytr.quantile(0.95)).astype(int)
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
    Xte = te_df[cols]
    Xte_s = sc.transform(imp.transform(Xte))
    r, c, g = [], [], []
    for s in seeds:
        m = HistGradientBoostingRegressor(random_state=s, max_iter=300)
        m.fit(Xtr, ytr); r.append(m.predict(Xte))
        m = HistGradientBoostingClassifier(random_state=s, max_iter=300,
                                           class_weight="balanced")
        m.fit(Xtr, ybin); c.append(m.predict_proba(Xte)[:, 1])
        m = Ridge(alpha=10.0); m.fit(Xtr_s, ytr); g.append(m.predict(Xte_s))
    rp = pd.Series(np.mean(r, 0)).rank(pct=True).values
    cp = pd.Series(np.mean(c, 0)).rank(pct=True).values
    gp = pd.Series(np.mean(g, 0)).rank(pct=True).values
    return w[0] * rp + w[1] * cp + w[2] * gp


def dev_precision(cols):
    a = prep(tr, tr)
    b = prep(tr, dv)
    s = fit_predict(a, b, cols)
    top = set(b.assign(s=s).nlargest(150, "s")["candidate_id"])
    return len(top & dev_w) / 150


def cv_precision(cols, n_splits=5):
    out = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_i, te_i in kf.split(tr):
        a_raw, b_raw = tr.iloc[tr_i], tr.iloc[te_i]
        a, b = prep(a_raw, a_raw), prep(a_raw, b_raw)
        s = fit_predict(a, b, cols)
        k = int(round(0.05 * len(b)))
        thr = b["post_hire_score"].quantile(0.95)
        truth = set(b[b["post_hire_score"] >= thr]["candidate_id"])
        top = set(b.assign(s=s).nlargest(k, "s")["candidate_id"])
        out.append(len(top & truth) / k)
    return float(np.mean(out)), float(np.std(out))


OLD = [
    "exp_years", "current_ctc_lpa", "expected_ctc_lpa", "aptitude_10", "rating_5",
    "notice_days", "last_change_yrs", "career_steps", "title_seniority",
    "kpi_met_bin", "enrolled_bin", "overtime_bin", "tech_100", "training_hours",
    "trainings_last_year", "num_employers", "note_pos", "award_count", "age",
    "yrs_since_grad", "tech_100_rz", "aptitude_10_rz", "rating_5_rz",
]

VARIANTS = {
    "A_old_featureset_fixed_parsing": OLD,
    "B_full_new_featureset": feats.FEATURES,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, cols in VARIANTS.items():
        if which != "all" and which not in name:
            continue
        d = dev_precision(cols)
        m, sd = cv_precision(cols)
        print(f"{name:34s} dev@150={d:.4f}   cv@5%={m:.4f} (+-{sd:.4f})", flush=True)

# --- extra variants appended ---
def bias_encoded(ref, other, cols=("inst_norm","current_city","recruitment_channel","degree","company_type","company_size")):
    other=other.copy(); gm=ref["post_hire_score"].mean()
    for c in cols:
        st=ref.groupby(c)["post_hire_score"].agg(["mean","count"])
        sm=(st["mean"]*st["count"]+gm*20)/(st["count"]+20)
        other[c+"_te"]=other[c].map(sm).fillna(gm)
    return other
