"""
Stretch goal: uplift modelling (who is *persuadable*, not just who is at risk?).

Cross-fitted T-learner on the randomized campaign:
  tau(x) = E[margin_90d | treated, x] - E[margin_90d | control, x]
Target a customer only if predicted uplift > cost per target. Because each customer is scored by
a model that never saw them (2-fold cross-fitting), the randomized treated-vs-control comparison
on the targeted set is an honest estimate of realised profit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from .config import GROSS_MARGIN, COST_PER_TARGET
from .train_churn import CAT, NUM

OUT, REP = Path("data/processed"), Path("reports")


def make_x(d, cats):
    x = d[NUM + CAT].copy()
    for c in CAT:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    return x


def rct_profit(d, mask):
    s = d[mask]
    t, c = s[s.campaign_treated == 1], s[s.campaign_treated == 0]
    return (t.margin_90d.mean() - c.margin_90d.mean() - COST_PER_TARGET) * len(s)


def bootstrap_ci(d, mask, n=300, seed=0):
    rng = np.random.default_rng(seed)
    s = d[mask]
    vals = [rct_profit(s.iloc[rng.integers(0, len(s), len(s))].reset_index(drop=True),
                       np.ones(len(s), bool)) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def main() -> None:
    d = pd.read_parquet(OUT / "test_scored.parquet").dropna(subset=["campaign_treated"]).reset_index(drop=True)
    d["margin_90d"] = d.fut_net_rev_90d * GROSS_MARGIN
    # lower-variance TRAINING target: orders x the customer's own pre-cutoff order value x margin.
    # (Evaluation below still uses the real, noisy margin_90d.)
    d["y_train"] = d.fut_orders_90d * d.avg_order_value * GROSS_MARGIN
    cats = {c: sorted(d[c].unique()) for c in CAT}
    params = dict(n_estimators=250, learning_rate=0.03, num_leaves=8, min_child_samples=80,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

    d["tau_hat"] = np.nan
    for tr_i, te_i in KFold(2, shuffle=True, random_state=0).split(d):
        tr, te = d.iloc[tr_i], d.iloc[te_i]
        m1 = lgb.LGBMRegressor(**params).fit(make_x(tr[tr.campaign_treated == 1], cats), tr[tr.campaign_treated == 1].y_train)
        m0 = lgb.LGBMRegressor(**params).fit(make_x(tr[tr.campaign_treated == 0], cats), tr[tr.campaign_treated == 0].y_train)
        d.loc[te.index, "tau_hat"] = m1.predict(make_x(te, cats)) - m0.predict(make_x(te, cats))

    rows = []
    order = d.tau_hat.rank(ascending=False, method="first")
    for k in np.arange(0.05, 1.0001, 0.05):
        rows.append({"pct_targeted": round(k, 2), "profit_rct": rct_profit(d, (order <= k * len(d)).values)})
    pd.DataFrame(rows).to_csv(REP / "uplift_profit_curve.csv", index=False)

    rule = (d.tau_hat > COST_PER_TARGET).values
    all_ = np.ones(len(d), bool)
    for name, m in [("Uplift rule (tau_hat > cost)", rule), ("Target everyone", all_)]:
        lo, hi = bootstrap_ci(d, m)
        print(f"{name:30s} n={m.sum():>6,}  RCT profit S${rct_profit(d, m):>8,.0f}  95% CI [{lo:,.0f}, {hi:,.0f}]")

    # sanity check vs ground truth (only possible because data is synthetic)
    truth = pd.read_parquet("data/raw/_truth_do_not_use_in_features.parquet")[["customer_id", "price_sens_true", "lambda_true"]]
    d = d.merge(truth, on="customer_id")
    d["tau_proxy"] = d.lambda_true * (0.05 + 0.9 * d.price_sens_true)   # true uplift is ~ this x basket
    print("corr(tau_hat, true uplift proxy):", round(d.tau_hat.corr(d.tau_proxy), 3))
    print("corr(p_churn, true uplift proxy):", round(d.p_churn.corr(d.tau_proxy), 3), "(risk != persuadability)")
    d[["customer_id", "tau_hat"]].to_parquet(OUT / "uplift_scores.parquet", index=False)


if __name__ == "__main__":
    main()
