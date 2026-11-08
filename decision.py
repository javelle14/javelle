"""
Decision layer: who should we send the S$5 retention voucher to?

Two ways to value a targeting rule:
  1. ASSUMPTION-based: sum(SAVE_RATE * P(churn) * 12m margin CLV - cost).  Uses no outcomes, so it
     is what you'd use in real life to choose the cutoff.
  2. EXPERIMENT-based: the campaign was randomized, so within any targeted set the treated-vs-control
     difference in 90-day margin is an unbiased estimate of incremental profit.

Simplification to flag in the write-up: P(churn) x CLV partly double-counts (BG/NBD CLV already
discounts for dropout). The uplift model in src/uplift.py avoids this by modelling incremental
profit directly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import GROSS_MARGIN, COST_PER_TARGET, SAVE_RATE

OUT, REP = Path("data/processed"), Path("reports")


def rct_profit(d: pd.DataFrame, mask: np.ndarray) -> float:
    s = d[mask]
    t, c = s[s.campaign_treated == 1], s[s.campaign_treated == 0]
    if len(t) < 30 or len(c) < 30:
        return np.nan
    inc = (t.margin_90d.mean() - c.margin_90d.mean())
    return inc * len(s) - COST_PER_TARGET * len(s)


def main() -> None:
    te = pd.read_parquet(OUT / "test_scored.parquet")
    clv = pd.read_parquet(OUT / "clv_test.parquet")[["customer_id", "clv_12m_margin", "p_alive"]]
    d = te.merge(clv, on="customer_id", how="left").dropna(subset=["clv_12m_margin"]).reset_index(drop=True)
    d["margin_90d"] = d.fut_net_rev_90d * GROSS_MARGIN
    d["value_at_risk"] = d.p_churn * d.clv_12m_margin
    d["exp_net_benefit"] = SAVE_RATE * d.value_at_risk - COST_PER_TARGET

    # simple RFM heuristic: valuable (F, M) AND lapsing (R)
    r = lambda col: d[col].rank(pct=True)
    d["rfm_score"] = r("n_orders_all") + r("net_rev_all") + r("recency_days")

    strategies = {"Model: P(churn) x CLV": d.value_at_risk, "RFM heuristic": d.rfm_score,
                  "P(churn) only": d.p_churn, "Random": pd.Series(np.random.default_rng(0).random(len(d)))}
    rows = []
    for name, score in strategies.items():
        order = score.rank(ascending=False, method="first")
        for k in np.arange(0.05, 1.0001, 0.05):
            m = (order <= k * len(d)).values
            rows.append({"strategy": name, "pct_targeted": round(k, 2), "n_targeted": int(m.sum()),
                         "profit_rct": rct_profit(d, m),
                         "profit_assumed": (SAVE_RATE * d.value_at_risk[m] - COST_PER_TARGET).sum()})
    curves = pd.DataFrame(rows)
    curves.to_csv(REP / "profit_curves.csv", index=False)

    # choose cutoff with the assumption-based curve (no outcome peeking), report experiment result
    m_ = curves[curves.strategy == "Model: P(churn) x CLV"]
    best = m_.loc[m_.profit_assumed.idxmax()]
    all_rct = curves[(curves.strategy == "Random") & (curves.pct_targeted == 1.0)].profit_rct.iloc[0]
    print(f"Chosen cutoff: top {best.pct_targeted:.0%} ({int(best.n_targeted):,} customers)")
    print(f"  assumed profit  S${best.profit_assumed:,.0f}")
    print(f"  RCT-measured    S${best.profit_rct:,.0f}")
    print(f"  Target everyone S${all_rct:,.0f} (RCT-measured)")
    for name in strategies:
        sub = curves[(curves.strategy == name) & (curves.pct_targeted.isin([0.2, 0.4]))]
        print(f"{name:24s}", {f"top {r.pct_targeted:.0%}": f"S${r.profit_rct:,.0f}" for r in sub.itertuples()})

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for name in strategies:
        s = curves[curves.strategy == name]
        ax[0].plot(s.pct_targeted * 100, s.profit_rct, label=name)
    ax[0].set(title="Incremental profit (randomized test), S$", xlabel="% of at-risk base targeted")
    ax[0].axhline(0, c="grey", lw=0.8); ax[0].legend(fontsize=8)
    ax[1].plot(m_.pct_targeted * 100, m_.profit_assumed, label="assumed (model)")
    ax[1].plot(m_.pct_targeted * 100, m_.profit_rct, label="measured (RCT)")
    ax[1].set(title="Model rule: assumed vs measured", xlabel="% targeted"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(REP / "figures" / "profit_curve.png", dpi=150); plt.close()

    # Tableau-ready export
    rank = d.value_at_risk.rank(ascending=False, method="first")
    d["target_flag"] = (rank <= best.pct_targeted * len(d)).astype(int)
    d["risk_tier"] = pd.qcut(d.p_churn, 3, labels=["Low", "Medium", "High"])
    d["value_tier"] = pd.qcut(d.clv_12m_margin, 3, labels=["Low", "Medium", "High"])
    cols = ["customer_id", "planning_area", "region", "housing_type", "age_band", "acquisition_channel",
            "p_churn", "p_alive", "clv_12m_margin", "value_at_risk", "exp_net_benefit", "risk_tier",
            "value_tier", "target_flag", "campaign_treated", "margin_90d"]
    d[cols].to_csv(REP / "tableau" / "customer_scores.csv", index=False)
    curves.to_csv(REP / "tableau" / "profit_curves.csv", index=False)


if __name__ == "__main__":
    main()
