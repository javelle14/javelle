"""Churn models: logistic baseline vs LightGBM, time-based split, calibrated probabilities."""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import TRAIN_CUTOFFS, VALID_CUTOFF, TEST_CUTOFF

OUT, FIG = Path("data/processed"), Path("reports/figures")
CAT = ["planning_area", "region", "housing_type", "age_band", "acquisition_channel"]
NUM = ["recency_days", "n_orders_all", "n_orders_30d", "n_orders_90d", "n_orders_180d",
       "net_rev_all", "net_rev_90d", "net_rev_180d", "avg_order_value", "order_trend_ratio",
       "avg_gap_days", "overdue_ratio", "gap_std", "voucher_share", "app_share", "tenure_days"]
TARGET = "churn_90d"   # NEVER add fut_* or campaign_* columns to the feature list


def metrics(y, p, name):
    return {"model": name, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def main() -> None:
    df = pd.read_parquet(OUT / "snapshots.parquet")
    df["cutoff"] = pd.to_datetime(df["cutoff"]).dt.strftime("%Y-%m-%d")
    tr = df[df.cutoff.isin(TRAIN_CUTOFFS)]
    va = df[df.cutoff == VALID_CUTOFF]
    te = df[df.cutoff == TEST_CUTOFF].copy()
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,} | base churn rate (test) {te[TARGET].mean():.1%}")

    # --- baseline: logistic regression ---
    pre = ColumnTransformer([("num", make_pipeline(_impute(), StandardScaler()), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    logit = make_pipeline(pre, LogisticRegression(max_iter=1000))
    logit.fit(tr[NUM + CAT], tr[TARGET])

    # --- LightGBM ---
    def prep(d):
        x = d[NUM + CAT].copy()
        for c in CAT:
            x[c] = pd.Categorical(x[c], categories=sorted(df[c].unique()))
        return x
    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    gbm.fit(prep(tr), tr[TARGET], eval_set=[(prep(va), va[TARGET])],
            callbacks=[lgb.early_stopping(50, verbose=False)])

    # --- calibration on the validation window (probabilities feed dollar decisions) ---
    iso = IsotonicRegression(out_of_bounds="clip").fit(gbm.predict_proba(prep(va))[:, 1], va[TARGET])
    te["p_churn_logit"] = logit.predict_proba(te[NUM + CAT])[:, 1]
    te["p_churn_raw"] = gbm.predict_proba(prep(te))[:, 1]
    te["p_churn"] = iso.predict(te["p_churn_raw"])

    res = pd.DataFrame([metrics(te[TARGET], te.p_churn_logit, "logistic"),
                        metrics(te[TARGET], te.p_churn_raw, "lightgbm"),
                        metrics(te[TARGET], te.p_churn, "lightgbm+isotonic")]).round(4)
    print(res.to_string(index=False))
    res.to_csv("reports/model_metrics.csv", index=False)

    # calibration plot
    te["bin"] = pd.qcut(te.p_churn, 10, duplicates="drop")
    cal = te.groupby("bin", observed=True).agg(pred=("p_churn", "mean"), actual=(TARGET, "mean"))
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.plot(cal.pred, cal.actual, "o-")
    plt.xlabel("Predicted P(churn)"); plt.ylabel("Observed churn rate"); plt.title("Calibration (test)")
    plt.tight_layout(); plt.savefig(FIG / "calibration.png", dpi=150); plt.close()

    # SHAP (optional)
    try:
        import shap
        sv = shap.TreeExplainer(gbm).shap_values(prep(te.sample(3000, random_state=0)))
        sv = sv[1] if isinstance(sv, list) else sv
        shap.summary_plot(sv, prep(te.sample(3000, random_state=0)), show=False, max_display=12)
        plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:  # noqa: BLE001
        print("SHAP skipped:", e)

    te.drop(columns="bin").to_parquet(OUT / "test_scored.parquet", index=False)


def _impute():
    from sklearn.impute import SimpleImputer
    return SimpleImputer(strategy="median")


if __name__ == "__main__":
    main()
