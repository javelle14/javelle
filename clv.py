"""
Customer lifetime value with BG/NBD (purchase frequency + dropout) and Gamma-Gamma (spend).
Fitted ONLY on orders before the test cutoff, then validated on the next 90 days.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from .config import TEST_CUTOFF, GST, GROSS_MARGIN

RAW, OUT = Path("data/raw"), Path("data/processed")


def main() -> None:
    cutoff = pd.Timestamp(TEST_CUTOFF)
    o = pd.read_parquet(RAW / "orders.parquet")
    o["order_date"] = pd.to_datetime(o["order_date"])
    o = o[o.order_date < cutoff]                                    # no peeking past the cutoff
    o["net_amt"] = o.amount_sgd / GST

    rfm = summary_data_from_transaction_data(
        o, "customer_id", "order_date", monetary_value_col="net_amt",
        observation_period_end=cutoff - pd.Timedelta(days=1), freq="D")

    bgf = BetaGeoFitter(penalizer_coef=0.001).fit(rfm.frequency, rfm.recency, rfm["T"])
    print(bgf.summary.round(3))

    rep = rfm[rfm.frequency > 0]
    print("freq vs monetary corr (Gamma-Gamma assumes ~0):", round(rep[["frequency", "monetary_value"]].corr().iloc[0, 1], 3))
    ggf = GammaGammaFitter(penalizer_coef=0.01).fit(rep.frequency, rep.monetary_value)

    rfm["p_alive"] = bgf.conditional_probability_alive(rfm.frequency, rfm.recency, rfm["T"])
    rfm["exp_orders_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(90, rfm.frequency, rfm.recency, rfm["T"])
    rfm["exp_aov"] = np.where(rfm.frequency > 0,
                              ggf.conditional_expected_average_profit(rfm.frequency, rfm.monetary_value),
                              o.groupby("customer_id").net_amt.mean().reindex(rfm.index))
    # 12-month discounted CLV in net-of-GST SGD, then converted to gross margin
    rfm["clv_12m_rev"] = ggf.customer_lifetime_value(bgf, rfm.frequency, rfm.recency, rfm["T"], rfm.monetary_value,
                                                     time=12, freq="D", discount_rate=0.01)
    rfm.loc[rfm.frequency == 0, "clv_12m_rev"] = (rfm.exp_orders_90d * 4 * rfm.exp_aov)[rfm.frequency == 0]
    rfm["clv_12m_margin"] = rfm.clv_12m_rev * GROSS_MARGIN
    rfm.reset_index().to_parquet(OUT / "clv_test.parquet", index=False)

    # ---- validation: predicted vs actual orders in the next 90 days, by decile ----
    snap = pd.read_parquet(OUT / "snapshots.parquet")
    snap = snap[pd.to_datetime(snap.cutoff) == cutoff].set_index("customer_id")
    chk = rfm.join(snap[["fut_orders_90d", "campaign_treated"]], how="inner")
    chk = chk[chk.campaign_treated == 0]                            # untreated = natural behaviour
    chk["decile"] = pd.qcut(chk.exp_orders_90d, 10, duplicates="drop")
    print(chk.groupby("decile", observed=True)[["exp_orders_90d", "fut_orders_90d"]].mean().round(2))


if __name__ == "__main__":
    main()
