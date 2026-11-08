"""
Synthetic Singapore loyalty-app dataset (grocery / lifestyle retailer).

Ground truth is a Pareto/NBD-style process: each customer has a purchase rate
and a latent lifetime, with Singapore seasonality (CNY, double-day sales, GSS,
public holidays). A randomized retention-voucher campaign runs on CAMPAIGN_DATE
with heterogeneous true uplift (price-sensitive customers respond more).

DISTRIBUTIONS ARE ILLUSTRATIVE. Calibrate planning-area and housing shares to
SingStat tables before publishing, and say clearly in the README the data is
synthetic.
"""
import numpy as np
import pandas as pd
import holidays
from pathlib import Path
from .config import START, END, CAMPAIGN_DATE

RAW = Path("data/raw")
SEED, N = 42, 40_000
rng = np.random.default_rng(SEED)

AREAS = {  # planning area -> (region, weight)  [illustrative]
    "Tampines": ("East", 6), "Bedok": ("East", 6), "Pasir Ris": ("East", 3),
    "Sengkang": ("North-East", 6), "Hougang": ("North-East", 5), "Punggol": ("North-East", 5),
    "Serangoon": ("North-East", 2), "Jurong West": ("West", 6), "Clementi": ("West", 3),
    "Bukit Batok": ("West", 3), "Woodlands": ("North", 6), "Yishun": ("North", 5),
    "Sembawang": ("North", 3), "Choa Chu Kang": ("West", 4), "Ang Mo Kio": ("North-East", 4),
    "Toa Payoh": ("Central", 3), "Bishan": ("Central", 2), "Queenstown": ("Central", 3),
    "Bukit Merah": ("Central", 4), "Kallang": ("Central", 3), "Geylang": ("Central", 3),
    "Novena": ("Central", 1), "Bukit Timah": ("Central", 1), "Marine Parade": ("Central", 2),
}
HOUSING = ["HDB 1-3rm", "HDB 4rm", "HDB 5rm/EA", "Condo/Private apt", "Landed"]
HOUSING_W = [0.10, 0.30, 0.28, 0.25, 0.07]
BASKET_MULT = [0.8, 0.9, 1.0, 1.2, 1.4]
AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_W = [0.18, 0.32, 0.24, 0.16, 0.10]
CHANNELS = ["Organic", "Referral", "Social ads", "Search ads", "Partner bank promo"]
CHAN_W = [0.30, 0.18, 0.22, 0.15, 0.15]


def seasonal_index(days: pd.DatetimeIndex) -> np.ndarray:
    sg = holidays.Singapore(years=range(START.year, END.year + 1))
    s = np.ones(len(days))
    s *= np.where(days.dayofweek >= 5, 1.15, 1.0)                          # weekends
    s *= np.where(days.month.isin([6, 7]), 1.15, 1.0)                      # Great Singapore Sale-ish
    s *= np.where((days.month == 12) & (days.day >= 15), 1.2, 1.0)         # year-end
    s *= np.where((days.month >= 9) & (days.month == days.day), 1.8, 1.0)  # 9.9 / 10.10 / 11.11 / 12.12
    for cny in ["2023-01-22", "2024-02-10", "2025-01-29"]:                 # CNY run-up
        cny = pd.Timestamp(cny)
        s *= np.where((days >= cny - pd.Timedelta(days=14)) & (days <= cny), 1.4, 1.0)
    s *= np.array([1.1 if d.date() in sg else 1.0 for d in days])          # public holidays
    return s


def simulate() -> None:
    days = pd.date_range(START, END)
    T = len(days)
    s = seasonal_index(days)
    smax = s.max()

    # ---------- customers ----------
    w = np.array([v[1] for v in AREAS.values()], dtype=float)
    area = rng.choice(list(AREAS), N, p=w / w.sum())
    housing_i = rng.choice(len(HOUSING), N, p=HOUSING_W)
    age = rng.choice(AGES, N, p=AGE_W)
    chan = rng.choice(CHANNELS, N, p=CHAN_W)
    signup_idx = rng.integers(0, T - 30, N)

    price_sens = rng.beta(2, 3, N)
    price_sens = price_sens + np.where(age == "18-24", 0.15, 0) \
        + np.where(np.isin(chan, ["Social ads", "Partner bank promo"]), 0.10, 0)
    price_sens = np.clip(price_sens, 0, 1)

    lam = rng.gamma(1.2, 0.035, N)  # orders/day while alive (mean ~0.042)
    lam = lam * np.where(chan == "Referral", 1.2, 1.0) * np.where(chan == "Partner bank promo", 0.85, 1.0)
    life = rng.exponential(400 * (1.3 - 0.6 * price_sens))  # days alive after signup
    basket = 30 * np.array(BASKET_MULT)[housing_i] * rng.lognormal(0, 0.25, N)
    p_instore = np.where(np.isin(age, ["45-54", "55+"]), 0.6, 0.3)
    death_idx = signup_idx + life

    # ---------- baseline orders (thinned non-homogeneous Poisson) ----------
    cust_ids, order_idx = [], []
    for i in range(N):
        horizon = min(T - 1, death_idx[i])
        span = horizon - signup_idx[i]
        t = [signup_idx[i]]                                     # joins with a first order
        if span > 0:
            k = rng.poisson(lam[i] * smax * span)
            cand = signup_idx[i] + rng.uniform(0, span, k)
            keep = rng.random(k) < s[np.minimum(cand.astype(int), T - 1)] / smax
            t.extend(cand[keep])
        t = np.floor(t).astype(int)
        cust_ids.append(np.full(len(t), i))
        order_idx.append(t)
    orders = pd.DataFrame({"cust": np.concatenate(cust_ids), "idx": np.concatenate(order_idx)})
    orders["campaign_id"] = None

    # ---------- randomized retention campaign ----------
    camp_idx = int((CAMPAIGN_DATE - START).days)
    pre = orders[orders.idx < camp_idx].groupby("cust").idx.max()
    elig = pre[pre >= camp_idx - 180].index.values              # "in play" customers
    treated = rng.random(len(elig)) < 0.5
    campaign = pd.DataFrame({"customer_id": [f"C{c:06d}" for c in elig], "campaign_id": "RET_2025Q1",
                             "campaign_date": CAMPAIGN_DATE.date(), "treated": treated.astype(int),
                             "voucher_value_sgd": 5.0})
    extra_c, extra_i = [], []
    for c in elig[treated]:
        span = min(30, death_idx[c] - camp_idx)                 # only living customers respond
        if span <= 0:
            continue
        boost = 0.05 + 0.9 * price_sens[c]                      # TRUE heterogeneous uplift
        k = rng.poisson(lam[c] * boost * span)
        extra_c += [c] * k
        extra_i += list(np.floor(camp_idx + rng.uniform(0, span, k)).astype(int))
    extra = pd.DataFrame({"cust": extra_c, "idx": extra_i, "campaign_id": "RET_2025Q1"})
    orders = pd.concat([orders, extra], ignore_index=True)
    orders = orders[orders.idx < T].reset_index(drop=True)

    # ---------- order attributes ----------
    c = orders.cust.values
    orders["order_date"] = days[orders.idx.values].date
    orders["order_channel"] = np.where(rng.random(len(orders)) < p_instore[c], "in_store", "app")
    orders["amount_sgd"] = np.round(basket[c] * rng.lognormal(0, 0.35, len(orders)), 2)  # GST-inclusive
    orders["voucher_used"] = np.where(orders.campaign_id.notna(), 1,
                                      (rng.random(len(orders)) < 0.05 + 0.25 * price_sens[c]).astype(int))
    orders = orders.sort_values(["cust", "order_date"]).reset_index(drop=True)
    orders["customer_id"] = [f"C{x:06d}" for x in orders.cust]
    orders["order_id"] = [f"O{x:08d}" for x in range(len(orders))]
    orders = orders[["order_id", "customer_id", "order_date", "order_channel", "amount_sgd",
                     "voucher_used", "campaign_id"]]

    customers = pd.DataFrame({
        "customer_id": [f"C{x:06d}" for x in range(N)],
        "signup_date": days[signup_idx].date, "planning_area": area,
        "region": [AREAS[a][0] for a in area], "housing_type": [HOUSING[i] for i in housing_i],
        "age_band": age, "acquisition_channel": chan})
    truth = pd.DataFrame({"customer_id": customers.customer_id, "lambda_true": lam,
                          "price_sens_true": price_sens,
                          "death_date": (START + pd.to_timedelta(death_idx, unit="D")).date})

    RAW.mkdir(parents=True, exist_ok=True)
    customers.to_parquet(RAW / "customers.parquet", index=False)
    orders.to_parquet(RAW / "orders.parquet", index=False)
    campaign.to_parquet(RAW / "campaign.parquet", index=False)
    truth.to_parquet(RAW / "_truth_do_not_use_in_features.parquet", index=False)
    print(f"customers={len(customers):,} orders={len(orders):,} campaign rows={len(campaign):,}")


if __name__ == "__main__":
    simulate()
