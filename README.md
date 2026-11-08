# Customer churn, CLV and retention targeting for a Singapore loyalty app

**Question:** Which at-risk customers should a Singapore retailer send a S$5 retention voucher to, and what is
the expected profit, measured honestly?

> **Data note:** There is no public customer-level transaction data for Singapore businesses, so this
> project uses a **synthetic** loyalty-app dataset (40,000 customers, ~2.75 years) with Singapore features:
> planning areas and regions, HDB/condo/landed housing, GST-inclusive prices (9%), CNY, 11.11/12.12
> sales, Great Singapore Sale and public-holiday seasonality. Distributions are illustrative
> and should be calibrated to SingStat. Because the data is simulated, the retention campaign is a
> **randomized experiment** with a known mechanism, so targeting rules can be evaluated without guessing.

## Pipeline

```
generate_data.py -> sql/features.sql (DuckDB) -> train_churn.py -> clv.py -> decision.py -> uplift.py
                                      \-> survival.csv -> r/survival.R
```

| Step | What it does |
|---|---|
| `make data` | Simulates customers, orders and a randomized campaign (`data/raw/`) |
| `make features` | SQL window-function features at 7 quarterly cutoffs; 90-day churn labels |
| `make train` | Logistic baseline vs LightGBM, time-based split, isotonic calibration, SHAP |
| `make clv` | BG/NBD + Gamma-Gamma 12-month CLV, validated on the next 90 days |
| `make decide` | Profit curves for 4 targeting rules, scored on the randomized test |
| `make uplift` | Cross-fitted T-learner uplift model, bootstrap confidence intervals |
| `make survival` | Kaplan-Meier + Cox models in R |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all            # about a few minutes
make survival       # needs R with the `survival` package
```

## Design choices to explain in interviews
- **Snapshot design:** features use orders before the cutoff; labels use the following 90 days. Train on
  the first five cutoffs, validate on the sixth, test on the last (2025-01-01), whose label windows
  never overlap the next cutoff's features.
- **"In play" population:** only customers who ordered in the previous 180 days are scored.
- **Calibration:** isotonic regression on the validation window, because probabilities feed dollar decisions.
- **Net of GST:** all revenue is divided by 1.09 before margin is applied.
- **Evaluation:** assumption-based profit chooses the cutoff (no outcome peeking); the randomized
  treated-vs-control difference measures realised incremental profit.

## Results
Fill in from your own run (numbers vary with the seed and assumptions in `src/config.py`). In the
author's run: LightGBM reached ROC-AUC ~0.86; ranking by P(churn) alone *lost* money in the randomized
test because it targets customers who are already gone; the uplift rule modestly beat "target
everyone" but with overlapping 95% confidence intervals. Report your own numbers with intervals.

## Limitations
Synthetic data; one campaign, one 90-day outcome window; CLV x P(churn) partly double counts dropout;
margin and cost are assumptions; the uplift signal is modest and noisy.

## Replacing the synthetic data
Keep the same schemas (`customers`, `orders`, `campaign`) and everything downstream works. Mind PDPA:
anonymise IDs and avoid precise addresses; planning area is the finest geography you need.
