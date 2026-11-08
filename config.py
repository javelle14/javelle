"""Single source of truth for dates and business assumptions."""
import pandas as pd

# ---- data window ----
START = pd.Timestamp("2022-07-01")
END = pd.Timestamp("2025-03-31")            # last observed day (inclusive)
CAMPAIGN_DATE = pd.Timestamp("2025-01-01")  # randomized retention campaign

# ---- snapshot design (prediction window = 90 days) ----
HORIZON_DAYS = 90
IN_PLAY_DAYS = 180                          # customer must have ordered in last 180d
CUTOFFS = ["2023-07-01", "2023-10-01", "2024-01-01", "2024-04-01",
           "2024-07-01", "2024-10-01", "2025-01-01"]
TRAIN_CUTOFFS = CUTOFFS[:5]   # their label windows end before the VALID cutoff
VALID_CUTOFF = CUTOFFS[5]
TEST_CUTOFF = CUTOFFS[6]      # == campaign date

# ---- Singapore / business assumptions (state these in your write-up) ----
GST = 1.09                    # GST 9% since 1 Jan 2024; revenue is analysed net of GST
GROSS_MARGIN = 0.25           # assumed gross margin on net revenue
COST_PER_TARGET = 2.0         # SGD: push/SMS cost + expected voucher redemption cost
SAVE_RATE = 0.15              # assumed share of at-risk customers a campaign retains
