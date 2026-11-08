"""Run sql/features.sql in DuckDB and write data/processed/*.parquet|csv."""
from pathlib import Path
import duckdb
from .config import CUTOFFS

RAW, OUT = Path("data/raw"), Path("data/processed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW customers AS SELECT customer_id, CAST(signup_date AS DATE) AS signup_date, "
                f"planning_area, region, housing_type, age_band, acquisition_channel "
                f"FROM read_parquet('{RAW / 'customers.parquet'}')")
    con.execute(f"CREATE VIEW orders AS SELECT order_id, customer_id, CAST(order_date AS DATE) AS order_date, "
                f"order_channel, amount_sgd, voucher_used FROM read_parquet('{RAW / 'orders.parquet'}')")
    con.execute(f"CREATE VIEW campaign AS SELECT customer_id, CAST(campaign_date AS DATE) AS campaign_date, treated "
                f"FROM read_parquet('{RAW / 'campaign.parquet'}')")
    vals = ", ".join(f"(DATE '{c}')" for c in CUTOFFS)
    con.execute(f"CREATE TABLE cutoffs AS SELECT * FROM (VALUES {vals}) t(cutoff)")

    con.execute(Path("sql/features.sql").read_text())
    con.execute(f"COPY snapshots TO '{OUT / 'snapshots.parquet'}' (FORMAT PARQUET)")
    con.execute(f"COPY survival TO '{OUT / 'survival.csv'}' (HEADER, DELIMITER ',')")
    n, churn = con.execute("SELECT COUNT(*), AVG(churn_90d) FROM snapshots").fetchone()
    print(f"snapshots: {n:,} rows | mean churn rate {churn:.1%}")
    print(con.execute("SELECT cutoff, COUNT(*) n, ROUND(AVG(churn_90d),3) churn FROM snapshots GROUP BY 1 ORDER BY 1").df())


if __name__ == "__main__":
    main()
