-- One row per (customer, cutoff) for customers "in play" at that cutoff.
-- LEAKAGE RULES
--   * every feature uses order_date <  cutoff
--   * every label   uses order_date >= cutoff AND < cutoff + 90
-- Tables expected: customers, orders, cutoffs(cutoff DATE), campaign
-- Revenue is analysed net of 9% GST (amount / 1.09).

CREATE OR REPLACE TABLE snapshots AS
WITH o AS (
    SELECT o.customer_id, o.order_date, o.amount_sgd / 1.09 AS net_amt,
           o.voucher_used, o.order_channel, c.cutoff
    FROM orders o
    JOIN cutoffs c ON o.order_date < c.cutoff
),
agg AS (
    SELECT customer_id, cutoff,
        MIN(order_date)  AS first_order,
        MAX(order_date)  AS last_order,
        COUNT(*)         AS n_orders_all,
        SUM(net_amt)     AS net_rev_all,
        COUNT(*) FILTER (WHERE order_date >= cutoff - 30)  AS n_orders_30d,
        COUNT(*) FILTER (WHERE order_date >= cutoff - 90)  AS n_orders_90d,
        COUNT(*) FILTER (WHERE order_date >= cutoff - 180) AS n_orders_180d,
        COUNT(*) FILTER (WHERE order_date >= cutoff - 180 AND order_date < cutoff - 90) AS n_orders_prev90d,
        COALESCE(SUM(net_amt) FILTER (WHERE order_date >= cutoff - 90), 0)  AS net_rev_90d,
        COALESCE(SUM(net_amt) FILTER (WHERE order_date >= cutoff - 180), 0) AS net_rev_180d,
        AVG(voucher_used)                                  AS voucher_share,
        AVG((order_channel = 'app')::INT)                  AS app_share
    FROM o
    GROUP BY 1, 2
),
gaps AS (
    SELECT customer_id, cutoff, STDDEV_SAMP(gap) AS gap_std
    FROM (
        SELECT customer_id, cutoff,
               date_diff('day', LAG(order_date) OVER (PARTITION BY customer_id, cutoff ORDER BY order_date), order_date) AS gap
        FROM o
    )
    WHERE gap IS NOT NULL
    GROUP BY 1, 2
),
labels AS (
    SELECT o.customer_id, c.cutoff,
           COUNT(*) AS fut_orders_90d,
           SUM(o.amount_sgd / 1.09) AS fut_net_rev_90d
    FROM orders o
    JOIN cutoffs c ON o.order_date >= c.cutoff AND o.order_date < c.cutoff + 90
    GROUP BY 1, 2
)
SELECT
    a.customer_id, a.cutoff,
    -- recency / frequency / monetary
    date_diff('day', a.last_order, a.cutoff)                    AS recency_days,
    a.n_orders_all, a.n_orders_30d, a.n_orders_90d, a.n_orders_180d,
    a.net_rev_all, a.net_rev_90d, a.net_rev_180d,
    a.net_rev_all / a.n_orders_all                              AS avg_order_value,
    -- trend: last 90d vs previous 90d
    (a.n_orders_90d + 1.0) / (a.n_orders_prev90d + 1.0)         AS order_trend_ratio,
    -- rhythm: how overdue is this customer vs their own usual gap?
    date_diff('day', a.first_order, a.last_order) / NULLIF(a.n_orders_all - 1, 0) AS avg_gap_days,
    date_diff('day', a.last_order, a.cutoff)
        / NULLIF(date_diff('day', a.first_order, a.last_order) / NULLIF(a.n_orders_all - 1, 0), 0) AS overdue_ratio,
    g.gap_std,
    a.voucher_share, a.app_share,
    date_diff('day', c.signup_date, a.cutoff)                   AS tenure_days,
    c.planning_area, c.region, c.housing_type, c.age_band, c.acquisition_channel,
    -- labels (never use as features)
    COALESCE(l.fut_orders_90d, 0)                               AS fut_orders_90d,
    COALESCE(l.fut_net_rev_90d, 0)                              AS fut_net_rev_90d,
    (COALESCE(l.fut_orders_90d, 0) = 0)::INT                    AS churn_90d,
    cp.treated                                                  AS campaign_treated
FROM agg a
JOIN customers c USING (customer_id)
LEFT JOIN gaps g   USING (customer_id, cutoff)
LEFT JOIN labels l USING (customer_id, cutoff)
LEFT JOIN campaign cp ON cp.customer_id = a.customer_id AND cp.campaign_date = a.cutoff
WHERE date_diff('day', a.last_order, a.cutoff) <= 180;          -- "in play" only


-- Survival table for R: time from signup until last order, event = lapsed >90d before data end
CREATE OR REPLACE TABLE survival AS
SELECT c.customer_id, c.housing_type, c.age_band, c.acquisition_channel, c.region,
       CASE WHEN date_diff('day', MAX(o.order_date), DATE '2025-03-31') > 90 THEN 1 ELSE 0 END AS churned,
       CASE WHEN date_diff('day', MAX(o.order_date), DATE '2025-03-31') > 90
            THEN date_diff('day', c.signup_date, MAX(o.order_date))
            ELSE date_diff('day', c.signup_date, DATE '2025-03-31') END AS duration_days,
       COUNT(*) AS n_orders
FROM customers c JOIN orders o USING (customer_id)
GROUP BY c.customer_id, c.housing_type, c.age_band, c.acquisition_channel, c.region, c.signup_date;
