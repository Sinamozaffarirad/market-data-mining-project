/* ============================================================================
   Power BI star schema for the dunnhumby "Complete Journey" market basket data
   ----------------------------------------------------------------------------
   Read-only views over the existing tables.  Nothing here writes data, so the
   whole layer can be removed with DROP VIEW without touching the warehouse.

   SYNTHETIC CALENDAR
   The source data has no calendar dates: transactions.day is an integer from
   1 to 711 and week_no runs 1 to 102.  A Year / Quarter / Month / Week / Day
   drill-down therefore needs a date dimension, so day 1 is anchored to
   2017-01-01 (a Sunday, which keeps week_no aligned to Sunday-start weeks).
   The calendar is a presentation device for drill-down only; the underlying
   day numbers are preserved as day_key and are the values to quote in any
   written analysis.
   ============================================================================ */

/* ---------- Dimension: date ------------------------------------------------ */
CREATE OR ALTER VIEW dbo.vw_dim_date AS
WITH day_numbers AS (
    SELECT TOP (711) ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS day_key
    FROM sys.all_objects a CROSS JOIN sys.all_objects b
)
SELECT
    d.day_key,
    CAST(DATEADD(DAY, d.day_key - 1, '2017-01-01') AS date)      AS date_value,
    DATEPART(YEAR,    DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS calendar_year,
    DATEPART(QUARTER, DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS calendar_quarter,
    'Q' + CAST(DATEPART(QUARTER, DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS varchar(1))
                                                                  AS quarter_name,
    DATEPART(MONTH,   DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS calendar_month,
    DATENAME(MONTH,   DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS month_name,
    FORMAT(DATEADD(DAY, d.day_key - 1, '2017-01-01'), 'yyyy-MM')  AS year_month,
    /* Dataset week, 1-102, matching transactions.week_no. */
    ((d.day_key - 1) / 7) + 1                                     AS week_no,
    /* Week within the calendar month, 1-5.  The month drill-down groups on
       this rather than week_no: a month overlaps five *dataset* weeks whose
       first and last are clipped by the month boundary, which made the ends of
       every month look like a slump.  Week 5 here is still a 1-3 day remnant,
       so the drill-down reports the days each bucket covers. */
    ((DATEPART(DAY, DATEADD(DAY, d.day_key - 1, '2017-01-01')) - 1) / 7) + 1
                                                                  AS week_of_month,
    DATENAME(WEEKDAY, DATEADD(DAY, d.day_key - 1, '2017-01-01'))  AS day_name,
    /* Sort keys so Power BI orders text columns chronologically. */
    DATEPART(MONTH,   DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS month_sort,
    DATEPART(WEEKDAY, DATEADD(DAY, d.day_key - 1, '2017-01-01')) AS day_sort,
    /* The 30-day periods the revenue forecaster uses, so BI figures reconcile
       with the Predictive tab.  Days 1-21 fall outside the aligned periods. */
    CASE WHEN d.day_key >= 22 THEN ((d.day_key - 22) / 30) + 1 END AS forecast_period
FROM day_numbers d;
GO

/* ---------- Dimension: product (Department > Commodity > Sub > Product) ---- */
CREATE OR ALTER VIEW dbo.vw_dim_product AS
SELECT
    p.product_id,
    COALESCE(NULLIF(LTRIM(RTRIM(p.department)), ''),         'UNKNOWN') AS department,
    COALESCE(NULLIF(LTRIM(RTRIM(p.commodity_desc)), ''),     'UNKNOWN') AS commodity,
    COALESCE(NULLIF(LTRIM(RTRIM(p.sub_commodity_desc)), ''), 'UNKNOWN') AS sub_commodity,
    COALESCE(NULLIF(LTRIM(RTRIM(p.brand)), ''),              'UNKNOWN') AS brand,
    p.manufacturer,
    COALESCE(NULLIF(LTRIM(RTRIM(p.curr_size_of_product)), ''), 'N/A')   AS product_size
FROM dbo.product p;
GO

/* ---------- Dimension: store ----------------------------------------------- */
CREATE OR ALTER VIEW dbo.vw_dim_store AS
SELECT
    t.store_id,
    'Store ' + CAST(t.store_id AS varchar(12)) AS store_name,
    COUNT(DISTINCT t.basket_id)                AS lifetime_baskets,
    CAST(SUM(t.sales_value) AS decimal(18, 2)) AS lifetime_revenue
FROM dbo.transactions t
WHERE t.store_id IS NOT NULL
GROUP BY t.store_id;
GO

/* ---------- Dimension: household (demographics + RFM segment) -------------- */
CREATE OR ALTER VIEW dbo.vw_dim_household AS
SELECT
    h.household_key,
    COALESCE(h.age_desc,            'Unknown') AS age_group,
    COALESCE(h.income_desc,         'Unknown') AS income_band,
    COALESCE(h.marital_status_code, 'Unknown') AS marital_status,
    COALESCE(h.homeowner_desc,      'Unknown') AS homeowner,
    COALESCE(h.hh_comp_desc,        'Unknown') AS household_composition,
    COALESCE(h.household_size_desc, 'Unknown') AS household_size,
    COALESCE(h.kid_category_desc,   'Unknown') AS kids,
    COALESCE(s.rfm_segment,         'Unsegmented') AS rfm_segment,
    s.recency_score,
    s.frequency_score,
    s.monetary_score,
    s.churn_probability,
    CASE WHEN h.age_desc IS NULL THEN 0 ELSE 1 END AS has_demographics
FROM dbo.household h
LEFT JOIN dbo.dunnhumby_customersegment s
       ON s.household_key = h.household_key;
GO

/* ---------- Fact: sales line ------------------------------------------------
   Transaction-line grain, one row per product per basket.  Net revenue applies
   the retailer and coupon discounts already present in the source. */
CREATE OR ALTER VIEW dbo.vw_fact_sales AS
SELECT
    t.id                AS sales_key,
    t.day               AS day_key,
    t.product_id,
    t.store_id,
    t.household_key,
    t.basket_id,
    t.quantity,
    /* sales_value is what the customer actually paid.  retail_disc and
       coupon_disc are stored as negative amounts, so they are negated to give
       positive discount figures and added back to recover the pre-discount
       price.  Adding them directly would subtract the discount twice. */
    CAST(t.sales_value AS decimal(18, 2))                       AS sales_value,
    CAST(-t.retail_disc AS decimal(18, 2))                      AS retail_discount,
    CAST(-t.coupon_disc AS decimal(18, 2))                      AS coupon_discount,
    CAST(-(t.retail_disc + t.coupon_disc) AS decimal(18, 2))    AS total_discount,
    CAST(t.sales_value - t.retail_disc - t.coupon_disc
         AS decimal(18, 2))                                     AS gross_before_discount,
    CASE WHEN t.coupon_disc <> 0 THEN 1 ELSE 0 END              AS used_coupon
FROM dbo.transactions t
WHERE t.product_id IS NOT NULL
  AND t.sales_value IS NOT NULL;
GO

/* ---------- Fact: basket ----------------------------------------------------
   Basket grain, so "average basket size" and "basket value" are additive and
   are not distorted by counting a basket once per line.

   BASKET SIZE: use basket_size (distinct products, averaging 9.39).  Do NOT
   headline total_units_raw: dunnhumby records weighted and dispensed goods in
   source units, so COUPON/MISC ITEMS reaches 89,638 and FUEL 30,080 on a
   single line, which drags the raw average to ~943 units per basket.
   total_units_excl_bulk removes those two commodities for the cases where a
   unit count is genuinely wanted. */
CREATE OR ALTER VIEW dbo.vw_fact_basket AS
SELECT
    t.basket_id,
    MIN(t.day)                                   AS day_key,
    MIN(t.store_id)                              AS store_id,
    MIN(t.household_key)                         AS household_key,
    COUNT(*)                                     AS line_count,
    COUNT(DISTINCT t.product_id)                 AS basket_size,
    SUM(CASE WHEN t.quantity > 0 THEN t.quantity ELSE 0 END) AS total_units_raw,
    SUM(CASE WHEN t.quantity > 0
              AND COALESCE(p.commodity_desc, '') NOT IN ('COUPON/MISC ITEMS', 'FUEL')
             THEN t.quantity ELSE 0 END)         AS total_units_excl_bulk,
    CAST(SUM(t.sales_value) AS decimal(18, 2))   AS basket_value
FROM dbo.transactions t
LEFT JOIN dbo.product p ON p.product_id = t.product_id
GROUP BY t.basket_id;
GO
