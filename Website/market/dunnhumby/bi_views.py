"""Business-intelligence dashboard over the star-schema reporting views.

Serves the drill-down reporting layer from Django rather than an external BI
tool: Department -> Commodity -> Sub-commodity -> Product on one axis,
Year -> Quarter -> Month -> Week -> Day on the other, with every panel sharing
one filter context so clicking a mark in any chart narrows all the others.

Everything reads the ``vw_`` views created by
``dunnhumby/sql/powerbi_star_schema.sql``.  No endpoint writes.
"""
from __future__ import annotations

import logging

from django.db import connection
from django.http import JsonResponse
from django.shortcuts import render

from .views import _bi_filter_options, admin_required


logger = logging.getLogger(__name__)

def _from(needs):
    """FROM clause carrying only the dimensions a query actually references.

    Joining all three unconditionally made every panel pay for dimensions it
    never used, which on a 2.6M row fact table cost seconds per request. Each
    endpoint declares the aliases it needs and the active filters add their own.
    """
    sql = ["FROM vw_fact_sales f"]
    if "d" in needs:
        sql.append("JOIN vw_dim_date d ON d.day_key = f.day_key")
    if "p" in needs:
        sql.append("JOIN vw_dim_product p ON p.product_id = f.product_id")
    if "h" in needs:
        sql.append("LEFT JOIN vw_dim_household h ON h.household_key = f.household_key")
    return "\n    ".join(sql)

# Every dimension a mark can be clicked on.  Panels all build their WHERE from
# this one table, so a filter set anywhere applies everywhere.
FILTER_COLUMNS = {
    "year": ("d.calendar_year", int),
    "quarter": ("d.quarter_name", str),
    "month": ("d.month_name", str),
    "week": ("d.week_of_month", int),
    "day": ("d.day_key", int),
    "weekday": ("d.day_name", str),
    "hour": ("f.trans_hour", int),
    "department": ("p.department", str),
    "commodity": ("p.commodity", str),
    "sub_commodity": ("p.sub_commodity", str),
    "product": ("f.product_id", int),
    "brand": ("p.brand", str),
    "store": ("f.store_id", int),
    "segment": ("h.rfm_segment", str),
    "age": ("h.age_group", str),
    "income": ("h.income_band", str),
    "household_size": ("h.household_size", str),
}

PRODUCT_LEVELS = [
    {"key": "department", "column": "p.department", "label": "Department"},
    {"key": "commodity", "column": "p.commodity", "label": "Commodity"},
    {"key": "sub_commodity", "column": "p.sub_commodity", "label": "Sub-commodity"},
    {"key": "product", "column": "CAST(f.product_id AS varchar(20))", "label": "Product"},
]

TIME_LEVELS = [
    {"key": "year", "select": "CAST(d.calendar_year AS varchar(4))",
     "group": "d.calendar_year", "label": "Year"},
    {"key": "quarter", "select": "d.quarter_name",
     "group": "d.calendar_quarter", "label": "Quarter"},
    {"key": "month", "select": "d.month_name",
     "group": "d.calendar_month", "label": "Month"},
    # Week within the month, not the dataset week: a month overlaps five dataset
    # weeks whose ends are clipped by the month boundary, which read as a slump.
    {"key": "week", "select": "'Week ' + CAST(d.week_of_month AS varchar(2))",
     "group": "d.week_of_month", "label": "Week"},
    {"key": "day", "select": "CAST(d.day_key AS varchar(4))",
     "group": "d.day_key", "label": "Day"},
]

DEMOGRAPHIC_DIMENSIONS = {
    "age": ("h.age_group", "Age band"),
    "income": ("h.income_band", "Income band"),
    "household_size": ("h.household_size", "Household size"),
    "kids": ("h.kids", "Children"),
    "homeowner": ("h.homeowner", "Home ownership"),
}


def _filters(request, needs=()):
    """Active filters as predicates, parameters, and the aliases they require."""
    where, params, required = [], [], set(needs)
    for key, (column, cast) in FILTER_COLUMNS.items():
        raw = (request.GET.get(key) or "").strip()
        if not raw or raw.lower() == "all":
            continue
        if cast is int:
            if not raw.lstrip("-").isdigit():
                continue
            params.append(int(raw))
        else:
            params.append(raw)
        where.append(f"{column} = %s")
        required.add(column.split(".")[0])
    return where, params, required


def _clause(where):
    return (" WHERE " + " AND ".join(where)) if where else ""


def _query(sql, params):
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _scalar_row(sql, params):
    rows = _query(sql, params)
    return rows[0] if rows else {}


def _rank_rows(rows, value_key="revenue"):
    """Attach float values and each row's share of the visible total."""
    for row in rows:
        row[value_key] = float(row.get(value_key) or 0)
    total = sum(row[value_key] for row in rows)
    for row in rows:
        row["share"] = (row[value_key] / total) if total else 0
    return rows


@admin_required
def bi_dashboard(request):
    """Render the dashboard shell; every panel loads over the API."""
    options = _bi_filter_options()
    context = {"title": "Business Intelligence Dashboard", "filter_options": options}
    if not options["departments"]:
        context["setup_error"] = (
            "The reporting views are missing. Run "
            "dunnhumby/sql/powerbi_star_schema.sql against marketdb, then reload."
        )
    return render(request, "site/dunnhumby/bi_dashboard.html", context)


@admin_required
def api_bi_kpis(request):
    """Headline figures for the current filter context."""
    where, params, needs = _filters(request)
    clause = _clause(where)
    totals = _scalar_row(f"""
        SELECT
            COALESCE(SUM(f.sales_value), 0)           AS revenue,
            COALESCE(SUM(f.total_discount), 0)        AS discount,
            COALESCE(SUM(f.gross_before_discount), 0) AS list_value,
            COUNT(DISTINCT f.basket_id)               AS baskets,
            COUNT(DISTINCT f.household_key)           AS households,
            COUNT(DISTINCT f.product_id)              AS products,
            COUNT(DISTINCT f.store_id)                AS stores,
            COUNT(DISTINCT f.day_key)                 AS days
        {_from(needs)}{clause}
    """, params)

    revenue = float(totals.get("revenue") or 0)
    baskets = int(totals.get("baskets") or 0)
    households = int(totals.get("households") or 0)
    list_value = float(totals.get("list_value") or 0)
    days = int(totals.get("days") or 0)

    # Basket size counts distinct products. dunnhumby records weighted and
    # dispensed goods in source units, so a units-based average is about 943.
    size = _scalar_row(f"""
        SELECT AVG(CAST(x.items AS float)) AS avg_items
        FROM (
            SELECT f.basket_id, COUNT(DISTINCT f.product_id) AS items
            {_from(needs)}{clause}
            GROUP BY f.basket_id
        ) x
    """, params)

    concentration = _scalar_row(f"""
        SELECT COALESCE(SUM(top_rev), 0) AS top20
        FROM (
            SELECT TOP 20 SUM(f.sales_value) AS top_rev
            {_from(needs)}{clause}
            GROUP BY f.product_id
            ORDER BY SUM(f.sales_value) DESC
        ) t
    """, params)
    top20 = float(concentration.get("top20") or 0)

    return JsonResponse({
        "success": True,
        "kpis": {
            "revenue": revenue,
            "baskets": baskets,
            "households": households,
            "products": int(totals.get("products") or 0),
            "stores": int(totals.get("stores") or 0),
            "days": days,
            "avg_basket_value": revenue / baskets if baskets else 0,
            "avg_basket_size": float(size.get("avg_items") or 0),
            "revenue_per_household": revenue / households if households else 0,
            "visits_per_household": baskets / households if households else 0,
            "revenue_per_day": revenue / days if days else 0,
            "discount_rate": float(totals.get("discount") or 0) / list_value if list_value else 0,
            "top20_concentration": top20 / revenue if revenue else 0,
        },
    })


def _drill(request, levels, is_product):
    """Shared drill-down handler for both hierarchies."""
    path = [v for v in request.GET.getlist("path") if v != ""]
    depth = min(len(path), len(levels) - 1)
    level = levels[depth]

    # The hierarchy being walked decides which dimension has to be joined.
    where, params, needs = _filters(request, ["p"] if is_product else ["d"])
    for index, value in enumerate(path):
        if index >= len(levels):
            break
        column = levels[index]["column"] if is_product else levels[index]["select"]
        where.append(f"{column} = %s")
        params.append(value)
    clause = _clause(where)

    label_sql = level["column"] if is_product else level["select"]
    group_sql = level["column"] if is_product else level["group"]
    select_label = label_sql
    group_by = group_sql if is_product else f"{group_sql}, {label_sql}"
    order_sql = "revenue DESC" if is_product else f"{group_sql} ASC"

    rows = _query(f"""
        SELECT TOP 40
            {select_label} AS label,
            SUM(f.sales_value)           AS revenue,
            COUNT(DISTINCT f.basket_id)  AS baskets,
            COUNT(DISTINCT f.day_key)    AS days
        {_from(needs)}{clause}
        GROUP BY {group_by}
        ORDER BY {order_sql}
    """, params)

    _rank_rows(rows)
    # Calendar buckets are not all the same length: the 711-day window clips the
    # last quarter and a month's fifth week is a 1-3 day remnant. Reporting days
    # covered, and revenue per day, stops a short bucket reading as a downturn.
    full = max((int(r["days"] or 0) for r in rows), default=0)
    for row in rows:
        row["days"] = int(row["days"] or 0)
        row["revenue_per_day"] = row["revenue"] / row["days"] if row["days"] else 0
        # Only a material shortfall counts: quarters legitimately run 90 to 92
        # days, so testing against the longest would brand every first quarter
        # as truncated.
        row["partial"] = bool(not is_product and full and row["days"] < full * 0.9)

    return JsonResponse({
        "success": True,
        "level": level["key"],
        "level_label": level["label"],
        "path": path,
        "breadcrumb": [
            {"label": levels[i]["label"], "value": value}
            for i, value in enumerate(path) if i < len(levels)
        ],
        "can_drill": depth < len(levels) - 1,
        "next_label": levels[depth + 1]["label"] if depth < len(levels) - 1 else None,
        "rows": rows,
    })


@admin_required
def api_bi_product_drill(request):
    """Department -> Commodity -> Sub-commodity -> Product."""
    return _drill(request, PRODUCT_LEVELS, is_product=True)


@admin_required
def api_bi_time_drill(request):
    """Year -> Quarter -> Month -> Week -> Day."""
    return _drill(request, TIME_LEVELS, is_product=False)


@admin_required
def api_bi_stores(request):
    """Store scatter: basket count against average basket value."""
    where, params, needs = _filters(request)
    rows = _query(f"""
        SELECT TOP 300
            f.store_id,
            SUM(f.sales_value)          AS revenue,
            COUNT(DISTINCT f.basket_id) AS baskets
        {_from(needs)}{_clause(where)}
        GROUP BY f.store_id
        ORDER BY revenue DESC
    """, params)
    _rank_rows(rows)
    for row in rows:
        row["avg_basket"] = row["revenue"] / row["baskets"] if row["baskets"] else 0
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_segments(request):
    """Revenue and household count per RFM segment."""
    where, params, needs = _filters(request, ["h"])
    rows = _query(f"""
        SELECT
            COALESCE(h.rfm_segment, 'Unsegmented') AS segment,
            SUM(f.sales_value)                     AS revenue,
            COUNT(DISTINCT f.household_key)        AS households,
            COUNT(DISTINCT f.basket_id)            AS baskets
        {_from(needs)}{_clause(where)}
        GROUP BY COALESCE(h.rfm_segment, 'Unsegmented')
        ORDER BY revenue DESC
    """, params)
    _rank_rows(rows)
    for row in rows:
        row["revenue_per_household"] = (
            row["revenue"] / row["households"] if row["households"] else 0
        )
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_basket_distribution(request):
    """How many baskets hold N distinct products.

    Every bar is one basket size.  Folding the tail into a single bucket stacked
    a long thin tail into a bar taller than the mid sizes and read as a second
    mode, so the tail is summarised in words instead.
    """
    where, params, needs = _filters(request)
    rows = _query(f"""
        SELECT items AS bucket, COUNT(*) AS baskets, SUM(value) AS revenue
        FROM (
            SELECT f.basket_id,
                   COUNT(DISTINCT f.product_id) AS items,
                   SUM(f.sales_value) AS value
            {_from(needs)}{_clause(where)}
            GROUP BY f.basket_id
        ) b
        GROUP BY items
        ORDER BY items
    """, params)
    for row in rows:
        row["revenue"] = float(row["revenue"] or 0)
        row["baskets"] = int(row["baskets"] or 0)
        row["bucket"] = int(row["bucket"] or 0)

    total_baskets = sum(r["baskets"] for r in rows)
    cutoff = 40
    tail = [r for r in rows if r["bucket"] > cutoff]
    return JsonResponse({
        "success": True,
        "rows": rows,
        "cutoff": cutoff,
        "max_size": max((r["bucket"] for r in rows), default=0),
        "median_size": _median_size(rows, total_baskets),
        "tail": {
            "from_size": cutoff + 1,
            "baskets": sum(r["baskets"] for r in tail),
            "revenue": sum(r["revenue"] for r in tail),
            "share": (sum(r["baskets"] for r in tail) / total_baskets) if total_baskets else 0,
        },
    })


def _median_size(rows, total_baskets):
    """Median basket size from the size histogram."""
    if not total_baskets:
        return 0
    seen, half = 0, total_baskets / 2
    for row in rows:
        seen += row["baskets"]
        if seen >= half:
            return row["bucket"]
    return rows[-1]["bucket"] if rows else 0


@admin_required
def api_bi_daypart(request):
    """Trading pattern by hour of day and by weekday."""
    where, params, needs = _filters(request)
    clause = _clause(where)
    hours = _query(f"""
        SELECT f.trans_hour AS hour,
               SUM(f.sales_value)          AS revenue,
               COUNT(DISTINCT f.basket_id) AS baskets
        {_from(needs)}{clause}
        GROUP BY f.trans_hour
        ORDER BY f.trans_hour
    """, params)
    for row in hours:
        row["revenue"] = float(row["revenue"] or 0)
        row["hour"] = int(row["hour"] or 0)
        row["avg_basket"] = row["revenue"] / row["baskets"] if row["baskets"] else 0

    weekdays = _query(f"""
        SELECT d.day_name AS weekday,
               SUM(f.sales_value)          AS revenue,
               COUNT(DISTINCT f.basket_id) AS baskets,
               COUNT(DISTINCT f.day_key)   AS days
        {_from(needs | {"d"})}{clause}
        GROUP BY d.day_sort, d.day_name
        ORDER BY d.day_sort
    """, params)
    for row in weekdays:
        row["revenue"] = float(row["revenue"] or 0)
        row["days"] = int(row["days"] or 0)
        # Weekdays occur a different number of times across the window, so the
        # per-day rate is the comparable figure.
        row["revenue_per_day"] = row["revenue"] / row["days"] if row["days"] else 0
    return JsonResponse({"success": True, "hours": hours, "weekdays": weekdays})


@admin_required
def api_bi_demographics(request):
    """Revenue by a chosen household attribute.

    Only 802 of 2,497 households carry demographics, so the uncovered share is
    returned rather than silently drawn as an 'Unknown' bar that dominates.
    """
    dimension = (request.GET.get("dimension") or "age").strip()
    column, label = DEMOGRAPHIC_DIMENSIONS.get(dimension, DEMOGRAPHIC_DIMENSIONS["age"])
    where, params, needs = _filters(request, ["h"])
    known = where + [f"{column} <> 'Unknown'", f"{column} IS NOT NULL"]
    rows = _query(f"""
        SELECT {column} AS label,
               SUM(f.sales_value)              AS revenue,
               COUNT(DISTINCT f.household_key) AS households,
               COUNT(DISTINCT f.basket_id)     AS baskets
        {_from(needs)}{_clause(known)}
        GROUP BY {column}
        ORDER BY revenue DESC
    """, params)
    _rank_rows(rows)
    for row in rows:
        row["revenue_per_household"] = (
            row["revenue"] / row["households"] if row["households"] else 0
        )

    coverage = _scalar_row(f"""
        SELECT SUM(CASE WHEN {column} <> 'Unknown' AND {column} IS NOT NULL
                        THEN f.sales_value ELSE 0 END) AS covered,
               SUM(f.sales_value) AS total
        {_from(needs)}{_clause(where)}
    """, params)
    total = float(coverage.get("total") or 0)
    return JsonResponse({
        "success": True,
        "dimension": dimension,
        "dimension_label": label,
        "rows": rows,
        "coverage": float(coverage.get("covered") or 0) / total if total else 0,
    })


@admin_required
def api_bi_brand(request):
    """National against private label, and the discount each carries."""
    where, params, needs = _filters(request, ["p"])
    rows = _query(f"""
        SELECT p.brand AS label,
               SUM(f.sales_value)           AS revenue,
               SUM(f.total_discount)        AS discount,
               SUM(f.gross_before_discount) AS list_value,
               COUNT(DISTINCT f.basket_id)  AS baskets,
               COUNT(DISTINCT f.product_id) AS products
        {_from(needs)}{_clause(where)}
        GROUP BY p.brand
        ORDER BY revenue DESC
    """, params)
    _rank_rows(rows)
    for row in rows:
        list_value = float(row.get("list_value") or 0)
        row["discount_rate"] = float(row.get("discount") or 0) / list_value if list_value else 0
        row["discount"] = float(row.get("discount") or 0)
        row["list_value"] = list_value
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_top_products(request):
    """Highest-revenue products for the current filter context.

    Ranking is done on the sum alone, which the columnstore index answers
    quickly, and the distinct basket count is then taken for the surviving rows
    only: doing both in one pass meant counting distinct baskets for all 92,353
    products and took over ten seconds.
    """
    where, params, needs = _filters(request)
    clause = _clause(where)
    rows = _query(f"""
        ;WITH ranked AS (
            SELECT TOP 25 f.product_id, SUM(f.sales_value) AS revenue
            {_from(needs)}{clause}
            GROUP BY f.product_id
            ORDER BY SUM(f.sales_value) DESC
        )
        SELECT r.product_id,
               MIN(p.department) AS department,
               MIN(p.commodity)  AS commodity,
               MIN(p.brand)      AS brand,
               MIN(r.revenue)    AS revenue,
               COUNT(DISTINCT f.basket_id) AS baskets
        FROM ranked r
        JOIN vw_fact_sales f  ON f.product_id = r.product_id
        JOIN vw_dim_product p ON p.product_id = r.product_id
        GROUP BY r.product_id
        ORDER BY MIN(r.revenue) DESC
    """, params)
    _rank_rows(rows)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_discount_trend(request):
    """Discount rate and revenue by month, to see promotion intensity move."""
    where, params, needs = _filters(request, ["d"])
    rows = _query(f"""
        SELECT d.year_month AS label,
               SUM(f.sales_value)           AS revenue,
               SUM(f.total_discount)        AS discount,
               SUM(f.gross_before_discount) AS list_value,
               COUNT(DISTINCT f.day_key)    AS days
        {_from(needs)}{_clause(where)}
        GROUP BY d.calendar_year, d.calendar_month, d.year_month
        ORDER BY d.calendar_year, d.calendar_month
    """, params)
    for row in rows:
        row["revenue"] = float(row["revenue"] or 0)
        row["discount"] = float(row["discount"] or 0)
        list_value = float(row.get("list_value") or 0)
        row["discount_rate"] = row["discount"] / list_value if list_value else 0
        row["days"] = int(row["days"] or 0)
        row["revenue_per_day"] = row["revenue"] / row["days"] if row["days"] else 0
    full = max((r["days"] for r in rows), default=0)
    for row in rows:
        row["partial"] = bool(full and row["days"] < full * 0.9)
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_insights(request):
    """Rule-based findings for the current filter context.

    Each statement comes from the same filtered queries that feed the charts, so
    the narrative cannot drift from what is on screen.
    """
    where, params, needs = _filters(request)
    clause = _clause(where)
    insights = []

    total = float(_scalar_row(
        f"SELECT SUM(f.sales_value) AS revenue {_from(needs)}{clause}", params
    ).get("revenue") or 0)
    if not total:
        return JsonResponse({"success": True, "insights": [{
            "kind": "empty",
            "title": "No sales match these filters",
            "detail": "Remove a filter to bring data back into view.",
        }]})

    quarters = _query(f"""
        SELECT CAST(d.calendar_year AS varchar(4)) + ' ' + d.quarter_name AS label,
               SUM(f.sales_value)        AS revenue,
               COUNT(DISTINCT f.day_key) AS days
        {_from(needs | {"d"})}{clause}
        GROUP BY d.calendar_year, d.calendar_quarter, d.quarter_name
        ORDER BY d.calendar_year, d.calendar_quarter
    """, params)
    rated = [
        (r["label"], float(r["revenue"] or 0) / int(r["days"]), int(r["days"]))
        for r in quarters if int(r["days"] or 0) > 0
    ]
    if len(rated) >= 2:
        best = max(rated, key=lambda r: r[1])
        worst = min(rated, key=lambda r: r[1])
        insights.append({
            "kind": "season",
            "title": f"{best[0]} trades hardest at ${best[1]:,.0f} a day",
            "detail": (
                f"Against {worst[0]} at ${worst[1]:,.0f} a day, "
                f"{(best[1] / worst[1] - 1):.0%} higher. Daily rates are used because "
                "the window clips the final quarter."
            ),
        })

    departments = _query(f"""
        SELECT TOP 3 p.department AS label, SUM(f.sales_value) AS revenue
        {_from(needs | {"p"})}{clause}
        GROUP BY p.department ORDER BY revenue DESC
    """, params)
    if departments:
        share = sum(float(d["revenue"] or 0) for d in departments) / total
        insights.append({
            "kind": "concentration",
            "title": f"{departments[0]['label']} leads the catalogue",
            "detail": (
                f"The top three departments ({', '.join(d['label'] for d in departments)}) "
                f"take {share:.1%} of revenue here."
            ),
        })

    segment_where = where + ["h.rfm_segment IS NOT NULL"]
    segments = _query(f"""
        SELECT TOP 1 h.rfm_segment AS label,
               SUM(f.sales_value) AS revenue,
               COUNT(DISTINCT f.household_key) AS households
        {_from(needs | {"h"})}{_clause(segment_where)}
        GROUP BY h.rfm_segment ORDER BY revenue DESC
    """, params)
    if segments:
        seg = segments[0]
        insights.append({
            "kind": "segment",
            "title": f"{seg['label']} households drive the most revenue",
            "detail": (
                f"{int(seg['households']):,} households generate "
                f"${float(seg['revenue']):,.0f}, {float(seg['revenue']) / total:.1%} of the total."
            ),
        })

    hours = _query(f"""
        SELECT TOP 1 f.trans_hour AS hour, SUM(f.sales_value) AS revenue
        {_from(needs)}{clause}
        GROUP BY f.trans_hour ORDER BY revenue DESC
    """, params)
    if hours:
        hour = int(hours[0]["hour"] or 0)
        insights.append({
            "kind": "daypart",
            "title": f"Trade peaks between {hour:02d}:00 and {hour + 1:02d}:00",
            "detail": (
                f"${float(hours[0]['revenue']):,.0f} passes through that hour, "
                f"{float(hours[0]['revenue']) / total:.1%} of revenue."
            ),
        })

    brands = _query(f"""
        SELECT p.brand AS label, SUM(f.sales_value) AS revenue,
               SUM(f.total_discount) AS discount,
               SUM(f.gross_before_discount) AS list_value
        {_from(needs | {"p"})}{clause}
        GROUP BY p.brand
    """, params)
    private = next((b for b in brands if b["label"] == "Private"), None)
    if private and total:
        insights.append({
            "kind": "brand",
            "title": f"Private label takes {float(private['revenue']) / total:.1%} of revenue",
            "detail": (
                "National brands hold the rest. Private label share is a standard "
                "measure of own-brand strength in grocery retail."
            ),
        })

    discount = _scalar_row(f"""
        SELECT COALESCE(SUM(f.total_discount), 0) AS discount,
               COALESCE(SUM(f.gross_before_discount), 0) AS list_value
        {_from(needs)}{clause}
    """, params)
    list_value = float(discount.get("list_value") or 0)
    if list_value:
        insights.append({
            "kind": "discount",
            "title": f"{float(discount['discount']) / list_value:.1%} of list value is discounted",
            "detail": (
                f"${float(discount['discount']):,.0f} of promotions against "
                f"${list_value:,.0f} at list price."
            ),
        })

    return JsonResponse({"success": True, "insights": insights})
