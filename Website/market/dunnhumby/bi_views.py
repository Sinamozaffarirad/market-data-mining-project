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

# ``key`` is also the filter each level sets: descending the hierarchy and
# filtering the dashboard are one action, so the breadcrumb and the filter chips
# can never disagree.
PRODUCT_LEVELS = [
    {"key": "department", "column": "p.department", "label": "Department"},
    {"key": "commodity", "column": "p.commodity", "label": "Commodity"},
    {"key": "sub_commodity", "column": "p.sub_commodity", "label": "Sub-commodity"},
    {"key": "product", "column": "CAST(f.product_id AS varchar(20))", "label": "Product"},
]

TIME_LEVELS = [
    {"key": "year", "select": "CAST(d.calendar_year AS varchar(4))",
     "value": "CAST(d.calendar_year AS varchar(4))",
     "group": "d.calendar_year", "label": "Year"},
    {"key": "quarter", "select": "d.quarter_name", "value": "d.quarter_name",
     "group": "d.calendar_quarter", "label": "Quarter"},
    {"key": "month", "select": "d.month_name", "value": "d.month_name",
     "group": "d.calendar_month", "label": "Month"},
    # Week within the month, not the dataset week: a month overlaps five dataset
    # weeks whose ends are clipped by the month boundary, which read as a slump.
    # It displays as "Week 5" but filters on the bare number, so the two are
    # carried separately.
    {"key": "week", "select": "'Week ' + CAST(d.week_of_month AS varchar(2))",
     "value": "CAST(d.week_of_month AS varchar(2))",
     "group": "d.week_of_month", "label": "Week", "crumb": "Week {}"},
    {"key": "day", "select": "'Day ' + CAST(d.day_key AS varchar(4))",
     "value": "CAST(d.day_key AS varchar(4))",
     "group": "d.day_key", "label": "Day", "crumb": "Day {}"},
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
    """Show the deepest hierarchy level the active filters have not yet pinned.

    There is no separate drill path: each level's filter key is what a click
    sets, so the level on display is simply the first one with no filter. That
    keeps the breadcrumb, the chips and every other panel describing the same
    selection.
    """
    where, params, needs = _filters(request, ["p"] if is_product else ["d"])

    depth = 0
    breadcrumb = []
    for index, level in enumerate(levels):
        value = (request.GET.get(level["key"]) or "").strip()
        if not value or value.lower() == "all":
            break
        breadcrumb.append({
            "label": level["label"],
            "value": value,
            "display": level.get("crumb", "{}").format(value),
            "key": level["key"],
        })
        depth = index + 1
    # The leaf stays selectable rather than drilling into nothing.
    depth = min(depth, len(levels) - 1)
    level = levels[depth]

    clause = _clause(where)
    label_sql = level["column"] if is_product else level["select"]
    value_sql = level["column"] if is_product else level["value"]
    group_sql = level["column"] if is_product else level["group"]
    select_label = label_sql
    group_by = group_sql if is_product else f"{group_sql}, {label_sql}, {value_sql}"
    order_sql = "revenue DESC" if is_product else f"{group_sql} ASC"

    rows = _query(f"""
        SELECT TOP 40
            {select_label} AS label,
            {value_sql} AS value,
            SUM(f.sales_value)           AS revenue,
            COUNT(DISTINCT f.basket_id)  AS baskets,
            COUNT(DISTINCT f.day_key)    AS days
        {_from(needs)}{clause}
        GROUP BY {group_by}
        ORDER BY {order_sql}
    """, params)

    _rank_rows(rows)
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
        "level_keys": [lvl["key"] for lvl in levels],
        "breadcrumb": breadcrumb,
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
    known = where + [f"{column} <> 'Unknown'", f"{column} <> ''", f"{column} IS NOT NULL"]
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
    rows = _apply_natural_order(dimension, rows, "label")

    coverage = _scalar_row(f"""
        SELECT SUM(CASE WHEN {column} <> 'Unknown' AND {column} <> ''
                         AND {column} IS NOT NULL
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


# Dimensions whose categories have a natural order.  Ranking them by revenue,
# which is the sensible default for an unordered dimension like department, put
# the income bands in the order 50-74K, 35-49K, 75-99K, 25-34K and so on, which
# reads as scrambled rather than ranked.
WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday"]
HOMEOWNER_ORDER = ["Homeowner", "Probable Owner", "Probable Renter", "Renter"]
NATURAL_ORDER_DIMENSIONS = {
    "income", "age", "household_size", "kids", "weekday", "quarter", "homeowner",
}


def _band_sort_key(value):
    """Sort key for a banded label such as 15-24K, Under 15K, 65+ or 5+.

    Bands are ordered by the first number they contain.  "Under 15K" shares its
    number with "15-24K", so it is nudged ahead of it, and labels standing for
    missing data are pushed to the end rather than sorted among real bands.
    """
    import re
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "unsegmented", "none", "n/a"}:
        return (2, 0.0, text)
    if text in WEEKDAY_ORDER:
        return (0, float(WEEKDAY_ORDER.index(text)), text)
    if text in HOMEOWNER_ORDER:
        return (0, float(HOMEOWNER_ORDER.index(text)), text)
    match = re.search(r"\d+", text)
    if not match:
        return (1, 0.0, text)
    number = float(match.group())
    if text.lower().startswith("under"):
        number -= 0.5
    return (0, number, text)


def _apply_natural_order(dimension, rows, key):
    """Order rows by their category's natural sequence where one exists."""
    if dimension not in NATURAL_ORDER_DIMENSIONS:
        return rows
    return sorted(rows, key=lambda row: _band_sort_key(row[key]))


# Groups that can be compared head to head.  Each maps to the column holding the
# group label and the dimension alias that column needs.
COMPARISON_DIMENSIONS = {
    "segment": ("h.rfm_segment", "h", "Customer segment"),
    "brand": ("p.brand", "p", "Brand type"),
    "weekday": ("d.day_name", "d", "Day of week"),
    "quarter": ("d.quarter_name", "d", "Quarter"),
    "department": ("p.department", "p", "Department"),
    "income": ("h.income_band", "h", "Income band"),
    "age": ("h.age_group", "h", "Age band"),
    "household_size": ("h.household_size", "h", "Household size"),
    "kids": ("h.kids", "h", "Children"),
    "homeowner": ("h.homeowner", "h", "Home ownership"),
}


def _cliffs_delta(first, second, sample=4000):
    """Cliff's delta: how often one group's baskets beat the other's.

    Reported alongside the p-value because with hundreds of thousands of baskets
    almost any difference reaches significance, so the p-value says only that a
    difference exists, not that it is large enough to act on.  Computed on a
    capped random sample because the exact form is quadratic in the group sizes.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if len(a) > sample:
        a = rng.choice(a, sample, replace=False)
    if len(b) > sample:
        b = rng.choice(b, sample, replace=False)
    if not len(a) or not len(b):
        return 0.0
    greater = int((a[:, None] > b[None, :]).sum())
    less = int((a[:, None] < b[None, :]).sum())
    return float((greater - less) / (len(a) * len(b)))


def _delta_label(value):
    """Conventional thresholds for interpreting an effect size."""
    size = abs(value)
    if size < 0.147:
        return "negligible"
    if size < 0.33:
        return "small"
    if size < 0.474:
        return "medium"
    return "large"


@admin_required
def api_bi_significance(request):
    """Compare two groups on basket value and on department mix.

    Only tests that suit this data are offered.  Basket value is heavily
    right-skewed and the groups are large and unequal, so spend is compared with
    Mann-Whitney U on ranks rather than a t-test, which would assume a normality
    basket value does not have.  Kolmogorov-Smirnov answers the separate
    question of distribution shape, and chi-square asks whether the two groups
    buy from different parts of the catalogue.  Every test carries an effect
    size, because at this sample size significance is close to guaranteed.
    """
    from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu

    dimension = (request.GET.get("dimension") or "segment").strip()
    column, alias, label = COMPARISON_DIMENSIONS.get(
        dimension, COMPARISON_DIMENSIONS["segment"]
    )
    group_a = (request.GET.get("group_a") or "").strip()
    group_b = (request.GET.get("group_b") or "").strip()

    where, params, needs = _filters(request, [alias])
    base = _clause(where + [column + " IS NOT NULL", column + " <> ''"])

    option_rows = [r for r in _query(f"""
        SELECT {column} AS label, SUM(f.sales_value) AS revenue
        {_from(needs)}{base}
        GROUP BY {column}
        ORDER BY revenue DESC
    """, params) if r["label"]]
    options = [r["label"] for r in _apply_natural_order(dimension, option_rows, "label")]

    if group_a not in options:
        group_a = options[0] if options else ""
    if group_b not in options or group_b == group_a:
        group_b = next((o for o in options if o != group_a), "")
    if not group_a or not group_b:
        return JsonResponse({
            "success": True, "dimension": dimension, "dimension_label": label,
            "options": options, "group_a": group_a, "group_b": group_b,
            "tests": [],
            "caveat": "Two distinct groups are needed for a comparison.",
        })

    # A capped hash-ordered sample per group. Pulling every basket into Python
    # took nearly nine seconds on the larger segments, and the rank tests gain
    # nothing from more: at 200,000 baskets the p-value is already pinned at
    # zero, which is exactly why the effect size is the figure to read. The
    # ordering is a hash of the basket id, so the sample is spread across the
    # whole period rather than taken from one end of it, and is reproducible.
    sample_cap = 20000

    def basket_values(group):
        rows = _query(f"""
            SELECT TOP {sample_cap} SUM(f.sales_value) AS basket_value
            {_from(needs)}{_clause(where + [column + " = %s"])}
            GROUP BY f.basket_id
            ORDER BY ABS(CHECKSUM(f.basket_id))
        """, params + [group])
        return [float(r["basket_value"] or 0) for r in rows]

    def basket_total(group):
        return int(_scalar_row(f"""
            SELECT COUNT(DISTINCT f.basket_id) AS baskets
            {_from(needs)}{_clause(where + [column + " = %s"])}
        """, params + [group]).get("baskets") or 0)

    values_a, values_b = basket_values(group_a), basket_values(group_b)
    total_a, total_b = basket_total(group_a), basket_total(group_b)
    sampled = len(values_a) < total_a or len(values_b) < total_b
    tests = []
    if len(values_a) >= 20 and len(values_b) >= 20:
        import numpy as np
        median_a, median_b = float(np.median(values_a)), float(np.median(values_b))
        statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
        delta = _cliffs_delta(values_a, values_b)
        higher, lower = (group_a, group_b) if median_a >= median_b else (group_b, group_a)
        gap = abs(median_a - median_b)
        matters = _delta_label(delta) not in ("negligible", "small")
        tests.append({
            "name": "Mann-Whitney U",
            "question": "Do the two groups spend differently per basket?",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_name": "Cliff's delta",
            "effect": delta,
            "effect_label": _delta_label(delta),
            "detail": (
                f"{group_a}: median ${median_a:,.2f} across {total_a:,} baskets. "
                f"{group_b}: median ${median_b:,.2f} across {total_b:,} baskets."
            ),
            "why": (
                "Basket value is right-skewed, so a rank test is used instead of a "
                "t-test, which assumes a normal distribution this data does not have."
            ),
            "headline": (
                f"{higher} baskets are worth about ${gap:,.2f} more than {lower}"
                if gap >= 0.005 else
                f"{group_a} and {group_b} baskets are worth about the same"
            ),
            "verdict": "acted-on" if matters else "too-small",
            "plain": (
                f"The difference is real, and big enough to plan around: "
                f"{_delta_label(delta)} on a standard scale."
                if matters else
                f"There is a difference, but it is {_delta_label(delta)} - too small on its "
                "own to justify treating these groups differently."
            ),
        })

        ks_statistic, ks_p = ks_2samp(values_a, values_b)
        tests.append({
            "name": "Kolmogorov-Smirnov",
            "question": "Do the two spend distributions have different shapes?",
            "statistic": float(ks_statistic),
            "p_value": float(ks_p),
            "effect_name": "D statistic",
            "effect": float(ks_statistic),
            "effect_label": _delta_label(float(ks_statistic)),
            "detail": (
                "D is the widest gap between the two cumulative distributions, so "
                "it doubles as the effect size."
            ),
            "why": (
                "A different question from the rank test: two groups can share a "
                "median while one has a far longer tail of large baskets."
            ),
            "headline": (
                f"The two spending patterns differ in shape by {float(ks_statistic):.0%}"
            ),
            "verdict": "acted-on" if _delta_label(float(ks_statistic)) not in ("negligible", "small") else "too-small",
            "plain": (
                "D is the widest gap between the two groups' spending curves. "
                f"At {float(ks_statistic):.0%} the shapes are "
                + ("clearly different." if float(ks_statistic) >= 0.33 else "broadly similar.")
            ),
        })

    mix = [] if dimension == "department" else _query(f"""
        SELECT p.department AS department, {column} AS grp,
               COUNT(DISTINCT f.basket_id) AS baskets
        {_from(needs | {"p", alias})}{_clause(where + [column + " IN (%s, %s)"])}
        GROUP BY p.department, {column}
    """, params + [group_a, group_b])
    # Comparing two departments and then asking whether they buy from different
    # departments is circular: it can only return a perfect association, which
    # is a property of the question rather than a finding about the data.
    departments = sorted({r["department"] for r in mix})
    if dimension != "department" and len(departments) >= 2:
        table = [
            [next((int(r["baskets"]) for r in mix
                   if r["department"] == dept and r["grp"] == grp), 0)
             for dept in departments]
            for grp in (group_a, group_b)
        ]
        # Departments neither group touches would leave an all-zero column, which
        # chi-square cannot take.
        keep = [i for i in range(len(departments)) if table[0][i] + table[1][i] > 0]
        table = [[row[i] for i in keep] for row in table]
        if len(keep) >= 2 and all(sum(row) > 0 for row in table):
            chi2, chi_p, _, _ = chi2_contingency(table)
            total = sum(sum(row) for row in table)
            # For a 2 x k table Cramer's V reduces to sqrt(chi2 / n).
            cramers_v = float((chi2 / total) ** 0.5) if total else 0.0
            tests.append({
                "name": "Chi-square",
                "question": "Do the two groups buy from different departments?",
                "statistic": float(chi2),
                "p_value": float(chi_p),
                "effect_name": "Cramer's V",
                "effect": cramers_v,
                "effect_label": _delta_label(cramers_v),
                "detail": f"Basket counts across {len(keep)} departments, {total:,} baskets in total.",
                "why": (
                    "Chi-square suits counts in categories. Cramer's V rescales it to "
                    "0-1 so the strength does not simply grow with the sample size."
                ),
                "headline": (
                    f"{group_a} and {group_b} shop across departments "
                    + ("quite differently" if cramers_v >= 0.33 else "in much the same way")
                ),
                "verdict": "acted-on" if _delta_label(cramers_v) not in ("negligible", "small") else "too-small",
                "plain": (
                    "Cramer's V runs 0 to 1: 0 means the two groups spread their baskets "
                    f"across departments identically, 1 means they never overlap. This is {cramers_v:.2f}."
                ),
            })

    return JsonResponse({
        "success": True,
        "dimension": dimension,
        "dimension_label": label,
        "options": options[:40],
        "group_a": group_a,
        "group_b": group_b,
        "tests": tests,
        "baskets_a": total_a,
        "baskets_b": total_b,
        "sampled": sampled,
        "sample_size": sample_cap,
        "caveat": (
            "At this sample size a p-value below 0.05 is close to guaranteed, so it "
            "only confirms a difference exists. The effect size is what says whether "
            "it is large enough to matter."
        ),
    })


# ---------------------------------------------------------------------------
# Decision panels
#
# Each of the following answers a question the earlier panels could not: they
# report levels, these report concentration, direction and mix.  All of them
# read the same filter context, so a department or segment chosen anywhere
# narrows them too.
# ---------------------------------------------------------------------------

# The calendar divides into 23 complete 30-day periods.  Growth compares the
# last two of them, so both sides of the comparison cover the same number of
# trading days; comparing against a partial period would read as a collapse.
COMPLETE_PERIODS = 23


@admin_required
def api_bi_growth(request):
    """Revenue change by department between the last two complete 30-day periods.

    This is a like-for-like comparison of two equal windows, not a trend line
    or a forecast.  A department can move because demand moved or because the
    assortment did; the panel says which departments changed, not why.
    """
    dimension = (request.GET.get("dimension") or "department").strip()
    column = {
        "department": "p.department",
        "commodity": "p.commodity",
        "store": "CAST(f.store_id AS varchar(20))",
        "segment": "COALESCE(h.rfm_segment, 'Unsegmented')",
    }.get(dimension, "p.department")
    alias = column.split(".")[0]
    where, params, needs = _filters(request, ["d", alias])
    current, previous = COMPLETE_PERIODS, COMPLETE_PERIODS - 1
    rows = _query(f"""
        SELECT TOP 40
            {column} AS label,
            SUM(CASE WHEN d.forecast_period = %s THEN f.sales_value ELSE 0 END) AS current_revenue,
            SUM(CASE WHEN d.forecast_period = %s THEN f.sales_value ELSE 0 END) AS previous_revenue,
            COUNT(DISTINCT CASE WHEN d.forecast_period = %s THEN f.basket_id END) AS current_baskets,
            COUNT(DISTINCT CASE WHEN d.forecast_period = %s THEN f.basket_id END) AS previous_baskets
        {_from(needs)}
        {_clause(where + ["d.forecast_period IN (%s, %s)"])}
        GROUP BY {column}
        ORDER BY SUM(CASE WHEN d.forecast_period = %s THEN f.sales_value ELSE 0 END) DESC
    """, [current, previous, current, previous] + params + [current, previous, current])
    for row in rows:
        prior = float(row["previous_revenue"] or 0)
        now = float(row["current_revenue"] or 0)
        row["change"] = now - prior
        row["change_pct"] = ((now - prior) / prior * 100) if prior > 0 else None
    rows = [r for r in rows if (r["current_revenue"] or r["previous_revenue"])]
    rows.sort(key=lambda r: r["change"], reverse=True)
    return JsonResponse({
        "success": True,
        "rows": rows,
        "current_period": current,
        "previous_period": previous,
        "note": (
            f"Period {current} against period {previous}, each a complete 30-day window."
        ),
    })


@admin_required
def api_bi_pareto(request):
    """How much of revenue the best-selling products account for.

    Products are ranked by revenue and the running share is reported at each
    rank, so the curve shows what share of the range earns what share of the
    money.  It is a description of concentration, not an argument for delisting
    anything: a product can be small and still be why a basket was opened.
    """
    where, params, needs = _filters(request, ["p"])
    revenues = [
        float(r["revenue"] or 0)
        for r in _query(f"""
            SELECT SUM(f.sales_value) AS revenue
            {_from(needs)}{_clause(where)}
            GROUP BY f.product_id
            ORDER BY revenue DESC
        """, params)
        if (r["revenue"] or 0) > 0
    ]
    products = len(revenues)
    total = sum(revenues)
    # One point per product would be a payload nobody can read; the curve is
    # sampled to about 200 points and always keeps the first and last.
    stride = max(1, products // 200)
    points, running, milestones = [], 0.0, {}
    pending = {50: None, 80: None, 90: None}
    for index, value in enumerate(revenues, start=1):
        running += value
        share = running / total * 100 if total else 0
        for level in pending:
            if pending[level] is None and share >= level:
                pending[level] = (index, index / products * 100)
        if index == 1 or index == products or index % stride == 0:
            points.append({
                "rank_no": index,
                "product_share": index / products * 100 if products else 0,
                "revenue_share": share,
            })
    for level, hit in pending.items():
        if hit:
            milestones[f"count_for_{level}pct"] = hit[0]
            milestones[f"products_for_{level}pct"] = round(hit[1], 2)
    return JsonResponse({
        "success": True,
        "points": points,
        "products": products,
        "total_revenue": total,
        "milestones": milestones,
    })


@admin_required
def api_bi_household_value(request):
    """Revenue share by household spend decile.

    Households are split into ten equal groups by what they spent in the current
    selection, so each band holds the same number of households and the bars
    compare their contribution.  Decile 1 is the heaviest.
    """
    where, params, needs = _filters(request, ["h"])
    rows = _query(f"""
        ;WITH spend AS (
            SELECT f.household_key, SUM(f.sales_value) AS revenue,
                   COUNT(DISTINCT f.basket_id) AS baskets
            {_from(needs)}
            {_clause(where + ["f.household_key IS NOT NULL"])}
            GROUP BY f.household_key
        ), banded AS (
            SELECT revenue, baskets,
                   NTILE(10) OVER (ORDER BY revenue DESC) AS decile
            FROM spend
        )
        SELECT decile,
               COUNT(*)      AS households,
               SUM(revenue)  AS revenue,
               SUM(baskets)  AS baskets,
               AVG(revenue)  AS avg_revenue
        FROM banded
        GROUP BY decile
        ORDER BY decile
    """, params)
    total = sum(float(r["revenue"] or 0) for r in rows) or 1.0
    running = 0.0
    for row in rows:
        value = float(row["revenue"] or 0)
        running += value
        row["revenue_share"] = value / total * 100
        row["cumulative_share"] = running / total * 100
        row["avg_baskets"] = (row["baskets"] / row["households"]) if row["households"] else 0
    return JsonResponse({"success": True, "rows": rows, "total_revenue": total})


@admin_required
def api_bi_heatmap(request):
    """Revenue by weekday and hour.

    The separate weekday and hour panels each average over the other, which
    hides the combinations that actually drive staffing: a busy Saturday
    afternoon and a quiet Tuesday one land in the same weekday bar.
    """
    where, params, needs = _filters(request, ["d"])
    rows = _query(f"""
        SELECT d.day_name, d.day_sort, f.trans_hour AS hour,
               SUM(f.sales_value)          AS revenue,
               COUNT(DISTINCT f.basket_id) AS baskets
        {_from(needs)}
        {_clause(where + ["f.trans_hour IS NOT NULL"])}
        GROUP BY d.day_name, d.day_sort, f.trans_hour
        ORDER BY d.day_sort, f.trans_hour
    """, params)
    peak = max(rows, key=lambda r: float(r["revenue"] or 0), default=None)
    return JsonResponse({
        "success": True,
        "rows": rows,
        "peak": ({"day": peak["day_name"], "hour": int(peak["hour"]),
                  "revenue": float(peak["revenue"])} if peak else None),
    })


@admin_required
def api_bi_repeat(request):
    """New against returning households, period by period.

    "New" means the first 30-day period in which a household appears *inside the
    current selection*, so filtering to a department reports households new to
    that department.  The data begins at period 1 with no history before it, so
    every household there counts as new and that period is marked as such rather
    than being read as a recruitment spike.
    """
    where, params, needs = _filters(request, ["d"])
    rows = _query(f"""
        ;WITH activity AS (
            SELECT f.household_key, d.forecast_period AS period,
                   SUM(f.sales_value) AS revenue,
                   COUNT(DISTINCT f.basket_id) AS baskets
            {_from(needs)}
            {_clause(where + ["f.household_key IS NOT NULL", "d.forecast_period IS NOT NULL"])}
            GROUP BY f.household_key, d.forecast_period
        ), first_seen AS (
            SELECT household_key, MIN(period) AS first_period
            FROM activity GROUP BY household_key
        )
        SELECT a.period,
               SUM(CASE WHEN a.period = s.first_period THEN 1 ELSE 0 END)             AS new_households,
               SUM(CASE WHEN a.period > s.first_period THEN 1 ELSE 0 END)             AS returning_households,
               SUM(CASE WHEN a.period = s.first_period THEN a.revenue ELSE 0 END)     AS new_revenue,
               SUM(CASE WHEN a.period > s.first_period THEN a.revenue ELSE 0 END)     AS returning_revenue
        FROM activity a
        JOIN first_seen s ON s.household_key = a.household_key
        GROUP BY a.period
        ORDER BY a.period
    """, params)
    for row in rows:
        total = float(row["new_revenue"] or 0) + float(row["returning_revenue"] or 0)
        row["revenue"] = total
        row["returning_share"] = (
            float(row["returning_revenue"] or 0) / total * 100 if total else 0
        )
        row["is_first_period"] = int(row["period"]) == 1
    return JsonResponse({
        "success": True,
        "rows": rows,
        "note": "Period 1 has no earlier history, so every household in it counts as new.",
    })


@admin_required
def api_bi_discount_mix(request):
    """Revenue by discount depth, and how the discount was given.

    Lines are banded by how much of the pre-discount price was taken off.  This
    describes where the money sits, not what discounting causes: deep-discount
    lines are not proof that discounting created the demand, because the lines
    that get discounted are chosen, not drawn at random.
    """
    where, params, needs = _filters(request)
    rows = _query(f"""
        ;WITH lines AS (
            SELECT f.sales_value, f.quantity, f.used_coupon,
                   f.retail_discount, f.coupon_discount,
                   CASE
                     WHEN f.gross_before_discount <= 0 THEN -1
                     ELSE f.total_discount / f.gross_before_discount
                   END AS depth
            {_from(needs)}{_clause(where)}
        )
        SELECT band, SUM(sales_value) AS revenue, COUNT(*) AS lines,
               SUM(quantity) AS units,
               SUM(CASE WHEN used_coupon = 1 THEN 1 ELSE 0 END) AS coupon_lines,
               SUM(retail_discount) AS retail_discount,
               SUM(coupon_discount) AS coupon_discount
        FROM (
            SELECT sales_value, quantity, used_coupon, retail_discount, coupon_discount,
                   -- Keyed rather than labelled: a literal per-cent sign in
                   -- the SQL collides with the parameter placeholders.
                   CASE
                     WHEN depth <= 0        THEN 'none'
                     WHEN depth < 0.10      THEN 'lt10'
                     WHEN depth < 0.25      THEN 'mid'
                     WHEN depth < 0.50      THEN 'deep'
                     ELSE 'deepest'
                   END AS band
            FROM lines
        ) banded
        GROUP BY band
    """, params)
    labels = {
        "none": "No discount", "lt10": "Under 10%", "mid": "10-25%",
        "deep": "25-50%", "deepest": "50% or more",
    }
    order = list(labels)
    rows.sort(key=lambda r: order.index(r["band"]) if r["band"] in order else 99)
    for row in rows:
        row["band"] = labels.get(row["band"], row["band"])
    total = sum(float(r["revenue"] or 0) for r in rows) or 1.0
    for row in rows:
        row["revenue_share"] = float(row["revenue"] or 0) / total * 100
    return JsonResponse({"success": True, "rows": rows, "total_revenue": total})


@admin_required
def api_bi_brand_mix(request):
    """Private-label against national-brand share, department by department.

    The overall brand split hides where own-label actually competes: a chain can
    sit at a third private label overall while running near zero in one aisle and
    over half in another.
    """
    where, params, needs = _filters(request, ["p"])
    rows = _query(f"""
        SELECT TOP 30
            p.department,
            SUM(CASE WHEN p.brand = 'Private' THEN f.sales_value ELSE 0 END)  AS private_revenue,
            SUM(CASE WHEN p.brand = 'National' THEN f.sales_value ELSE 0 END) AS national_revenue,
            SUM(f.sales_value) AS revenue
        {_from(needs)}{_clause(where)}
        GROUP BY p.department
        ORDER BY SUM(f.sales_value) DESC
    """, params)
    for row in rows:
        total = float(row["revenue"] or 0)
        row["private_share"] = float(row["private_revenue"] or 0) / total * 100 if total else 0
        row["national_share"] = float(row["national_revenue"] or 0) / total * 100 if total else 0
    rows.sort(key=lambda r: r["private_share"], reverse=True)
    return JsonResponse({"success": True, "rows": rows})
