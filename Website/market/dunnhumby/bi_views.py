
from __future__ import annotations

import logging

from django.db import connection
from django.http import JsonResponse
from django.shortcuts import render

from .views import _bi_filter_options, admin_required


logger = logging.getLogger(__name__)

def _from(needs):
    sql = ["FROM vw_fact_sales f"]
    if "d" in needs:
        sql.append("JOIN vw_dim_date d ON d.day_key = f.day_key")
    if "p" in needs:
        sql.append("JOIN vw_dim_product p ON p.product_id = f.product_id")
    if "h" in needs:
        sql.append("LEFT JOIN vw_dim_household h ON h.household_key = f.household_key")
    return "\n    ".join(sql)

FILTER_COLUMNS = {
    "year": ("d.calendar_year", int),
    "quarter": ("d.quarter_name", str),
    "month": ("d.month_name", str),
    "week": ("d.week_of_month", int),
    "day": ("d.day_key", int),
    "weekday": ("d.day_name", str),
    "hour": ("f.trans_hour", int),
    "period": ("d.forecast_period", int),
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
     "value": "CAST(d.calendar_year AS varchar(4))",
     "group": "d.calendar_year", "label": "Year"},
    {"key": "quarter", "select": "d.quarter_name", "value": "d.quarter_name",
     "group": "d.calendar_quarter", "label": "Quarter"},
    {"key": "month", "select": "d.month_name", "value": "d.month_name",
     "group": "d.calendar_month", "label": "Month"},
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


def _predicate(request, key, column, cast):
    """One filter as a predicate and its parameters, or None when unset."""
    raw = (request.GET.get(key) or "").strip()
    if not raw or raw.lower() == "all":
        return None
    values = [v.strip() for v in raw.split("|") if v.strip()]
    if cast is int:
        values = [int(v) for v in values if v.lstrip("-").isdigit()]
    if not values:
        return None
    if len(values) == 1:
        return f"{column} = %s", values
    return f"{column} IN ({', '.join(['%s'] * len(values))})", values


def _filters(request, needs=()):
    where, params, required = [], [], set(needs)
    for key, (column, cast) in FILTER_COLUMNS.items():
        found = _predicate(request, key, column, cast)
        if not found:
            continue
        clause, values = found
        where.append(clause)
        params.extend(values)
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


def _rank_rows(rows, value_key="revenue", total=None):
    for row in rows:
        row[value_key] = float(row.get(value_key) or 0)
    shown = sum(row[value_key] for row in rows)
    for row in rows:
        row["share"] = (row[value_key] / shown) if shown else 0
        if total:
            row["share_of_total"] = row[value_key] / total
    return rows


def _filtered_revenue(where, params, needs):
    return float(_scalar_row(f"""
        SELECT SUM(f.sales_value) AS revenue
        {_from(needs)}{_clause(where)}
    """, params).get("revenue") or 0)


@admin_required
def bi_dashboard(request):
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

    grand_total = _filtered_revenue(where, params, needs)
    _rank_rows(rows, total=grand_total)
    full = max((int(r["days"] or 0) for r in rows), default=0)
    for row in rows:
        row["days"] = int(row["days"] or 0)
        row["revenue_per_day"] = row["revenue"] / row["days"] if row["days"] else 0
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
        "shown": len(rows),
        "filtered_revenue": grand_total,
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
        row["revenue_per_day"] = row["revenue"] / row["days"] if row["days"] else 0
    return JsonResponse({"success": True, "hours": hours, "weekdays": weekdays})


@admin_required
def api_bi_demographics(request):
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
    grand_total = _filtered_revenue(where, params, needs)
    _rank_rows(rows, total=grand_total)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return JsonResponse({
        "success": True,
        "rows": rows,
        "shown": len(rows),
        "filtered_revenue": grand_total,
    })


@admin_required
def api_bi_discount_trend(request):
    where, params, needs = _filters(request, ["d"])
    rows = _query(f"""
        SELECT d.year_month AS label,
               MAX(d.calendar_year)         AS calendar_year,
               MAX(d.month_name)            AS month_name,
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
               d.calendar_year AS yr, d.calendar_quarter AS qtr,
               SUM(f.sales_value)        AS revenue,
               COUNT(DISTINCT f.day_key) AS days
        {_from(needs | {"d"})}{clause}
        GROUP BY d.calendar_year, d.calendar_quarter, d.quarter_name
        ORDER BY d.calendar_year, d.calendar_quarter
    """, params)
    rated = [
        (r["label"], float(r["revenue"] or 0) / int(r["days"]), int(r["days"]),
         int(r["yr"]), int(r["qtr"]))
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

        movers = _query(f"""
            SELECT TOP 5
                f.product_id,
                MAX(p.commodity) AS commodity,
                SUM(CASE WHEN d.calendar_year = %s AND d.calendar_quarter = %s
                         THEN f.sales_value ELSE 0 END) AS best_revenue,
                SUM(CASE WHEN d.calendar_year = %s AND d.calendar_quarter = %s
                         THEN f.sales_value ELSE 0 END) AS worst_revenue
            {_from(needs | {"d", "p"})}
            {_clause(where + ["(d.calendar_year = %s AND d.calendar_quarter = %s)"
                              " OR (d.calendar_year = %s AND d.calendar_quarter = %s)"])}
            GROUP BY f.product_id
            ORDER BY best_revenue DESC
        """, [best[3], best[4], worst[3], worst[4]]
             + params + [best[3], best[4], worst[3], worst[4]])
        moved = []
        for row in movers:
            best_rate = float(row["best_revenue"] or 0) / best[2]
            worst_rate = float(row["worst_revenue"] or 0) / worst[2]
            if best_rate <= 0:
                continue
            change = (best_rate / worst_rate - 1) if worst_rate > 0 else None
            moved.append((int(row["product_id"]), row["commodity"] or "Unknown", change))
        if moved:
            parts = []
            for product_id, commodity, change in moved[:3]:
                if change is None:
                    parts.append(f"{product_id} ({commodity.title()}) sold nothing in {worst[0]}")
                else:
                    parts.append(
                        f"{product_id} ({commodity.title()}) "
                        f"{'up' if change >= 0 else 'down'} {abs(change):.0%}"
                    )
            insights.append({
                "kind": "season",
                "title": f"What {best[0]}'s best sellers did in {worst[0]}",
                "detail": (
                    "Same products, both quarters, compared per trading day: "
                    + " · ".join(parts) + "."
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


WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday"]
HOMEOWNER_ORDER = ["Homeowner", "Probable Owner", "Probable Renter", "Renter"]
NATURAL_ORDER_DIMENSIONS = {
    "income", "age", "household_size", "kids", "weekday", "quarter", "homeowner",
}


def _band_sort_key(value):
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
    if dimension not in NATURAL_ORDER_DIMENSIONS:
        return rows
    return sorted(rows, key=lambda row: _band_sort_key(row[key]))


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

MEASURES = {
    "basket_value": {
        "label": "Basket value",
        "unit": "money",
        "observation": "basket",
        "observations": "baskets",
        "expression": "SUM(f.sales_value)",
        "group_by": "f.basket_id",
        "key": "f.basket_id",
        "noun": "spend per basket",
        "rfm_component": "monetary value",
        "tail_noun": "unusually large baskets",
        "unit_word": "",
        "discrete": False,
        "scan_minimum": 300,
    },
    "visits": {
        "label": "Visits per household",
        "unit": "count",
        "observation": "household",
        "observations": "households",
        "expression": "COUNT(DISTINCT f.basket_id)",
        "group_by": "f.household_key",
        "key": "f.household_key",
        "noun": "shopping trips per household",
        "rfm_component": "frequency",
        "tail_noun": "households that shop far more often than the rest",
        "unit_word": " trips",
        "discrete": True,
        "scan_minimum": 30,
    },
}


def _measure(request):
    name = (request.GET.get("measure") or "basket_value").strip()
    return name if name in MEASURES else "basket_value", MEASURES.get(
        name, MEASURES["basket_value"])


def _dimension_samples(column, where, params, needs, per_group=1200, minimum=100,
                       measure=None):
    spec = measure or MEASURES["basket_value"]
    rows = _query(f"""
        ;WITH baskets AS (
            SELECT {column} AS grp, {spec['group_by']} AS unit_id,
                   {spec['expression']} AS basket_value
            {_from(needs)}{_clause(where + [column + " IS NOT NULL", column + " <> ''"])}
            GROUP BY {column}, {spec['group_by']}
        ), ranked AS (
            SELECT grp, basket_value,
                   ROW_NUMBER() OVER (PARTITION BY grp ORDER BY ABS(CHECKSUM(unit_id))) AS rn,
                   COUNT(*)     OVER (PARTITION BY grp) AS baskets
            FROM baskets
        )
        SELECT grp, basket_value FROM ranked WHERE rn <= %s AND baskets >= %s
    """, params + [per_group, minimum])
    grouped = {}
    for row in rows:
        name = (row["grp"] or "").strip()
        if name:
            grouped.setdefault(name, []).append(float(row["basket_value"] or 0))
    return {name: values for name, values in grouped.items() if len(values) >= 20}


def _cliffs_delta(first, second, sample=4000):
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
    from scipy.stats import (chi2_contingency, f_oneway, kruskal, ks_2samp,
                             mannwhitneyu, skew, t as student_t, ttest_ind)

    dimension = (request.GET.get("dimension") or "segment").strip()
    column, alias, label = COMPARISON_DIMENSIONS.get(
        dimension, COMPARISON_DIMENSIONS["segment"]
    )
    measure, spec = _measure(request)
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

    sample_cap = 20000

    def basket_values(group):
        rows = _query(f"""
            SELECT TOP {sample_cap} {spec['expression']} AS basket_value
            {_from(needs)}{_clause(where + [column + " = %s"])}
            GROUP BY {spec['group_by']}
            ORDER BY ABS(CHECKSUM({spec['key']}))
        """, params + [group])
        return [float(r["basket_value"] or 0) for r in rows]

    def basket_total(group):
        return int(_scalar_row(f"""
            SELECT COUNT(DISTINCT {spec['key']}) AS baskets
            {_from(needs)}{_clause(where + [column + " = %s"])}
        """, params + [group]).get("baskets") or 0)

    def basket_median(group):
        row = _scalar_row(f"""
            ;WITH baskets AS (
                SELECT {spec['group_by']} AS unit_id, {spec['expression']} AS basket_value
                {_from(needs)}{_clause(where + [column + " = %s"])}
                GROUP BY {spec['group_by']}
            ), ordered AS (
                SELECT basket_value,
                       ROW_NUMBER() OVER (ORDER BY basket_value) AS rn,
                       COUNT(*)     OVER () AS n
                FROM baskets
            )
            SELECT AVG(basket_value) AS median_value
            FROM ordered WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
        """, params + [group])
        return float(row.get("median_value") or 0)

    def amount(value):
        return f"${value:,.2f}" if spec["unit"] == "money" else f"{value:,.1f}"

    units = spec["observations"]
    values_a, values_b = basket_values(group_a), basket_values(group_b)
    total_a, total_b = basket_total(group_a), basket_total(group_b)
    sampled = len(values_a) < total_a or len(values_b) < total_b
    tests = []
    if len(values_a) >= 20 and len(values_b) >= 20:
        import numpy as np
        median_a, median_b = basket_median(group_a), basket_median(group_b)
        statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
        delta = _cliffs_delta(values_a, values_b)
        higher, lower = (group_a, group_b) if median_a >= median_b else (group_b, group_a)
        gap = abs(median_a - median_b)
        matters = _delta_label(delta) not in ("negligible", "small")
        tests.append({
            "name": "Mann-Whitney U",
            "question": f"Do the two groups differ in {spec['noun']}?",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_name": "Cliff's delta",
            "effect": delta,
            "effect_label": _delta_label(delta),
            "detail": (
                f"{group_a}: median {amount(median_a)} across {total_a:,} {units}. "
                f"{group_b}: median {amount(median_b)} across {total_b:,} {units}."
            ),
            "why": (
                f"The distribution of {spec['noun']} is right-skewed, so a rank test is "
                "used instead of a t-test, which assumes a normal distribution this data "
                "does not have."
            ),
            "headline": (
                f"{higher} is higher than {lower} by about {amount(gap)}{spec['unit_word']}"
                if gap >= 0.005 else
                f"{group_a} and {group_b} sit at about the same level"
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
            "question": f"Do the two distributions of {spec['noun']} have different shapes?",
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
                f"median while one has a far longer tail of {spec['tail_noun']}."
                + (
                    " D itself is exact here, but its p-value assumes no two "
                    "observations are equal, and whole numbers repeat: about three "
                    "quarters of households share a count with another. That makes "
                    "the test reject less often than its p-value claims, never more, "
                    "so a significant result stands and a borderline one is better "
                    "read from the rank test above."
                    if spec.get("discrete") else ""
                )
            ),
            "headline": (
                f"The two distributions of {spec['noun']} differ in shape by "
                f"{float(ks_statistic):.0%}"
            ),
            "verdict": "acted-on" if _delta_label(float(ks_statistic)) not in ("negligible", "small") else "too-small",
            "plain": (
                f"D is the widest gap between the two groups' curves for {spec['noun']}. "
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
    departments = sorted({r["department"] for r in mix})
    if dimension != "department" and len(departments) >= 2:
        table = [
            [next((int(r["baskets"]) for r in mix
                   if r["department"] == dept and r["grp"] == grp), 0)
             for dept in departments]
            for grp in (group_a, group_b)
        ]
        keep = [i for i in range(len(departments)) if table[0][i] + table[1][i] > 0]
        table = [[row[i] for i in keep] for row in table]
        if len(keep) >= 2 and all(sum(row) > 0 for row in table):
            chi2, chi_p, _, _ = chi2_contingency(table)
            total = sum(sum(row) for row in table)
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
                    "This one does not follow the measure above: it always counts "
                    "baskets across departments, because what a group buys is the "
                    "same question however its spend or its visits are counted. "
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

    if len(values_a) >= 20 and len(values_b) >= 20:
        import numpy as np

        a = np.asarray(values_a, dtype=float)
        b = np.asarray(values_b, dtype=float)
        t_stat, t_p = ttest_ind(a, b, equal_var=False)
        mean_gap = float(a.mean() - b.mean())
        var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        standard_error = float((var_a + var_b) ** 0.5)
        degrees = ((var_a + var_b) ** 2 /
                   (var_a ** 2 / (len(a) - 1) + var_b ** 2 / (len(b) - 1))) if standard_error else 1
        margin = float(student_t.ppf(0.975, degrees) * standard_error) if standard_error else 0.0
        pooled = float((((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                        (len(a) + len(b) - 2)) ** 0.5)
        cohens_d = float(mean_gap / pooled) if pooled else 0.0
        worst_skew = max(abs(float(skew(a))), abs(float(skew(b))))
        d_label = ("negligible" if abs(cohens_d) < 0.2 else
                   "small" if abs(cohens_d) < 0.5 else
                   "medium" if abs(cohens_d) < 0.8 else "large")
        tests.append({
            "name": "Welch's t-test",
            "question": f"Do the two groups differ in average {spec['noun']}?",
            "statistic": float(t_stat),
            "p_value": float(t_p),
            "effect_name": "Cohen's d",
            "effect": cohens_d,
            "effect_label": d_label,
            "confidence_interval": [mean_gap - margin, mean_gap + margin],
            "confidence_level": 95,
            "detail": (
                f"Mean {group_a} {amount(a.mean())} against {group_b} {amount(b.mean())}. "
                f"95% confident the true gap lies between {amount(mean_gap - margin)} and "
                f"{amount(mean_gap + margin)}."
            ),
            "why": (
                "Welch's form is used because the two groups differ in size and spread. "
                f"It assumes roughly normal data and the skew here is {worst_skew:.1f}, so "
                "the rank test above is the one to quote; this is reported for the "
                "confidence interval, which is the figure to budget with."
            ),
            "headline": (
                f"Means differ by {amount(abs(mean_gap))}{spec['unit_word']}, "
                f"{'higher' if mean_gap > 0 else 'lower'} for {group_a}"
            ),
            "verdict": "acted-on" if d_label not in ("negligible", "small") else "too-small",
            "plain": (
                "Cohen's d is the gap measured in standard deviations. Under 0.2 is "
                f"negligible, over 0.8 is large. This is {cohens_d:.2f}."
            ),
        })

    group_samples = _dimension_samples(column, where, params, needs, measure=spec)
    if len(group_samples) >= 3:
        import numpy as np

        names = list(group_samples)
        arrays = [np.asarray(group_samples[n], dtype=float) for n in names]
        observations = int(sum(len(x) for x in arrays))

        f_stat, f_p = f_oneway(*arrays)
        grand = float(np.concatenate(arrays).mean())
        between = float(sum(len(x) * (x.mean() - grand) ** 2 for x in arrays))
        total_ss = float(sum(((x - grand) ** 2).sum() for x in arrays))
        eta_squared = (between / total_ss) if total_ss else 0.0
        eta_label = ("negligible" if eta_squared < 0.01 else
                     "small" if eta_squared < 0.06 else
                     "medium" if eta_squared < 0.14 else "large")
        tests.append({
            "name": "One-way ANOVA",
            "question": (
                f"Do all {len(names)} {label.lower()} groups differ in {spec['noun']}?"
            ),
            "statistic": float(f_stat),
            "p_value": float(f_p),
            "effect_name": "Eta squared",
            "effect": float(eta_squared),
            "effect_label": eta_label,
            "detail": (
                f"All {len(names)} groups at once ({', '.join(names[:4])}"
                f"{'...' if len(names) > 4 else ''}), {observations:,} sampled {units}."
            ),
            "why": (
                "ANOVA compares more than two groups in one test, avoiding the inflated "
                "false-positive rate of testing every pair separately. It assumes normality, "
                "so Kruskal-Wallis below is the safer read on this data."
            ),
            "headline": (
                f"{label} explains {eta_squared:.1%} of the variation in {spec['noun']}"
            ),
            "verdict": "acted-on" if eta_label not in ("negligible", "small") else "too-small",
            "plain": (
                f"Eta squared is the share of the variation in {spec['noun']} that group "
                f"membership accounts for. This is {eta_squared:.1%}; the rest is everything else."
            ),
        })

        h_stat, h_p = kruskal(*arrays)
        epsilon = float((h_stat - len(names) + 1) / (observations - len(names))) if observations > len(names) else 0.0
        epsilon = max(0.0, epsilon)
        eps_label = ("negligible" if epsilon < 0.01 else
                     "small" if epsilon < 0.06 else
                     "medium" if epsilon < 0.14 else "large")
        tests.append({
            "name": "Kruskal-Wallis",
            "question": f"Same question without assuming normal data: do the {len(names)} groups differ?",
            "statistic": float(h_stat),
            "p_value": float(h_p),
            "effect_name": "Epsilon squared",
            "effect": epsilon,
            "effect_label": eps_label,
            "detail": f"Rank-based across all {len(names)} groups, {observations:,} sampled {units}.",
            "why": (
                f"The non-parametric counterpart of ANOVA. It ranks the {units} instead of "
                f"averaging them, so the skew in {spec['noun']} does not distort it. Where "
                "the two disagree, this is the one to trust here."
            ),
            "headline": (
                f"The {len(names)} {label.lower()} groups "
                + (f"do not separate on {spec['noun']}" if eps_label == "negligible"
                   else f"separate on {spec['noun']}")
            ),
            "verdict": "acted-on" if eps_label not in ("negligible", "small") else "too-small",
            "plain": (
                "Epsilon squared rescales the test to 0-1 as a share of rank variation "
                f"explained by the group. This is {epsilon:.1%}."
            ),
        })

    if tests:
        for test, q in zip(tests, _benjamini_hochberg([t["p_value"] for t in tests])):
            test["q_value"] = q
            test["comparisons"] = len(tests)

    return JsonResponse({
        "success": True,
        "dimension": dimension,
        "dimension_label": label,
        "measure": measure,
        "measure_label": spec["label"],
        "measure_unit": spec["unit"],
        "measures": [{"key": k, "label": v["label"]} for k, v in MEASURES.items()],
        "observation_noun": spec["observations"],
        "circular": dimension == "segment",
        "circular_note": (
            f"RFM segments are cut partly on {spec['rfm_component']}, so a gap in "
            f"{spec['noun']} between two segments is partly true by definition. "
            "The size is still worth reading; it is not independent evidence that "
            "the segments differ."
        ) if dimension == "segment" else "",
        "options": options[:40],
        "group_a": group_a,
        "group_b": group_b,
        "tests": tests,
        "baskets_a": total_a,
        "baskets_b": total_b,
        "sampled": sampled,
        "sample_size": sample_cap,
        "caveat": (
            f"With this many {spec['observations']} a low p-value is almost guaranteed, "
            "so it only tells you a difference exists. The size of the difference tells "
            "you whether it is worth doing anything about."
        ),
    })


PERIOD_DAYS = 30


def _growth_windows(request):
    where, params = [], []
    for key, (column, cast) in FILTER_COLUMNS.items():
        if not column.startswith("d."):
            continue
        found = _predicate(request, key, column, cast)
        if not found:
            continue
        clause, values = found
        where.append(clause)
        params.extend(values)
    days = _query(f"""
        SELECT d.day_key AS day_key, d.forecast_period AS period
        FROM vw_dim_date d{_clause(where)}
        ORDER BY d.day_key
    """, params)
    if len(days) < 2:
        return None, None, (
            "The time filter leaves fewer than two trading days, so there is nothing "
            "to compare. Widen it to cover a longer stretch."
        )

    held = {}
    for row in days:
        if row["period"] is not None:
            period = int(row["period"])
            held[period] = held.get(period, 0) + 1
    whole = sorted(period for period, count in held.items() if count == PERIOD_DAYS)
    if len(whole) >= 2:
        current, previous = whole[-1], whole[-2]
        spans = {}
        for row in days:
            if row["period"] is None:
                continue
            period = int(row["period"])
            if period in (current, previous):
                key = int(row["day_key"])
                low, high = spans.get(period, (key, key))
                spans[period] = (min(low, key), max(high, key))
        return (
            (spans[current][0], spans[current][1], PERIOD_DAYS),
            (spans[previous][0], spans[previous][1], PERIOD_DAYS),
            f"Period {current} against period {previous}, each a complete 30-day "
            "window inside the current filter.",
        )

    keys = [int(row["day_key"]) for row in days]
    half = len(keys) // 2
    plural = "" if half == 1 else "s"
    return (
        (keys[len(keys) - half], keys[-1], half),
        (keys[0], keys[half - 1], half),
        "The filter leaves no whole 30-day period, so this compares the newer half "
        f"of the selected days against the older half, {half} trading day{plural} "
        "on each side.",
    )


@admin_required
def api_bi_growth(request):
    dimension = (request.GET.get("dimension") or "department").strip()
    column, alias = {
        "department": ("p.department", "p"),
        "commodity": ("p.commodity", "p"),
        "store": ("CAST(f.store_id AS varchar(20))", "f"),
        "segment": ("COALESCE(h.rfm_segment, 'Unsegmented')", "h"),
    }.get(dimension, ("p.department", "p"))
    sort = "percent" if (request.GET.get("sort") or "").strip() == "percent" else "absolute"
    try:
        min_revenue = max(0.0, float(request.GET.get("min_revenue") or 0))
    except (TypeError, ValueError):
        min_revenue = 0.0

    where, params, needs = _filters(request, ["d", alias])
    current, previous, note = _growth_windows(request)
    if not current:
        return JsonResponse({"success": True, "rows": [], "note": note, "window_days": 0,
                             "sort": sort, "min_revenue": min_revenue, "excluded": 0})
    span_low, span_high = min(current[0], previous[0]), max(current[1], previous[1])
    rows = _query(f"""
        SELECT TOP 200
            {column} AS label,
            SUM(CASE WHEN d.day_key BETWEEN %s AND %s THEN f.sales_value ELSE 0 END) AS current_revenue,
            SUM(CASE WHEN d.day_key BETWEEN %s AND %s THEN f.sales_value ELSE 0 END) AS previous_revenue,
            COUNT(DISTINCT CASE WHEN d.day_key BETWEEN %s AND %s THEN f.basket_id END) AS current_baskets,
            COUNT(DISTINCT CASE WHEN d.day_key BETWEEN %s AND %s THEN f.basket_id END) AS previous_baskets
        {_from(needs)}
        {_clause(where + ["d.day_key BETWEEN %s AND %s"])}
        GROUP BY {column}
        ORDER BY SUM(CASE WHEN d.day_key BETWEEN %s AND %s THEN f.sales_value ELSE 0 END)
               + SUM(CASE WHEN d.day_key BETWEEN %s AND %s THEN f.sales_value ELSE 0 END) DESC
    """,
        [current[0], current[1], previous[0], previous[1],
         current[0], current[1], previous[0], previous[1]]
        + params + [span_low, span_high,
                    current[0], current[1], previous[0], previous[1]])
    for row in rows:
        prior = float(row["previous_revenue"] or 0)
        now = float(row["current_revenue"] or 0)
        row["change"] = now - prior
        row["change_pct"] = ((now - prior) / prior * 100) if prior > 0 else None
    rows = [r for r in rows if (r["current_revenue"] or r["previous_revenue"])]

    def peak(row):
        return max(float(row["current_revenue"] or 0), float(row["previous_revenue"] or 0))

    kept = [r for r in rows if peak(r) >= min_revenue]
    excluded = len(rows) - len(kept)
    rows = kept

    def percent_key(row):
        if row["change_pct"] is not None:
            return float(row["change_pct"])
        return float("inf") if float(row["current_revenue"] or 0) > 0 else float("-inf")

    rows.sort(key=percent_key if sort == "percent" else (lambda r: r["change"]),
              reverse=True)
    return JsonResponse({
        "success": True,
        "rows": rows,
        "sort": sort,
        "min_revenue": min_revenue,
        "excluded": excluded,
        "current_window": {"first_day": current[0], "last_day": current[1], "days": current[2]},
        "previous_window": {"first_day": previous[0], "last_day": previous[1], "days": previous[2]},
        "window_days": current[2],
        "note": note,
    })


@admin_required
def api_bi_pareto(request):
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


SCAN_MIN_BASKETS = 300
SCAN_SAMPLE = 2000
SCAN_MAX_GROUPS = 6


def _benjamini_hochberg(p_values):
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    total = len(p_values)
    adjusted = [1.0] * total
    running = 1.0
    for rank, (position, value) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, value * total / rank)
        adjusted[position] = running
    return adjusted


@admin_required
def api_bi_significance_scan(request):
    from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu

    import numpy as np

    measure, spec = _measure(request)
    minimum = spec.get("scan_minimum", SCAN_MIN_BASKETS)
    scanned, skipped, dimension_rows = [], [], []
    for key, (column, alias, label) in COMPARISON_DIMENSIONS.items():
        where, params, needs = _filters(request, [alias])
        clause = _clause(where + [column + " IS NOT NULL", column + " <> ''"])
        rows = _query(f"""
            ;WITH baskets AS (
                SELECT {column} AS grp, {spec['group_by']} AS unit_id,
                       {spec['expression']} AS basket_value
                {_from(needs)}{clause}
                GROUP BY {column}, {spec['group_by']}
            ), ranked AS (
                SELECT grp, basket_value,
                       ROW_NUMBER() OVER (PARTITION BY grp ORDER BY ABS(CHECKSUM(unit_id))) AS rn,
                       COUNT(*)     OVER (PARTITION BY grp) AS baskets
                FROM baskets
            )
            SELECT grp, basket_value, baskets FROM ranked WHERE rn <= %s
        """, params + [SCAN_SAMPLE])

        groups = {}
        for row in rows:
            name = (row["grp"] or "").strip()
            if not name:
                continue
            entry = groups.setdefault(name, {
                "values": [], "baskets": int(row["baskets"] or 0), "median": 0.0,
            })
            entry["values"].append(float(row["basket_value"] or 0))

        usable = {n: g for n, g in groups.items() if g["baskets"] >= minimum}
        if len(usable) < 2:
            skipped.append(label)
            continue
        largest = sorted(usable.items(), key=lambda kv: kv[1]["baskets"], reverse=True)[:SCAN_MAX_GROUPS]

        for row in _query(f"""
            ;WITH baskets AS (
                SELECT {column} AS grp, {spec['expression']} AS basket_value
                {_from(needs)}{_clause(where + [column + " IS NOT NULL", column + " <> ''"])}
                GROUP BY {column}, {spec['group_by']}
            ), ordered AS (
                SELECT grp, basket_value,
                       ROW_NUMBER() OVER (PARTITION BY grp ORDER BY basket_value) AS rn,
                       COUNT(*)     OVER (PARTITION BY grp) AS n
                FROM baskets
            )
            SELECT grp, AVG(basket_value) AS median_value
            FROM ordered WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
            GROUP BY grp
        """, params):
            name = (row["grp"] or "").strip()
            if name in groups:
                groups[name]["median"] = float(row["median_value"] or 0)

        mix_where, mix_params, mix_needs = _filters(request, [alias, "p"])
        mix_rows = _query(f"""
            SELECT {column} AS grp, p.department AS dept,
                   COUNT(DISTINCT f.basket_id) AS baskets
            {_from(mix_needs | {"p"})}
            {_clause(mix_where + [column + " IS NOT NULL", column + " <> ''"])}
            GROUP BY {column}, p.department
        """, mix_params)
        mix = {}
        for row in mix_rows:
            name = (row["grp"] or "").strip()
            if name:
                mix.setdefault(name, {})[row["dept"] or "Unknown"] = int(row["baskets"] or 0)

        arrays = [np.asarray(g["values"], dtype=float) for _, g in largest]
        if len(arrays) >= 3 and all(len(x) >= 20 for x in arrays):
            observations = int(sum(len(x) for x in arrays))
            h_stat, h_p = kruskal(*arrays)
            epsilon = max(0.0, float((h_stat - len(arrays) + 1) / (observations - len(arrays))))
            f_stat, f_p = f_oneway(*arrays)
            grand = float(np.concatenate(arrays).mean())
            between = float(sum(len(x) * (x.mean() - grand) ** 2 for x in arrays))
            total_ss = float(sum(((x - grand) ** 2).sum() for x in arrays))
            dimension_rows.append({
                "dimension": key,
                "dimension_label": label,
                "circular": key == "segment",
                "groups": len(arrays),
                "kruskal_h": float(h_stat),
                "kruskal_p": float(h_p),
                "epsilon_squared": epsilon,
                "effect": _delta_label(epsilon ** 0.5),
                "anova_f": float(f_stat),
                "anova_p": float(f_p),
                "eta_squared": (between / total_ss) if total_ss else 0.0,
            })

        for i in range(len(largest)):
            for j in range(i + 1, len(largest)):
                name_a, a = largest[i]
                name_b, b = largest[j]
                values_a = np.asarray(a["values"], dtype=float)
                values_b = np.asarray(b["values"], dtype=float)
                if len(values_a) < 20 or len(values_b) < 20:
                    continue
                statistic, p_value = mannwhitneyu(
                    values_a, values_b, alternative="two-sided", method="asymptotic")
                delta = float(2 * statistic / (len(values_a) * len(values_b)) - 1)
                median_a = float(a["median"])
                median_b = float(b["median"])
                leader, trailer = ((name_a, name_b) if delta >= 0 else (name_b, name_a))
                mix_v, mix_label = (None, "n/a") if key == "department" else (0.0, "negligible")
                mix_a, mix_b = mix.get(name_a, {}), mix.get(name_b, {})
                columns = [d for d in set(mix_a) | set(mix_b)
                           if mix_a.get(d, 0) + mix_b.get(d, 0) > 0]
                if key != "department" and len(columns) >= 2:
                    table = [[mix_a.get(d, 0) for d in columns], [mix_b.get(d, 0) for d in columns]]
                    if all(sum(row) > 0 for row in table):
                        chi2, _, _, _ = chi2_contingency(table)
                        n = sum(sum(row) for row in table)
                        mix_v = float((chi2 / n) ** 0.5) if n else 0.0
                        mix_label = _delta_label(mix_v)
                scanned.append({
                    "circular": key == "segment",
                    "circular_note": (
                        f"RFM segments are cut partly on {spec['rfm_component']}, "
                        f"so a gap in {spec['noun']} between two segments is partly "
                        "true by definition."
                    ) if key == "segment" else "",
                    "mix_v": mix_v,
                    "mix_effect": mix_label,
                    "dimension": key,
                    "dimension_label": label,
                    "group_a": name_a,
                    "group_b": name_b,
                    "leader": leader,
                    "trailer": trailer,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "effect": _delta_label(delta),
                    "p_value": float(p_value),
                    "median_a": median_a,
                    "median_b": median_b,
                    "median_gap": abs(median_a - median_b),
                    "baskets_a": a["baskets"],
                    "baskets_b": b["baskets"],
                })

    if scanned:
        for row, q in zip(scanned, _benjamini_hochberg([r["p_value"] for r in scanned])):
            row["q_value"] = q
            row["actionable"] = q < 0.05 and row["abs_delta"] >= 0.147
    scanned.sort(key=lambda r: r["abs_delta"], reverse=True)

    notable = [r for r in scanned if r["actionable"]] if scanned else []
    dimension_rows.sort(key=lambda r: r["epsilon_squared"], reverse=True)
    return JsonResponse({
        "success": True,
        "rows": scanned,
        "dimensions": dimension_rows,
        "compared": len(scanned),
        "actionable": len(notable),
        "skipped_dimensions": skipped,
        "sample_per_group": SCAN_SAMPLE,
        "minimum_baskets": minimum,
        "measure": measure,
        "measure_label": spec["label"],
        "measure_unit": spec["unit"],
        "observation_noun": spec["observations"],
        "headline": (
            f"{len(notable)} of {len(scanned)} comparisons are big enough to be worth a look."
            if scanned else
            f"Not enough {spec['observations']} in this selection to compare any pair of groups."
        ),
        "method": (
            f"Groups are compared on {spec['noun']} and ranked by how often one group's "
            f"{spec['observation']} beats the other's. Because many pairs are checked at "
            "once, the odds of a fluke are adjusted for that. Each group uses up to "
            f"{SCAN_SAMPLE:,} sampled {spec['observations']}, and needs at least "
            f"{minimum:,} to be compared at all."
        ),
    })
