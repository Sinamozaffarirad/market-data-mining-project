"""Business-intelligence dashboard over the Power BI star-schema views.

Serves the drill-down reporting layer directly from Django instead of an
external BI tool: Department -> Commodity -> Sub-commodity -> Product on one
axis, Year -> Quarter -> Month -> Week -> Day on the other, sliced by store,
customer segment and calendar year.

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

# Product hierarchy, outermost first.  ``column`` is the grouping column and
# ``label`` is what the breadcrumb shows when that level is the active drill.
PRODUCT_LEVELS = [
    {"key": "department", "column": "p.department", "label": "Department"},
    {"key": "commodity", "column": "p.commodity", "label": "Commodity"},
    {"key": "sub_commodity", "column": "p.sub_commodity", "label": "Sub-commodity"},
    {"key": "product", "column": "CAST(p.product_id AS varchar(20))", "label": "Product"},
]

# Calendar hierarchy.  ``select`` builds the label, ``group`` is what the query
# groups and orders by so months stay chronological rather than alphabetical.
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


def _filters(request):
    """Read the global slicers and return SQL predicates plus parameters.

    Slicers are shared by every panel so a filtered figure never sits next to
    an unfiltered one.
    """
    where, params = [], []
    year = (request.GET.get("year") or "").strip()
    department = (request.GET.get("department") or "").strip()
    store = (request.GET.get("store") or "").strip()
    segment = (request.GET.get("segment") or "").strip()

    if year and year.isdigit():
        where.append("d.calendar_year = %s")
        params.append(int(year))
    if department and department.lower() != "all":
        where.append("p.department = %s")
        params.append(department)
    if store and store.isdigit():
        where.append("f.store_id = %s")
        params.append(int(store))
    if segment and segment.lower() != "all":
        where.append("h.rfm_segment = %s")
        params.append(segment)
    return where, params


def _drill_predicates(levels, path, is_product):
    """Constrain the query to the drill path already chosen by the user."""
    where, params = [], []
    for index, value in enumerate(path):
        if index >= len(levels):
            break
        level = levels[index]
        column = level["column"] if is_product else level["select"]
        where.append(f"{column} = %s")
        params.append(value)
    return where, params


SALES_FROM = """
    FROM vw_fact_sales f
    JOIN vw_dim_date d      ON d.day_key = f.day_key
    JOIN vw_dim_product p   ON p.product_id = f.product_id
    LEFT JOIN vw_dim_household h ON h.household_key = f.household_key
"""


def _query(sql, params):
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _scalar_row(sql, params):
    rows = _query(sql, params)
    return rows[0] if rows else {}


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
    """Headline figures for the current slicer selection."""
    where, params = _filters(request)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    totals = _scalar_row(f"""
        SELECT
            COALESCE(SUM(f.sales_value), 0)        AS revenue,
            COALESCE(SUM(f.total_discount), 0)     AS discount,
            COALESCE(SUM(f.gross_before_discount), 0) AS list_value,
            COUNT(DISTINCT f.basket_id)            AS baskets,
            COUNT(DISTINCT f.household_key)        AS households,
            COUNT(DISTINCT f.product_id)           AS products,
            COUNT(DISTINCT f.store_id)             AS stores
        {SALES_FROM}{clause}
    """, params)

    revenue = float(totals.get("revenue") or 0)
    baskets = int(totals.get("baskets") or 0)
    list_value = float(totals.get("list_value") or 0)
    discount = float(totals.get("discount") or 0)

    # Basket size is distinct products per basket. dunnhumby records weighted
    # and dispensed goods in source units, so a units-based average is ~943 and
    # would be meaningless here.
    size = _scalar_row(f"""
        SELECT AVG(CAST(x.items AS float)) AS avg_items
        FROM (
            SELECT f.basket_id, COUNT(DISTINCT f.product_id) AS items
            {SALES_FROM}{clause}
            GROUP BY f.basket_id
        ) x
    """, params)

    # Revenue concentration in the top 20 products: the single number that
    # characterises how top-heavy this catalogue is.
    concentration = _scalar_row(f"""
        SELECT COALESCE(SUM(top_rev), 0) AS top20
        FROM (
            SELECT TOP 20 SUM(f.sales_value) AS top_rev
            {SALES_FROM}{clause}
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
            "households": int(totals.get("households") or 0),
            "products": int(totals.get("products") or 0),
            "stores": int(totals.get("stores") or 0),
            "avg_basket_value": revenue / baskets if baskets else 0,
            "avg_basket_size": float(size.get("avg_items") or 0),
            "revenue_per_household": (
                revenue / int(totals["households"]) if totals.get("households") else 0
            ),
            "discount_rate": discount / list_value if list_value else 0,
            "top20_concentration": top20 / revenue if revenue else 0,
        },
    })


def _drill(request, levels, is_product):
    """Shared drill-down handler for both hierarchies."""
    path = [v for v in request.GET.getlist("path") if v != ""]
    depth = min(len(path), len(levels) - 1)
    level = levels[depth]

    where, params = _filters(request)
    drill_where, drill_params = _drill_predicates(levels, path, is_product)
    where += drill_where
    params += drill_params
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    label_sql = level["column"] if is_product else level["select"]
    group_sql = level["column"] if is_product else level["group"]
    # Grouping by the sort key keeps calendar levels chronological; the label is
    # functionally dependent on it, so it is aggregated rather than grouped.
    select_label = label_sql if is_product else f"MIN({label_sql})"
    order_sql = "revenue DESC" if is_product else f"{group_sql} ASC"

    rows = _query(f"""
        SELECT TOP 40
            {select_label} AS label,
            SUM(f.sales_value)          AS revenue,
            COUNT(DISTINCT f.basket_id) AS baskets,
            COUNT(DISTINCT f.product_id) AS products,
            COUNT(DISTINCT f.day_key)   AS days
        {SALES_FROM}{clause}
        GROUP BY {group_sql}
        ORDER BY {order_sql}
    """, params)

    total = sum(float(r["revenue"] or 0) for r in rows)
    # Calendar buckets are not all the same length: the 711-day window clips the
    # first and last quarter, and the fifth week of a month is a 1-3 day
    # remnant.  Reporting the days covered, and the revenue per day, keeps a
    # short bucket from reading as a downturn.
    full = max((int(r["days"] or 0) for r in rows), default=0)
    for row in rows:
        row["revenue"] = float(row["revenue"] or 0)
        row["share"] = (row["revenue"] / total) if total else 0
        row["days"] = int(row["days"] or 0)
        row["revenue_per_day"] = row["revenue"] / row["days"] if row["days"] else 0
        # Only a material shortfall counts.  Quarters legitimately run 90 to 92
        # days and months 28 to 31, so a strict "fewer days than the longest"
        # test would brand every February and first quarter as truncated; the
        # cases worth flagging are the clipped end of the window and a month's
        # 1-3 day fifth week.
        row["partial"] = bool(
            not is_product and full and row["days"] < full * 0.9
        )

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
    where, params = _filters(request)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = _query(f"""
        SELECT TOP 200
            f.store_id,
            SUM(f.sales_value)          AS revenue,
            COUNT(DISTINCT f.basket_id) AS baskets
        {SALES_FROM}{clause}
        GROUP BY f.store_id
        ORDER BY revenue DESC
    """, params)
    for row in rows:
        row["revenue"] = float(row["revenue"] or 0)
        row["avg_basket"] = row["revenue"] / row["baskets"] if row["baskets"] else 0
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_segments(request):
    """Revenue and household count per RFM segment."""
    where, params = _filters(request)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = _query(f"""
        SELECT
            COALESCE(h.rfm_segment, 'Unsegmented') AS segment,
            SUM(f.sales_value)                     AS revenue,
            COUNT(DISTINCT f.household_key)        AS households,
            COUNT(DISTINCT f.basket_id)            AS baskets
        {SALES_FROM}{clause}
        GROUP BY COALESCE(h.rfm_segment, 'Unsegmented')
        ORDER BY revenue DESC
    """, params)
    for row in rows:
        row["revenue"] = float(row["revenue"] or 0)
        row["revenue_per_household"] = (
            row["revenue"] / row["households"] if row["households"] else 0
        )
    return JsonResponse({"success": True, "rows": rows})


@admin_required
def api_bi_basket_distribution(request):
    """How many baskets hold N distinct products.

    Every bar is one basket size.  An earlier version folded everything from 25
    upwards into a single bucket, which stacked a long thin tail into a bar
    taller than sizes 10-24 and read as a second mode at 25.  The tail is
    returned unbucketed so bar heights stay comparable, and the caller is given
    the tail's own summary to state in words instead of drawing it as one bar.
    """
    where, params = _filters(request)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = _query(f"""
        SELECT items AS bucket, COUNT(*) AS baskets, SUM(value) AS revenue
        FROM (
            SELECT f.basket_id,
                   COUNT(DISTINCT f.product_id) AS items,
                   SUM(f.sales_value) AS value
            {SALES_FROM}{clause}
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
    # Where the bars become too short to read, describe the remainder instead.
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
def api_bi_insights(request):
    """Rule-based findings for the current selection.

    Each statement is derived from the same filtered queries that feed the
    charts, so the narrative cannot drift from what is on screen.
    """
    where, params = _filters(request)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    insights = []

    quarters = _query(f"""
        SELECT MIN(CAST(d.calendar_year AS varchar(4)) + ' ' + d.quarter_name) AS label,
               SUM(f.sales_value) AS revenue
        {SALES_FROM}{clause}
        GROUP BY d.calendar_year, d.calendar_quarter
        ORDER BY d.calendar_year, d.calendar_quarter
    """, params)
    if len(quarters) >= 2:
        # First and last quarters of the 711-day window are partial, so the
        # comparison uses only the complete ones.
        complete = quarters[1:-1] if len(quarters) > 2 else quarters
        best = max(complete, key=lambda r: float(r["revenue"] or 0))
        worst = min(complete, key=lambda r: float(r["revenue"] or 0))
        spread = float(best["revenue"] or 0) - float(worst["revenue"] or 0)
        insights.append({
            "kind": "season",
            "title": f"{best['label']} is the strongest complete quarter",
            "detail": (
                f"It brings ${float(best['revenue']):,.0f}, "
                f"${spread:,.0f} more than {worst['label']}. "
                "The first and last quarters are excluded because the 711-day "
                "window starts and ends mid-quarter."
            ),
        })

    departments = _query(f"""
        SELECT TOP 3 p.department AS label, SUM(f.sales_value) AS revenue
        {SALES_FROM}{clause}
        GROUP BY p.department ORDER BY revenue DESC
    """, params)
    total_row = _scalar_row(f"SELECT SUM(f.sales_value) AS revenue {SALES_FROM}{clause}", params)
    total = float(total_row.get("revenue") or 0)
    if departments and total:
        top_share = sum(float(d["revenue"] or 0) for d in departments) / total
        insights.append({
            "kind": "concentration",
            "title": f"{departments[0]['label']} leads the catalogue",
            "detail": (
                f"The top three departments ({', '.join(d['label'] for d in departments)}) "
                f"take {top_share:.1%} of revenue in this selection."
            ),
        })

    # Build this one from the predicate list rather than splicing text into the
    # rendered clause: doing the latter emitted a second WHERE, and left the
    # parameter count short, as soon as any slicer was set.
    segment_where = where + ["h.rfm_segment IS NOT NULL"]
    segment_clause = " WHERE " + " AND ".join(segment_where)
    segments = _query(f"""
        SELECT TOP 1 h.rfm_segment AS label,
               SUM(f.sales_value) AS revenue,
               COUNT(DISTINCT f.household_key) AS households
        {SALES_FROM}{segment_clause}
        GROUP BY h.rfm_segment ORDER BY revenue DESC
    """, params)
    if segments and total:
        seg = segments[0]
        insights.append({
            "kind": "segment",
            "title": f"{seg['label']} households drive the most revenue",
            "detail": (
                f"{int(seg['households']):,} households generate "
                f"${float(seg['revenue']):,.0f}, "
                f"{float(seg['revenue']) / total:.1%} of the selection."
            ),
        })

    discount = _scalar_row(f"""
        SELECT COALESCE(SUM(f.total_discount), 0) AS discount,
               COALESCE(SUM(f.gross_before_discount), 0) AS list_value
        {SALES_FROM}{clause}
    """, params)
    list_value = float(discount.get("list_value") or 0)
    if list_value:
        rate = float(discount.get("discount") or 0) / list_value
        insights.append({
            "kind": "discount",
            "title": f"{rate:.1%} of list value is given away as discount",
            "detail": (
                f"${float(discount['discount']):,.0f} of promotions against "
                f"${list_value:,.0f} at list price."
            ),
        })

    return JsonResponse({"success": True, "insights": insights})
