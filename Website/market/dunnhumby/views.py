from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.db import connection
from django.db.models import Sum, Count, Avg, Max
from math import sqrt
from .models import (
    Transaction, DunnhumbyProduct, Household, Campaign, Coupon,
    CouponRedemption, CampaignMember, CausalData, BasketAnalysis,
    AssociationRule, CustomerSegment
)
from .ml_models import ml_analyzer
import json
from collections import defaultdict
import threading
from django.views.decorators.csrf import csrf_exempt


def _generate_association_rules(min_support, min_confidence):
    baskets = defaultdict(list)
    transactions = Transaction.objects.values('basket_id', 'product_id')[:10000]
    for t in transactions:
        baskets[t['basket_id']].append(str(t['product_id']))

    # Get product details for enhanced display
    from django.db import connection
    product_details = {}
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT product_id, department, commodity_desc, brand, curr_size_of_product
            FROM product
            WHERE product_id IN (
                SELECT DISTINCT TOP 1000 product_id FROM transactions
            )
        """)
        for row in cursor.fetchall():
            product_details[str(row[0])] = {
                'department': row[1] or 'GENERAL',
                'commodity': row[2] or 'No Description',
                'brand': row[3] or 'Generic',
                'size': row[4] or 'N/A'
            }

    pair_counts = defaultdict(int)
    total_baskets = len(baskets)
    for items in baskets.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = tuple(sorted([items[i], items[j]]))
                pair_counts[pair] += 1

    rules = []
    for pair, count in pair_counts.items():
        support = count / (total_baskets or 1)
        if support >= min_support:
            ant_count = sum(1 for b in baskets.values() if pair[0] in b)
            confidence = count / (ant_count or 1)
            if confidence >= min_confidence:
                cons_count = sum(1 for b in baskets.values() if pair[1] in b)
                lift = confidence / (cons_count / (total_baskets or 1) or 1)

                # Get product details for antecedent and consequent
                ant_detail = product_details.get(pair[0], {
                    'department': 'GENERAL', 'commodity': f'Product {pair[0]}',
                    'brand': 'Generic', 'size': 'N/A'
                })
                cons_detail = product_details.get(pair[1], {
                    'department': 'GENERAL', 'commodity': f'Product {pair[1]}',
                    'brand': 'Generic', 'size': 'N/A'
                })

                rules.append({
                    'antecedent': [pair[0]],
                    'consequent': [pair[1]],
                    'antecedent_details': [ant_detail],
                    'consequent_details': [cons_detail],
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'rule_type': 'frequent_itemset',
                })
    return sorted(rules, key=lambda x: x['lift'], reverse=True)[:50]


def _get_data_statistics():
    return {
        'total_transactions': Transaction.objects.count(),
        'total_products': DunnhumbyProduct.objects.count(),
        'total_households': Household.objects.count(),
        'total_campaigns': Campaign.objects.count(),
        'basket_analyses': BasketAnalysis.objects.count(),
        'association_rules': AssociationRule.objects.count(),
        'customer_segments': CustomerSegment.objects.count(),
    }


@login_required(login_url='/admin/login/')
def site_index(request):
    tools = [
        { 'title': 'Shopping Basket Analysis', 'description': 'Analyze baskets, top products, and patterns', 'url': 'basket-analysis/', 'icon': '📊' },
        { 'title': 'Association Rules', 'description': 'Market basket association rules', 'url': 'association-rules/', 'icon': '🔗' },
        { 'title': 'Customer Segments', 'description': 'RFM segments & behavior', 'url': 'customer-segments/', 'icon': '👥' },
        { 'title': 'Data Management', 'description': 'View, edit, import/export data', 'url': 'data-management/', 'icon': '⚙️' },
    ]
    return render(request, 'site/index.html', { 'analysis_tools': tools })


@login_required(login_url='/admin/login/')
def basket_analysis(request):
    basket_stats = Transaction.objects.values('basket_id').annotate(
        total_items=Sum('quantity'),
        total_value=Sum('sales_value'),
        unique_products=Count('product_id', distinct=True)
    ).order_by('-total_value')[:20]

    dept_analysis = Transaction.objects.values('product_id').annotate(
        total_sales=Sum('sales_value'),
        total_transactions=Count('product_id')
    ).order_by('-total_sales')[:10]

    top_products = Transaction.objects.values('product_id').annotate(
        frequency=Count('product_id'),
        total_sales=Sum('sales_value')
    ).order_by('-frequency')[:20]

    return render(request, 'site/dunnhumby/basket_analysis.html', {
        'title': 'Shopping Basket Analysis',
        'basket_stats': basket_stats,
        'dept_analysis': dept_analysis,
        'top_products': top_products,
    })


@login_required(login_url='/admin/login/')
def association_rules(request):
    if request.method == 'POST':
        min_support = float(request.POST.get('min_support', 0.01))
        min_confidence = float(request.POST.get('min_confidence', 0.5))
        rules = _generate_association_rules(min_support, min_confidence)
        ctx = {
            'title': 'Association Rules',
            'rules': rules,
            'min_support': min_support,
            'min_confidence': min_confidence,
        }
    else:
        ctx = {
            'title': 'Association Rules',
            'rules': AssociationRule.objects.all().order_by('-lift')[:50],
        }
    return render(request, 'site/dunnhumby/association_rules.html', ctx)


@login_required(login_url='/admin/login/')
def customer_segments(request):
    segments = CustomerSegment.objects.values('rfm_segment').annotate(
        count=Count('household_key'),
        avg_spend=Avg('total_spend'),
        avg_transactions=Avg('total_transactions')
    ).order_by('-count')

    recent_customers = CustomerSegment.objects.order_by('-updated_at')[:20]

    return render(request, 'site/dunnhumby/customer_segments.html', {
        'title': 'Customer Segmentation',
        'segments': segments,
        'recent_customers': recent_customers,
    })


@login_required(login_url='/admin/login/')
def data_management(request):
    return render(request, 'site/dunnhumby/data_management.html', {
        'title': 'Database Manipulation & Management',
        'data_stats': _get_data_statistics(),
    })


@login_required(login_url='/admin/login/')
def api_get_table_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    table_name = request.POST.get('table_name')
    page = int(request.POST.get('page', 1))
    limit = int(request.POST.get('limit', 50))
    search = request.POST.get('search', '')
    filters = json.loads(request.POST.get('filters', '{}')) if request.POST.get('filters') else {}

    model_map = {
        'transactions': Transaction,
        'products': DunnhumbyProduct,
        'households': Household,
        'campaigns': Campaign,
        'basket_analysis': BasketAnalysis,
        'association_rules': AssociationRule,
        'customer_segments': CustomerSegment,
    }
    model = model_map.get(table_name)
    if not model:
        return JsonResponse({'error': 'Table not found'}, status=400)

    offset = (page - 1) * limit

    if table_name == 'transactions':
        queryset = model.objects.values(
            'basket_id', 'household_key', 'product_id', 'quantity',
            'sales_value', 'day', 'week_no', 'store_id'
        ).order_by('basket_id', 'product_id', 'day')
        if search:
            queryset = queryset.filter(basket_id__icontains=search)
        # numeric and text filters
        if filters:
            if 'household_key' in filters:
                queryset = queryset.filter(household_key__icontains=filters['household_key'])
            if 'product_id' in filters:
                queryset = queryset.filter(product_id__icontains=filters['product_id'])
            if 'sales_value_min' in filters:
                queryset = queryset.filter(sales_value__gte=float(filters['sales_value_min']))
            if 'sales_value_max' in filters:
                queryset = queryset.filter(sales_value__lte=float(filters['sales_value_max']))
            if 'day_min' in filters:
                queryset = queryset.filter(day__gte=int(filters['day_min']))
            if 'day_max' in filters:
                queryset = queryset.filter(day__lte=int(filters['day_max']))
    elif table_name == 'products':
        queryset = model.objects.values(
            'product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'
        ).order_by('product_id')
        if search:
            queryset = queryset.filter(commodity_desc__icontains=search)
        if filters:
            if 'department' in filters:
                queryset = queryset.filter(department__icontains=filters['department'])
            if 'brand' in filters:
                queryset = queryset.filter(brand__icontains=filters['brand'])
    elif table_name == 'households':
        queryset = model.objects.values(
            'household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'
        ).order_by('household_key')
        if search:
            queryset = queryset.filter(household_key__icontains=search)
        if filters:
            if 'age_desc' in filters:
                queryset = queryset.filter(age_desc__icontains=filters['age_desc'])
            if 'income_desc' in filters:
                queryset = queryset.filter(income_desc__icontains=filters['income_desc'])
    else:
        queryset = model.objects.all().order_by('pk')

    total_count = queryset.count()
    data = list(queryset[offset:offset + limit])
    return JsonResponse({
        'data': data,
        'total': total_count,
        'page': page,
        'pages': (total_count + limit - 1) // limit,
        'has_next': offset + limit < total_count,
        'has_prev': page > 1
    })


@login_required(login_url='/admin/login/')
def api_table_schema(request):
    """Return a minimal schema for a table to build dynamic filters client-side."""
    table = request.GET.get('table')
    model_map = {
        'transactions': Transaction,
        'products': DunnhumbyProduct,
        'households': Household,
        'campaigns': Campaign,
        'basket_analysis': BasketAnalysis,
        'association_rules': AssociationRule,
        'customer_segments': CustomerSegment,
    }
    model = model_map.get(table)
    if not model:
        return JsonResponse({'error': 'table not found'}, status=404)

    def field_type(f):
        t = getattr(f, 'get_internal_type', lambda: 'TextField')()
        if t in ('IntegerField','BigIntegerField','SmallIntegerField','PositiveIntegerField','FloatField','DecimalField'): return 'number'
        return 'text'

    fields = []
    include = []
    if table == 'transactions':
        include = ['basket_id','household_key','product_id','quantity','sales_value','day','week_no','store_id']
    elif table == 'products':
        include = ['product_id','commodity_desc','brand','department','manufacturer']
    elif table == 'households':
        include = ['household_key','age_desc','income_desc','homeowner_desc','hh_comp_desc']
    else:
        include = [f.name for f in model._meta.fields]

    # Map to types
    meta = {f.name: f for f in model._meta.fields}
    for name in include:
        f = meta.get(name)
        ftype = field_type(f) if f else 'text'
        fields.append({'name': name, 'type': ftype})
    return JsonResponse({'fields': fields})


@login_required(login_url='/admin/login/')
def api_basket_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    basket_id = request.POST.get('basket_id')
    if not basket_id:
        return JsonResponse({'error': 'basket_id required'}, status=400)
    try:
        qs = Transaction.objects.values('product_id','quantity','sales_value','day','store_id','household_key').filter(basket_id=basket_id).order_by('day')
        items = list(qs[:200])
        total_sales = sum(float(i.get('sales_value') or 0) for i in items)
        unique_products = list({i['product_id'] for i in items})
        prod_info = {}
        if unique_products:
            for p in DunnhumbyProduct.objects.filter(product_id__in=unique_products).values('product_id','brand','department','commodity_desc'):
                prod_info[p['product_id']] = {'brand': p['brand'], 'department': p['department'], 'commodity_desc': p['commodity_desc']}
        for i in items:
            meta = prod_info.get(i['product_id'], {})
            i['brand'] = meta.get('brand')
            i['department'] = meta.get('department')
            i['commodity_desc'] = meta.get('commodity_desc')
        resp = {
            'basket_id': basket_id,
            'total_sales': total_sales,
            'item_count': len(items),
            'unique_products': len(unique_products),
            'items': items,
        }
        return JsonResponse(resp)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_product_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    product_id = request.POST.get('product_id')
    if not product_id:
        return JsonResponse({'error': 'product_id required'}, status=400)
    try:
        agg = Transaction.objects.filter(product_id=product_id).aggregate(
            total_sales=Sum('sales_value'),
            total_txns=Count('product_id'),
            unique_households=Count('household_key', distinct=True)
        )
        top_households = list(
            Transaction.objects.filter(product_id=product_id).values('household_key').annotate(spend=Sum('sales_value')).order_by('-spend')[:5]
        )
        prod = DunnhumbyProduct.objects.filter(product_id=product_id).values('commodity_desc','brand','department','manufacturer').first() or {}
        return JsonResponse({
            'product_id': product_id,
            'meta': prod,
            'stats': {k: float(v) if v is not None and k=='total_sales' else v for k,v in agg.items()},
            'top_households': top_households,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_household_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    household_key = request.POST.get('household_key')
    if not household_key:
        return JsonResponse({'error': 'household_key required'}, status=400)
    try:
        seg = CustomerSegment.objects.filter(household_key=household_key).values(
            'rfm_segment','recency_score','frequency_score','monetary_score','total_spend','total_transactions','avg_basket_value','updated_at'
        ).first()
        recent_txns = list(Transaction.objects.filter(household_key=household_key).values(
            'basket_id','product_id','quantity','sales_value','day'
        ).order_by('-day')[:15])
        # enrich with product names
        pids = list({t['product_id'] for t in recent_txns})
        prodmap = {p['product_id']: p['commodity_desc'] for p in DunnhumbyProduct.objects.filter(product_id__in=pids).values('product_id','commodity_desc')}
        for t in recent_txns:
            t['commodity_desc'] = prodmap.get(t['product_id'])
        return JsonResponse({'household_key': household_key, 'segment': seg, 'recent_transactions': recent_txns})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)



@login_required(login_url='/admin/login/')
def api_differential_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    compare_by = request.POST.get('compare_by', 'time')
    stat_test = request.POST.get('stat_test', 'chi_square')

    try:
        import numpy as np
        from collections import defaultdict

        try:
            from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, ttest_ind
        except ImportError:
            chi2_contingency = ks_2samp = mannwhitneyu = ttest_ind = None

        QUARTER_RANGES = {
            'Q1': (1, 91),
            'Q2': (92, 182),
            'Q3': (183, 273),
            'Q4': (274, 366),
        }

        SEASON_RANGES = {
            'Winter': (1, 90),
            'Spring': (91, 181),
            'Summer': (182, 273),
            'Fall': (274, 365),
        }

        def to_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def determine_impact(pct_change):
            if pct_change >= 60:
                return 'High'
            if pct_change >= 30:
                return 'Medium'
            return 'Low'

        def format_stat_value(value, decimals=3):
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return 'N/A'

        def build_limit_clause(limit):
            """Return database-specific clauses for limiting result sets."""
            limit = max(int(limit or 0), 0)
            if limit <= 0:
                return "", ""

            vendor = connection.vendor
            if vendor in ("microsoft", "mssql", "sql_server"):
                return f"TOP {limit}", ""
            return "", f"LIMIT {limit}"

        def fetch_top_products_by_department(department, start_day, end_day, total_sales, limit=3):
            if not department:
                return []
            limit = max(int(limit or 3), 1)
            select_limit, suffix_limit = build_limit_clause(limit)
            query = f"""
                SELECT {select_limit}
                    t.product_id,
                    p.commodity_desc,
                    SUM(t.sales_value) as total_sales,
                    SUM(t.quantity) as total_quantity
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                WHERE p.department = %s
                  AND t.day BETWEEN %s AND %s
                GROUP BY t.product_id, p.commodity_desc
                ORDER BY total_sales DESC
                {suffix_limit}
            """
            with connection.cursor() as cursor:
                cursor.execute(query, [department, start_day, end_day])
                rows = cursor.fetchall()

            denominator = total_sales or sum(to_float(row[2]) for row in rows) or 1
            products = []
            for row in rows:
                products.append({
                    'product_id': int(row[0]),
                    'name': row[1] or f'Product {row[0]}',
                    'sales': round(to_float(row[2]), 2),
                    'quantity': int(row[3] or 0),
                    'share': round((to_float(row[2]) / denominator) * 100, 1)
                })
            return products

        def fetch_top_products_for_segment(segment, total_sales, limit=3):
            if not segment:
                return []
            limit = max(int(limit or 3), 1)
            select_limit, suffix_limit = build_limit_clause(limit)
            query = f"""
                SELECT {select_limit}
                    t.product_id,
                    p.commodity_desc,
                    SUM(t.sales_value) as total_sales,
                    SUM(t.quantity) as total_quantity
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                JOIN dunnhumby_customersegment cs ON cs.household_key = t.household_key
                WHERE cs.rfm_segment = %s
                GROUP BY t.product_id, p.commodity_desc
                ORDER BY total_sales DESC
                {suffix_limit}
            """
            with connection.cursor() as cursor:
                cursor.execute(query, [segment])
                rows = cursor.fetchall()

            denominator = total_sales or sum(to_float(row[2]) for row in rows) or 1
            return [{
                'product_id': int(row[0]),
                'name': row[1] or f'Product {row[0]}',
                'sales': round(to_float(row[2]), 2),
                'quantity': int(row[3] or 0),
                'share': round((to_float(row[2]) / denominator) * 100, 1)
            } for row in rows]

        def fetch_top_products_for_store(store_id, total_sales, limit=3):
            if store_id is None:
                return []
            limit = max(int(limit or 3), 1)
            select_limit, suffix_limit = build_limit_clause(limit)
            query = f"""
                SELECT {select_limit}
                    t.product_id,
                    p.commodity_desc,
                    SUM(t.sales_value) as total_sales,
                    SUM(t.quantity) as total_quantity
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                WHERE t.store_id = %s
                GROUP BY t.product_id, p.commodity_desc
                ORDER BY total_sales DESC
                {suffix_limit}
            """
            with connection.cursor() as cursor:
                cursor.execute(query, [store_id])
                rows = cursor.fetchall()

            denominator = total_sales or sum(to_float(row[2]) for row in rows) or 1
            return [{
                'product_id': int(row[0]),
                'name': row[1] or f'Product {row[0]}',
                'sales': round(to_float(row[2]), 2),
                'quantity': int(row[3] or 0),
                'share': round((to_float(row[2]) / denominator) * 100, 1)
            } for row in rows]

        def fetch_top_departments(limit=10):
            limit = max(int(limit or 10), 1)
            select_limit, suffix_limit = build_limit_clause(limit)
            query = f"""
                SELECT {select_limit}
                    p.department,
                    SUM(t.sales_value) AS total_sales
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                WHERE p.department IS NOT NULL
                GROUP BY p.department
                ORDER BY total_sales DESC
                {suffix_limit}
            """
            with connection.cursor() as cursor:
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall() if row[0]]

        def fetch_basket_totals_for_range(start_day, end_day, limit=2500):
            totals = (
                Transaction.objects
                .filter(day__gte=start_day, day__lte=end_day)
                .values('basket_id')
                .annotate(total_value=Sum('sales_value'))
                .order_by()[:limit]
            )
            return [to_float(item['total_value']) for item in totals if to_float(item['total_value']) > 0]

        def fetch_basket_totals_for_segment(segment, limit=4000):
            limit = max(int(limit or 4000), 1)
            select_limit, suffix_limit = build_limit_clause(limit)
            query = f"""
                SELECT {select_limit}
                    t.basket_id,
                    SUM(t.sales_value) as total_value
                FROM transactions t
                WHERE t.household_key IN (
                    SELECT household_key FROM dunnhumby_customersegment WHERE rfm_segment = %s
                )
                GROUP BY t.basket_id
                {suffix_limit}
            """
            with connection.cursor() as cursor:
                cursor.execute(query, [segment])
                rows = cursor.fetchall()
            return [to_float(row[1]) for row in rows if to_float(row[1]) > 0]

        def fetch_basket_totals_for_store(store_id, limit=4000):
            if store_id is None:
                return []
            limit = max(int(limit or 4000), 1)
            select_limit, suffix_limit = build_limit_clause(limit)
            query = f"""
                SELECT {select_limit}
                    t.basket_id,
                    SUM(t.sales_value) as total_value
                FROM transactions t
                WHERE t.store_id = %s
                GROUP BY t.basket_id
                {suffix_limit}
            """
            with connection.cursor() as cursor:
                cursor.execute(query, [store_id])
                rows = cursor.fetchall()
            return [to_float(row[1]) for row in rows if to_float(row[1]) > 0]

        def compute_statistics(test_name, observed=None, group_a=None, group_b=None):
            group_a = list(group_a or [])
            group_b = list(group_b or [])
            stats = {
                'p_value': None,
                'effect_size': None,
                'confidence': 0,
                'test_used': 'baseline',
                'note': '',
                'sample_sizes': {
                    'group_a': len(group_a),
                    'group_b': len(group_b),
                }
            }

            try:
                if test_name == 'chi_square' and observed is not None and chi2_contingency:
                    observed_arr = np.array(observed, dtype=float)
                    if observed_arr.size and observed_arr.sum() > 0 and observed_arr.shape[0] > 1 and observed_arr.shape[1] > 1:
                        chi2, p_value, _, _ = chi2_contingency(observed_arr)
                        n = observed_arr.sum()
                        r, c = observed_arr.shape
                        min_dim = min(r - 1, c - 1)
                        effect = sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
                        stats['p_value'] = format_stat_value(p_value)
                        stats['effect_size'] = format_stat_value(effect, decimals=2)
                        stats['confidence'] = max(50, min(99, int(round((1 - p_value) * 100))))
                        stats['test_used'] = 'chi_square'
                        stats['note'] = (
                            f"Chi-square test across {r} groups and {c} categories "
                            f"(n={int(n)} observations)."
                        )
                        return stats

                elif test_name == 't_test' and ttest_ind and len(group_a) > 1 and len(group_b) > 1:
                    group_a_np = np.array(group_a, dtype=float)
                    group_b_np = np.array(group_b, dtype=float)
                    _, p_value = ttest_ind(group_a_np, group_b_np, equal_var=False)
                    mean_diff = abs(group_a_np.mean() - group_b_np.mean())
                    pooled_std = np.sqrt((group_a_np.var(ddof=1) + group_b_np.var(ddof=1)) / 2)
                    effect = mean_diff / pooled_std if pooled_std > 0 else 0.0
                    stats['p_value'] = format_stat_value(p_value)
                    stats['effect_size'] = format_stat_value(effect, decimals=2)
                    stats['confidence'] = max(50, min(99, int(round((1 - p_value) * 100))))
                    stats['test_used'] = 't_test'
                    stats['note'] = (
                        f"Welch's t-test comparing {len(group_a_np)} vs {len(group_b_np)} baskets."
                    )
                    return stats

                elif test_name == 'mann_whitney' and mannwhitneyu and len(group_a) and len(group_b):
                    u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
                    n1, n2 = len(group_a), len(group_b)
                    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
                    stats['p_value'] = format_stat_value(p_value)
                    stats['effect_size'] = format_stat_value(abs(rank_biserial), decimals=2)
                    stats['confidence'] = max(50, min(95, int(round((1 - p_value) * 100))))
                    stats['test_used'] = 'mann_whitney'
                    stats['note'] = (
                        f"Mann-Whitney U test comparing {n1} vs {n2} baskets (rank-biserial correlation)."
                    )
                    return stats

                elif test_name == 'kolmogorov' and ks_2samp and len(group_a) and len(group_b):
                    ks_stat, p_value = ks_2samp(group_a, group_b, alternative='two-sided', mode='auto')
                    stats['p_value'] = format_stat_value(p_value)
                    stats['effect_size'] = format_stat_value(ks_stat, decimals=2)
                    stats['confidence'] = max(50, min(99, int(round((1 - p_value) * 100))))
                    stats['test_used'] = 'kolmogorov'
                    stats['note'] = (
                        f"Kolmogorov-Smirnov test on cumulative distributions ({len(group_a)} vs {len(group_b)} samples)."
                    )
                    return stats

            except Exception:
                pass

            if group_a and group_b:
                mean_a = np.mean(group_a)
                mean_b = np.mean(group_b)
                diff = abs(mean_a - mean_b)
                baseline = abs(mean_b) if mean_b else 1
                ratio = diff / baseline
                stats['note'] = (
                    f"Insufficient data to run the {test_name.replace('_', ' ')} test reliably. "
                    f"Average basket values differ by {diff:.2f} ({ratio * 100:.1f}% change)."
                )
            else:
                stats['note'] = (
                    f"Not enough samples available to evaluate the {test_name.replace('_', ' ')} test."
                )

            return stats

        def analyze_time_comparison():
            insights = []
            dept_quarters = defaultdict(dict)
            quarter_totals = defaultdict(lambda: {'sales': 0.0, 'transactions': 0})
            observed_matrix = []

            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        p.department,
                        CASE
                            WHEN t.day BETWEEN 1 AND 91 THEN 'Q1'
                            WHEN t.day BETWEEN 92 AND 182 THEN 'Q2'
                            WHEN t.day BETWEEN 183 AND 273 THEN 'Q3'
                            ELSE 'Q4'
                        END as quarter,
                        SUM(t.sales_value) as total_sales,
                        COUNT(*) as transaction_count
                    FROM transactions t
                    JOIN product p ON t.product_id = p.product_id
                    WHERE p.department IS NOT NULL
                    GROUP BY p.department,
                        CASE
                            WHEN t.day BETWEEN 1 AND 91 THEN 'Q1'
                            WHEN t.day BETWEEN 92 AND 182 THEN 'Q2'
                            WHEN t.day BETWEEN 183 AND 273 THEN 'Q3'
                            ELSE 'Q4'
                        END
                """)
                for dept, quarter, total_sales, txn_count in cursor.fetchall():
                    if not dept or not quarter:
                        continue
                    dept_quarters[dept][quarter] = {
                        'sales': to_float(total_sales),
                        'count': int(txn_count or 0),
                    }
                    quarter_totals[quarter]['sales'] += to_float(total_sales)
                    quarter_totals[quarter]['transactions'] += int(txn_count or 0)

            ordered_quarters = [q for q in ['Q1', 'Q2', 'Q3', 'Q4'] if q in quarter_totals]
            department_order = sorted(
                dept_quarters.keys(),
                key=lambda dept: sum(item['sales'] for item in dept_quarters[dept].values()),
                reverse=True
            )[:6]

            if ordered_quarters and department_order:
                for quarter in ordered_quarters:
                    observed_matrix.append([
                        dept_quarters.get(dept, {}).get(quarter, {}).get('count', 0)
                        for dept in department_order
                    ])

            insight_candidates = []
            for dept in department_order:
                quarter_data = dept_quarters.get(dept, {})
                if len(quarter_data) < 2:
                    continue
                max_quarter = max(quarter_data, key=lambda q: quarter_data[q]['sales'])
                min_quarter = min(quarter_data, key=lambda q: quarter_data[q]['sales'])
                max_sales = quarter_data[max_quarter]['sales']
                min_sales = quarter_data[min_quarter]['sales']
                if min_sales <= 0:
                    continue
                pct_diff = ((max_sales - min_sales) / min_sales) * 100
                if pct_diff < 12:
                    continue
                top_products = fetch_top_products_by_department(
                    dept,
                    QUARTER_RANGES[max_quarter][0],
                    QUARTER_RANGES[max_quarter][1],
                    max_sales
                ) if max_quarter in QUARTER_RANGES else []
                commodity_name = top_products[0]['name'] if top_products else 'Seasonal Demand'
                insight_candidates.append((pct_diff, {
                    'title': f'{max_quarter} Peak for {dept}',
                    'description': f'{dept} sales reach ${max_sales:,.2f} in {max_quarter} versus ${min_sales:,.2f} in {min_quarter} ({pct_diff:.0f}% lift)',
                    'impact': determine_impact(pct_diff),
                    'recommendation': f'Align {dept.lower()} assortment and marketing before the {max_quarter} demand spike.',
                    'department': dept,
                    'commodity': commodity_name,
                    'top_products': top_products,
                }))

            insights = [entry for _, entry in sorted(insight_candidates, key=lambda item: item[0], reverse=True)[:4]]

            quarter_average = {}
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        CASE
                            WHEN day BETWEEN 1 AND 91 THEN 'Q1'
                            WHEN day BETWEEN 92 AND 182 THEN 'Q2'
                            WHEN day BETWEEN 183 AND 273 THEN 'Q3'
                            ELSE 'Q4'
                        END as quarter,
                        COUNT(DISTINCT basket_id) as baskets,
                        SUM(sales_value) as total_sales
                    FROM transactions
                    GROUP BY CASE
                        WHEN day BETWEEN 1 AND 91 THEN 'Q1'
                        WHEN day BETWEEN 92 AND 182 THEN 'Q2'
                        WHEN day BETWEEN 183 AND 273 THEN 'Q3'
                        ELSE 'Q4'
                    END
                """)
                for quarter, baskets, total_sales in cursor.fetchall():
                    if baskets:
                        quarter_average[quarter] = to_float(total_sales) / int(baskets)
                    else:
                        quarter_average[quarter] = 0.0

            chart = {
                'labels': ordered_quarters,
                'datasets': [
                    {
                        'label': 'Total Sales',
                        'data': [round(quarter_totals[q]['sales'], 2) for q in ordered_quarters],
                        'backgroundColor': 'rgba(102, 126, 234, 0.8)',
                        'borderColor': 'rgba(102, 126, 234, 1)',
                        'borderWidth': 2,
                        'type': 'bar'
                    },
                    {
                        'label': 'Average Basket Value',
                        'data': [round(quarter_average.get(q, 0.0), 2) for q in ordered_quarters],
                        'borderColor': 'rgba(118, 75, 162, 1)',
                        'backgroundColor': 'rgba(118, 75, 162, 0.15)',
                        'borderWidth': 3,
                        'type': 'line',
                        'fill': False,
                    }
                ],
                'yAxisLabel': 'Sales ($)'
            }

            stats = {'p_value': 'N/A', 'effect_size': 'N/A', 'confidence': 0}
            if len(quarter_totals) >= 2:
                peak_quarter = max(quarter_totals, key=lambda q: quarter_totals[q]['sales'])
                low_quarter = min(quarter_totals, key=lambda q: quarter_totals[q]['sales'])
                if peak_quarter in QUARTER_RANGES and low_quarter in QUARTER_RANGES:
                    group_a = fetch_basket_totals_for_range(*QUARTER_RANGES[peak_quarter])
                    group_b = fetch_basket_totals_for_range(*QUARTER_RANGES[low_quarter])
                    stats = compute_statistics(
                        stat_test,
                        observed=observed_matrix if stat_test == 'chi_square' else None,
                        group_a=group_a[:3000],
                        group_b=group_b[:3000]
                    )

            return insights, stats, chart

        def analyze_segment_comparison():
            insights = []
            segments = list(
                CustomerSegment.objects.values('rfm_segment').annotate(
                    avg_spend=Avg('total_spend'),
                    avg_basket=Avg('avg_basket_value'),
                    count=Count('household_key')
                ).order_by('-avg_spend')
            )

            if len(segments) < 2:
                return insights, {'p_value': 'N/A', 'effect_size': 'N/A', 'confidence': 0}, {}

            high_seg = segments[0]
            low_seg = segments[-1]
            mid_seg = segments[1] if len(segments) > 2 else None

            segment_departments = defaultdict(lambda: defaultdict(lambda: {'sales': 0.0, 'count': 0}))
            segment_totals = defaultdict(float)

            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        cs.rfm_segment,
                        p.department,
                        SUM(t.sales_value) as total_sales,
                        COUNT(*) as transaction_count
                    FROM dunnhumby_customersegment cs
                    JOIN transactions t ON cs.household_key = t.household_key
                    JOIN product p ON t.product_id = p.product_id
                    WHERE cs.rfm_segment IN (%s, %s)
                    GROUP BY cs.rfm_segment, p.department
                """, [high_seg['rfm_segment'], low_seg['rfm_segment']])
                for segment, department, sales, txn_count in cursor.fetchall():
                    if not department:
                        continue
                    segment_departments[segment][department]['sales'] += to_float(sales)
                    segment_departments[segment][department]['count'] += int(txn_count or 0)
                    segment_totals[segment] += to_float(sales)

            spend_high = to_float(high_seg['avg_spend'])
            spend_low = to_float(low_seg['avg_spend'])
            spend_diff = ((spend_high - spend_low) / spend_low * 100) if spend_low else 0
            top_products_high = fetch_top_products_for_segment(high_seg['rfm_segment'], segment_totals.get(high_seg['rfm_segment']))

            insights.append({
                'title': f'Premium Segment: {high_seg["rfm_segment"]}',
                'description': f'{high_seg["rfm_segment"]} households spend ${spend_high:,.2f} on average vs ${spend_low:,.2f} for {low_seg["rfm_segment"]} ({spend_diff:.0f}% gap).',
                'impact': determine_impact(abs(spend_diff)),
                'recommendation': f'Create tailored value propositions and premium bundles for {high_seg["rfm_segment"]}.',
                'department': high_seg['rfm_segment'],
                'commodity': top_products_high[0]['name'] if top_products_high else 'Customer Behavior',
                'top_products': top_products_high,
            })

            if mid_seg:
                basket_high = to_float(high_seg['avg_basket'])
                basket_mid = to_float(mid_seg['avg_basket'])
                if basket_mid:
                    basket_diff = ((basket_high - basket_mid) / basket_mid) * 100
                    top_products_mid = fetch_top_products_for_segment(mid_seg['rfm_segment'], segment_totals.get(mid_seg['rfm_segment']))
                    insights.append({
                        'title': f'Basket Size Gap: {high_seg["rfm_segment"]} vs {mid_seg["rfm_segment"]}',
                        'description': f'{high_seg["rfm_segment"]} baskets average ${basket_high:,.2f} versus ${basket_mid:,.2f} for {mid_seg["rfm_segment"]} ({basket_diff:.0f}% difference).',
                        'impact': determine_impact(abs(basket_diff)),
                        'recommendation': f'Promote cross-category bundles to lift {mid_seg["rfm_segment"]} basket values.',
                        'department': 'Customer Experience',
                        'commodity': top_products_mid[0]['name'] if top_products_mid else 'Basket Optimization',
                        'top_products': top_products_mid,
                    })

            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        cs.rfm_segment,
                        t.basket_id,
                        SUM(t.sales_value) as basket_total
                    FROM dunnhumby_customersegment cs
                    JOIN transactions t ON cs.household_key = t.household_key
                    WHERE cs.rfm_segment IN (%s, %s)
                    GROUP BY cs.rfm_segment, t.basket_id
                """, [high_seg['rfm_segment'], low_seg['rfm_segment']])
                rows = cursor.fetchall()

            high_values, low_values = [], []
            for segment, _, total in rows:
                if segment == high_seg['rfm_segment']:
                    high_values.append(to_float(total))
                else:
                    low_values.append(to_float(total))

            if high_values and low_values:
                freq_high = len(high_values) / max(high_seg['count'] or 1, 1)
                freq_low = len(low_values) / max(low_seg['count'] or 1, 1)
                frequency_diff = ((freq_high - freq_low) / freq_low * 100) if freq_low else 0
                insights.append({
                    'title': f'Visit Frequency Lift: {high_seg["rfm_segment"]}',
                    'description': f'{high_seg["rfm_segment"]} shoppers average {freq_high:.1f} baskets per household vs {freq_low:.1f} for {low_seg["rfm_segment"]} ({frequency_diff:.0f}% more trips).',
                    'impact': determine_impact(abs(frequency_diff)),
                    'recommendation': f'Extend loyalty incentives to nurture the {low_seg["rfm_segment"]} segment.',
                    'department': 'Customer Loyalty',
                    'commodity': 'Visit Frequency',
                    'top_products': fetch_top_products_for_segment(low_seg['rfm_segment'], segment_totals.get(low_seg['rfm_segment'])),
                })

            labels = sorted({dept for dept_counts in segment_departments.values() for dept in dept_counts.keys()})
            labels = sorted(labels, key=lambda d: sum(segment_departments[seg][d]['sales'] for seg in segment_departments if d in segment_departments[seg]), reverse=True)[:6]

            observed = []
            for segment in [high_seg['rfm_segment'], low_seg['rfm_segment']]:
                observed.append([segment_departments[segment].get(label, {}).get('count', 0) for label in labels])

            chart = {
                'labels': labels,
                'datasets': [
                    {
                        'label': high_seg['rfm_segment'],
                        'data': [round(segment_departments[high_seg['rfm_segment']].get(label, {}).get('sales', 0.0), 2) for label in labels],
                        'backgroundColor': 'rgba(40, 167, 69, 0.8)',
                        'borderColor': 'rgba(40, 167, 69, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': low_seg['rfm_segment'],
                        'data': [round(segment_departments[low_seg['rfm_segment']].get(label, {}).get('sales', 0.0), 2) for label in labels],
                        'backgroundColor': 'rgba(255, 193, 7, 0.8)',
                        'borderColor': 'rgba(255, 193, 7, 1)',
                        'borderWidth': 2
                    }
                ],
                'yAxisLabel': 'Spend ($)'
            }

            stats = compute_statistics(
                stat_test,
                observed=observed if stat_test == 'chi_square' else None,
                group_a=high_values[:3000],
                group_b=low_values[:3000]
            )

            return insights, stats, chart

        def analyze_store_comparison():
            insights = []
            store_departments = defaultdict(lambda: defaultdict(lambda: {'sales': 0.0, 'count': 0}))
            store_totals = defaultdict(float)

            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        t.store_id,
                        p.department,
                        SUM(t.sales_value) as total_sales,
                        COUNT(*) as transaction_count
                    FROM transactions t
                    JOIN product p ON t.product_id = p.product_id
                    WHERE p.department IS NOT NULL
                    GROUP BY t.store_id, p.department
                    HAVING SUM(t.sales_value) > 0
                """)
                for store_id, department, sales, txn_count in cursor.fetchall():
                    store_departments[store_id][department]['sales'] += to_float(sales)
                    store_departments[store_id][department]['count'] += int(txn_count or 0)
                    store_totals[store_id] += to_float(sales)

            if len(store_totals) < 2:
                return insights, {'p_value': 'N/A', 'effect_size': 'N/A', 'confidence': 0}, {}

            top_stores = sorted(store_totals.keys(), key=lambda sid: store_totals[sid], reverse=True)[:2]
            best_store = top_stores[0]
            runner_store = top_stores[1]

            best_store_sales = store_totals[best_store]
            best_dept = max(store_departments[best_store], key=lambda dept: store_departments[best_store][dept]['sales'])
            top_products_best = fetch_top_products_for_store(best_store, best_store_sales)

            insights.append({
                'title': f'Store #{best_store} Leads Revenue',
                'description': f'Store #{best_store} delivers ${best_store_sales:,.2f} with {best_dept} driving the majority share.',
                'impact': 'High',
                'recommendation': f'Study store #{best_store} merchandising blueprint for wider rollout.',
                'department': best_dept,
                'commodity': top_products_best[0]['name'] if top_products_best else 'Store Operations',
                'top_products': top_products_best,
            })

            engagement_high = store_departments[best_store]
            engagement_low = store_departments[runner_store]
            total_customers_high = sum(entry['count'] for entry in engagement_high.values())
            total_customers_low = sum(entry['count'] for entry in engagement_low.values())
            customer_diff = ((total_customers_high - total_customers_low) / total_customers_low * 100) if total_customers_low else 0

            insights.append({
                'title': f'Customer Reach Gap: Store #{best_store} vs #{runner_store}',
                'description': f'Store #{best_store} handles {total_customers_high:,} transactions vs {total_customers_low:,} at store #{runner_store} ({customer_diff:.0f}% lift).',
                'impact': determine_impact(abs(customer_diff)),
                'recommendation': f'Adopt engagement tactics from store #{best_store} to uplift store #{runner_store}.',
                'department': 'Store Operations',
                'commodity': 'Customer Engagement',
                'top_products': fetch_top_products_for_store(runner_store, store_totals[runner_store]),
            })

            labels = sorted({dept for store in top_stores for dept in store_departments[store].keys()}, key=lambda d: sum(store_departments[s][d]['sales'] for s in top_stores if d in store_departments[s]), reverse=True)[:6]

            observed = []
            for store_id in top_stores:
                observed.append([store_departments[store_id].get(label, {}).get('count', 0) for label in labels])

            chart = {
                'labels': labels,
                'datasets': [
                    {
                        'label': f'Store #{best_store}',
                        'data': [round(store_departments[best_store].get(label, {}).get('sales', 0.0), 2) for label in labels],
                        'backgroundColor': 'rgba(220, 53, 69, 0.8)',
                        'borderColor': 'rgba(220, 53, 69, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': f'Store #{runner_store}',
                        'data': [round(store_departments[runner_store].get(label, {}).get('sales', 0.0), 2) for label in labels],
                        'backgroundColor': 'rgba(23, 162, 184, 0.8)',
                        'borderColor': 'rgba(23, 162, 184, 1)',
                        'borderWidth': 2
                    }
                ],
                'yAxisLabel': 'Sales ($)'
            }

            group_a = fetch_basket_totals_for_store(best_store)
            group_b = fetch_basket_totals_for_store(runner_store)
            stats = compute_statistics(
                stat_test,
                observed=observed if stat_test == 'chi_square' else None,
                group_a=group_a[:3000],
                group_b=group_b[:3000]
            )

            return insights, stats, chart

        def analyze_season_comparison():
            insights = []
            dept_season = defaultdict(lambda: defaultdict(lambda: {'sales': 0.0, 'count': 0}))
            season_totals = {}

            top_departments = fetch_top_departments(limit=8)
            if not top_departments:
                stats = compute_statistics(stat_test)
                stats.setdefault('note', 'Not enough data to compare seasonal demand patterns.')
                return insights, stats, {}

            season_case = """
                CASE
                    WHEN t.day BETWEEN 1 AND 90 THEN 'Winter'
                    WHEN t.day BETWEEN 91 AND 181 THEN 'Spring'
                    WHEN t.day BETWEEN 182 AND 273 THEN 'Summer'
                    ELSE 'Fall'
                END
            """

            placeholders = ','.join(['%s'] * len(top_departments))
            params = list(top_departments)

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT
                        p.department,
                        {season_case} as season,
                        SUM(t.sales_value) as total_sales,
                        COUNT(*) as transaction_count
                    FROM transactions t
                    JOIN product p ON t.product_id = p.product_id
                    WHERE p.department IN ({placeholders})
                    GROUP BY p.department, {season_case}
                """, params)
                for dept, season, sales, txn_count in cursor.fetchall():
                    if not dept or not season:
                        continue
                    dept_season[dept][season]['sales'] += to_float(sales)
                    dept_season[dept][season]['count'] += int(txn_count or 0)

            if not dept_season:
                stats = compute_statistics(stat_test)
                stats.setdefault('note', 'Seasonal breakdowns are unavailable for the selected data sample.')
                return insights, stats, {}

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT
                        {season_case} as season,
                        SUM(t.sales_value) as total_sales,
                        COUNT(DISTINCT t.household_key) as unique_customers,
                        COUNT(DISTINCT t.basket_id) as baskets
                    FROM transactions t
                    GROUP BY {season_case}
                """)
                for season, sales, customers, baskets in cursor.fetchall():
                    if not season:
                        continue
                    season_totals[season] = {
                        'sales': to_float(sales),
                        'customers': int(customers or 0),
                        'baskets': int(baskets or 0)
                    }

            season_order = [season for season in ['Winter', 'Spring', 'Summer', 'Fall'] if season in season_totals]
            if not season_order:
                stats = compute_statistics(stat_test)
                stats.setdefault('note', 'Seasonal totals not available for comparison.')
                return insights, stats, {}

            insight_candidates = []
            for dept, seasons in dept_season.items():
                if len(seasons) < 2:
                    continue
                peak_season = max(seasons, key=lambda s: seasons[s]['sales'])
                low_season = min(seasons, key=lambda s: seasons[s]['sales'])
                peak_sales = seasons[peak_season]['sales']
                low_sales = seasons[low_season]['sales']
                if low_sales <= 0:
                    continue
                pct_diff = ((peak_sales - low_sales) / low_sales) * 100
                if pct_diff < 15:
                    continue
                top_products = fetch_top_products_by_department(
                    dept,
                    SEASON_RANGES.get(peak_season, (1, 365))[0],
                    SEASON_RANGES.get(peak_season, (1, 365))[1],
                    peak_sales
                )
                commodity_name = top_products[0]['name'] if top_products else dept
                insight_candidates.append((pct_diff, {
                    'title': f'Seasonal Peak: {dept} in {peak_season}',
                    'description': f'{dept} demand climbs to ${peak_sales:,.2f} during {peak_season} vs ${low_sales:,.2f} in {low_season} ({pct_diff:.0f}% change).',
                    'impact': determine_impact(abs(pct_diff)),
                    'recommendation': f'Prepare seasonal merchandising for {dept.lower()} before {peak_season}.',
                    'department': dept,
                    'commodity': commodity_name,
                    'top_products': top_products,
                }))

            insights = [entry for _, entry in sorted(insight_candidates, key=lambda item: item[0], reverse=True)[:4]]

            top_departments_matrix = [dept for dept in top_departments if dept in dept_season][:5]
            observed = []
            for season in season_order:
                observed.append([
                    dept_season.get(dept, {}).get(season, {}).get('count', 0)
                    for dept in top_departments_matrix
                ])

            chart = {
                'labels': season_order,
                'datasets': [
                    {
                        'label': 'Total Sales',
                        'data': [round(season_totals[s]['sales'], 2) for s in season_order],
                        'backgroundColor': 'rgba(255, 159, 64, 0.8)',
                        'borderColor': 'rgba(255, 159, 64, 1)',
                        'borderWidth': 2
                    },
                    {
                        'label': 'Average Basket Value',
                        'data': [
                            round(season_totals[s]['sales'] / season_totals[s]['baskets'], 2)
                            if season_totals[s]['baskets'] else 0
                            for s in season_order
                        ],
                        'borderColor': 'rgba(0, 123, 255, 1)',
                        'backgroundColor': 'rgba(0, 123, 255, 0.15)',
                        'borderWidth': 3,
                        'type': 'line',
                        'fill': False
                    }
                ],
                'yAxisLabel': 'Sales ($)'
            }

            peak_season = max(season_totals, key=lambda s: season_totals[s]['sales'])
            low_season = min(season_totals, key=lambda s: season_totals[s]['sales'])
            group_a = fetch_basket_totals_for_range(*SEASON_RANGES.get(peak_season, (1, 365)), limit=2000)
            group_b = fetch_basket_totals_for_range(*SEASON_RANGES.get(low_season, (1, 365)), limit=2000)

            observed_for_stats = None
            if stat_test == 'chi_square' and len(season_order) >= 2 and len(top_departments_matrix) >= 2:
                observed_for_stats = observed

            stats = compute_statistics(
                stat_test,
                observed=observed_for_stats,
                group_a=group_a[:2000],
                group_b=group_b[:2000]
            )

            return insights, stats, chart

        if compare_by == 'time':
            insights, stats, chart = analyze_time_comparison()
        elif compare_by == 'customer_segment':
            insights, stats, chart = analyze_segment_comparison()
        elif compare_by == 'store':
            insights, stats, chart = analyze_store_comparison()
        else:
            insights, stats, chart = analyze_season_comparison()

        return JsonResponse({
            'insights': insights[:5],
            'statistics': {
                'p_value': stats.get('p_value'),
                'effect_size': stats.get('effect_size'),
                'confidence': stats.get('confidence', 0),
                'test_type': stats.get('test_used', stat_test),
                'note': stats.get('note'),
                'sample_sizes': stats.get('sample_sizes', {})
            },
            'comparison_type': compare_by,
            'chart': chart
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='/admin/login/')
def api_segment_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    name = request.POST.get('rfm_segment')
    if not name:
        return JsonResponse({'error': 'rfm_segment required'}, status=400)
    try:
        qs = CustomerSegment.objects.filter(rfm_segment=name)
        agg = qs.aggregate(
            customers=Count('household_key'),
            avg_spend=Avg('total_spend'),
            avg_txns=Avg('total_transactions'),
            avg_basket=Avg('avg_basket_value')
        )
        top_households = list(
            qs.values('household_key','total_spend','total_transactions','avg_basket_value','recency_score','frequency_score','monetary_score','updated_at')
              .order_by('-total_spend')[:12]
        )
        return JsonResponse({
            'rfm_segment': name,
            'metrics': {
                'customers': agg['customers'] or 0,
                'avg_spend': float(agg['avg_spend'] or 0),
                'avg_txns': float(agg['avg_txns'] or 0),
                'avg_basket': float(agg['avg_basket'] or 0),
            },
            'top_households': top_households,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_update_record(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    table_name = request.POST.get('table_name')
    record_id = request.POST.get('record_id')
    field_data = json.loads(request.POST.get('field_data', '{}'))

    model_map = {
        'products': DunnhumbyProduct,
        'households': Household,
        'campaigns': Campaign,
        'basket_analysis': BasketAnalysis,
        'association_rules': AssociationRule,
        'customer_segments': CustomerSegment,
    }
    model = model_map.get(table_name)
    if not model:
        return JsonResponse({'success': False, 'error': 'Unsupported table'}, status=400)

    try:
        if table_name == 'products':
            record = model.objects.get(product_id=record_id)
        elif table_name == 'households':
            record = model.objects.get(household_key=record_id)
        else:
            record = model.objects.get(pk=record_id)
        for k, v in field_data.items():
            if hasattr(record, k):
                setattr(record, k, v)
        record.save()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required(login_url='/admin/login/')
def api_export_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    import csv, io
    table_name = request.POST.get('table_name')
    filters = json.loads(request.POST.get('filters', '{}'))

    model_map = {
        'transactions': Transaction,
        'products': DunnhumbyProduct,
        'households': Household,
        'campaigns': Campaign,
        'basket_analysis': BasketAnalysis,
        'association_rules': AssociationRule,
        'customer_segments': CustomerSegment,
    }
    model = model_map.get(table_name)
    if not model:
        return JsonResponse({'error': 'Table not found'}, status=400)

    output = io.StringIO()
    writer = csv.writer(output)

    # values selection for frequently used sets
    if table_name in ('transactions', 'products', 'households'):
        if table_name == 'transactions':
            data = model.objects.values(
                'basket_id', 'household_key', 'product_id', 'quantity',
                'sales_value', 'day', 'week_no', 'store_id'
            )[:1000]
        elif table_name == 'products':
            data = model.objects.values(
                'product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'
            )[:1000]
        else:
            data = model.objects.values(
                'household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'
            )[:1000]
        if not data:
            return HttpResponse('', content_type='text/csv')
        fieldnames = list(data.first().keys())
        writer.writerow(fieldnames)
        for row in data:
            writer.writerow([row.get(f, '') for f in fieldnames])
        csv_content = output.getvalue()
    else:
        qs = model.objects.all()[:1000]
        fieldnames = [f.name for f in model._meta.fields]
        writer.writerow(fieldnames)
        for rec in qs:
            writer.writerow([getattr(rec, f, '') for f in fieldnames])
        csv_content = output.getvalue()

    resp = HttpResponse(csv_content, content_type='text/csv')
    resp['Content-Disposition'] = f'attachment; filename="{table_name}_export.csv"'
    return resp


# Global variable to track training status
ml_training_status = {'is_training': False, 'progress': 0, 'message': ''}


@csrf_exempt
def predictive_analysis_api(request):
    """API endpoint for predictive market basket analysis"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        action = request.POST.get('action', 'predict')
        model_type = request.POST.get('model_type', 'neural_network')
        training_size = float(request.POST.get('training_size', 0.8))
        
        if action == 'train':
            return train_ml_models(request, model_type, training_size)
        elif action == 'predict':
            return get_predictions(request, model_type)
        elif action == 'recommendations':
            return get_recommendations(request, model_type)
        elif action == 'performance':
            return get_model_performance(request)
        else:
            return JsonResponse({'error': 'Invalid action'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def train_ml_models(request):
    """Train ML models in background"""
    global ml_training_status
    
    # Get parameters from POST data
    model_type = request.POST.get('model_type', 'neural_network')
    training_size = float(request.POST.get('training_size', 0.8))
    
    if ml_training_status['is_training']:
        return JsonResponse({
            'success': False,
            'error': 'Training already in progress',
            'status': 'training',
            'message': 'Training already in progress',
            'progress': ml_training_status['progress']
        })
    
    def train_models():
        global ml_training_status
        try:
            ml_training_status = {'is_training': True, 'progress': 10, 'message': 'Starting training...'}
            
            # Train models
            success = ml_analyzer.train_models(training_size)
            
            if success:
                ml_training_status = {'is_training': False, 'progress': 100, 'message': 'Training completed successfully!'}
            else:
                ml_training_status = {'is_training': False, 'progress': 0, 'message': 'Training failed'}
                
        except Exception as e:
            ml_training_status = {'is_training': False, 'progress': 0, 'message': f'Training error: {str(e)}'}
    
    # Start training in background thread
    thread = threading.Thread(target=train_models)
    thread.daemon = True
    thread.start()
    
    return JsonResponse({
        'success': True,
        'status': 'started',
        'message': 'Model training started in background',
        'progress': 10
    })


@csrf_exempt
def get_predictions(request):
    """Get department-level predictions"""
    # Get parameters from POST data
    model_type = request.POST.get('model_type', 'neural_network')
    try:
        predictions = ml_analyzer.get_department_predictions(model_type)
        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'predictions': predictions
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Prediction error: {str(e)}',
            'predictions': []
        })


@csrf_exempt
def get_recommendations(request):
    """Get AI-powered product recommendations"""
    # Get parameters from POST data
    model_type = request.POST.get('model_type', 'neural_network')
    top_n = int(request.POST.get('top_n', 10))
    try:
        customer_id = request.POST.get('customer_id')
        
        recommendations = ml_analyzer.predict_customer_preferences(
            model_type, customer_id, top_n
        )
        
        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'customer_id': customer_id,
            'recommendations': recommendations
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Recommendation error: {str(e)}',
            'recommendations': []
        })


@csrf_exempt
def get_model_performance(request):
    """Get model performance metrics"""
    try:
        performance = ml_analyzer.get_model_performance()
        return JsonResponse({
            'status': 'success',
            'performance': performance,
            'training_status': ml_training_status
        })
    except Exception as e:
        return JsonResponse({'error': f'Performance error: {str(e)}'}, status=500)


@csrf_exempt  
def training_status_api(request):
    """Get current training status"""
    global ml_training_status
    return JsonResponse(ml_training_status)
