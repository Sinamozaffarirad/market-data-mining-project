from django.contrib.auth.decorators import login_required
from django.db import models
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, get_object_or_404
from django.db import connection
from django.urls import reverse
from django.db.models import Sum, Count, Avg, Max, Q
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
from django.views.decorators.http import require_POST
from django.db import transaction as db_transaction

# Define table categories based on their CRUD properties
READ_ONLY_ANALYTICAL_TABLES = ['basket_analysis', 'customer_segments']
MANAGED_ANALYTICAL_TABLES = ['association_rules']


def refresh_basket_analysis_logic():
    """Refresh basket analysis data for all transactions."""
    with db_transaction.atomic():
        BasketAnalysis.objects.all().delete()
        
        baskets = Transaction.objects.values(
            'basket_id', 'household_key'
        ).annotate(
            total_items=Sum('quantity'),
            total_value=Sum('sales_value')
        ).iterator()

        batch_size = 5000
        basket_analysis_objects = []
        count = 0

        for basket in baskets:
            basket_analysis_objects.append(
                BasketAnalysis(
                    basket_id=basket['basket_id'],
                    household_key=basket['household_key'],
                    total_items=basket['total_items'] or 0,
                    total_value=basket['total_value'] or 0.0,
                    department_mix={}
                )
            )
            
            if len(basket_analysis_objects) >= batch_size:
                BasketAnalysis.objects.bulk_create(basket_analysis_objects)
                count += len(basket_analysis_objects)
                basket_analysis_objects = []
        
        if basket_analysis_objects:
            BasketAnalysis.objects.bulk_create(basket_analysis_objects)
            count += len(basket_analysis_objects)

    return count

@login_required(login_url='/admin/login/')
@require_POST
def api_refresh_basket_analysis(request):
    try:
        count = refresh_basket_analysis_logic()
        return JsonResponse({'success': True, 'message': f'Basket analysis refreshed successfully. {count} records processed.'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def _generate_association_rules(min_support, min_confidence, transaction_period='all', max_results=100):
    """
    Efficient association rules generation using database-level queries
    to handle large datasets without memory issues
    """
    from django.db import connection
    import logging

    logger = logging.getLogger(__name__)
    rules = []

    try:
        # Calculate date filter based on transaction period
        start_day = None
        if transaction_period != 'all':
            # Get the maximum day from transactions to calculate the period
            with connection.cursor() as cursor:
                cursor.execute("SELECT MAX(day) FROM transactions")
                max_day_result = cursor.fetchone()
                max_day = max_day_result[0] if max_day_result and max_day_result[0] else 365

                # Calculate start day based on period
                period_days = {
                    '1_month': 30,
                    '3_months': 90,
                    '6_months': 180,
                    '12_months': 365
                }
                days_back = period_days.get(transaction_period, 365)
                start_day = max(1, max_day - days_back + 1)
                logger.info(f"Filtering transactions from day {start_day} to {max_day} ({transaction_period})")

        with connection.cursor() as cursor:
            # First, get total number of unique baskets for support calculation
            logger.info("Starting basket count query...")
            if start_day is not None:
                basket_count_query = f"SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= {start_day}"
            else:
                basket_count_query = "SELECT COUNT(DISTINCT basket_id) FROM transactions"
            cursor.execute(basket_count_query)
            result = cursor.fetchone()
            total_baskets = result[0] if result else 0

            if total_baskets == 0:
                logger.warning("No baskets found in transactions table")
                return rules

            # Calculate minimum basket count threshold
            min_basket_count = max(1, int(total_baskets * min_support))
            logger.info(f"Total baskets: {total_baskets}, Min basket count: {min_basket_count}")

            # Validate parameters to prevent extremely long queries
            if min_basket_count < 10 and total_baskets > 100000:
                logger.warning(f"Support threshold too low for large dataset. Adjusting from {min_basket_count} to 10")
                min_basket_count = 10

            # Find frequent product pairs using a simpler SQL approach for SQL Server
            # Build the query with optional date filtering
            if start_day is not None:
                date_filter_pairs = f" WHERE t1.day >= {start_day} AND t2.day >= {start_day}"
                date_filter_single = f" WHERE day >= {start_day}"
            else:
                date_filter_pairs = ""
                date_filter_single = ""

            pairs_query = f"""
            SELECT TOP 2000
                pairs.product_a,
                pairs.product_b,
                pairs.pair_count,
                counts_a.product_count as count_a,
                counts_b.product_count as count_b
            FROM (
                SELECT
                    t1.product_id as product_a,
                    t2.product_id as product_b,
                    COUNT(DISTINCT t1.basket_id) as pair_count
                FROM transactions t1
                JOIN transactions t2 ON t1.basket_id = t2.basket_id
                    AND t1.product_id < t2.product_id
                {date_filter_pairs}
                GROUP BY t1.product_id, t2.product_id
                HAVING COUNT(DISTINCT t1.basket_id) >= %s
            ) pairs
            JOIN (
                SELECT
                    product_id,
                    COUNT(DISTINCT basket_id) as product_count
                FROM transactions
                {date_filter_single}
                GROUP BY product_id
            ) counts_a ON pairs.product_a = counts_a.product_id
            JOIN (
                SELECT
                    product_id,
                    COUNT(DISTINCT basket_id) as product_count
                FROM transactions
                {date_filter_single}
                GROUP BY product_id
            ) counts_b ON pairs.product_b = counts_b.product_id
            ORDER BY pairs.pair_count DESC
            """

            logger.info(f"Executing pairs query with min_basket_count: {min_basket_count}")
            cursor.execute(pairs_query, [min_basket_count])

            product_pairs = cursor.fetchall()
            logger.info(f"Found {len(product_pairs)} product pairs")

            # Get product details for the products we found
            if product_pairs:
                product_ids = set()
                for pair in product_pairs:
                    product_ids.add(pair[0])
                    product_ids.add(pair[1])

                product_ids_list = list(product_ids)
                placeholders = ','.join(['%s'] * len(product_ids_list))

                cursor.execute(f"""
                    SELECT product_id, department, commodity_desc, brand, curr_size_of_product
                    FROM product
                    WHERE product_id IN ({placeholders})
                """, product_ids_list)

                product_details = {}
                for row in cursor.fetchall():
                    product_details[str(row[0])] = {
                        'department': row[1] or 'GENERAL',
                        'commodity': row[2] or 'No Description',
                        'brand': row[3] or 'Generic',
                        'size': row[4] or 'N/A'
                    }
            else:
                product_details = {}

        # Process the pairs to generate rules (outside cursor context since we have all data)
        for product_a, product_b, pair_count, count_a, count_b in product_pairs:
            # Calculate metrics
            support = pair_count / total_baskets
            confidence_a_to_b = pair_count / count_a if count_a > 0 else 0
            confidence_b_to_a = pair_count / count_b if count_b > 0 else 0

            # Generate rule A -> B
            if confidence_a_to_b >= min_confidence:
                lift = confidence_a_to_b / (count_b / total_baskets) if count_b > 0 else 0

                ant_detail = product_details.get(str(product_a), {
                    'department': 'GENERAL', 'commodity': f'Product {product_a}',
                    'brand': 'Generic', 'size': 'N/A'
                })
                cons_detail = product_details.get(str(product_b), {
                    'department': 'GENERAL', 'commodity': f'Product {product_b}',
                    'brand': 'Generic', 'size': 'N/A'
                })

                rules.append({
                    'antecedent': [str(product_a)],
                    'consequent': [str(product_b)],
                    'antecedent_details': [ant_detail],
                    'consequent_details': [cons_detail],
                    'support': support,
                    'confidence': confidence_a_to_b,
                    'lift': lift,
                    'rule_type': 'product',
                    'min_support_threshold': min_support,
                    'min_confidence_threshold': min_confidence,
                    'min_lift_threshold': None,
                    'source_view': 'analysis.association_rules',
                    'metadata': {
                        'antecedent_details': [ant_detail],
                        'consequent_details': [cons_detail],
                    },
                })

            # Generate rule B -> A (if different from A -> B)
            if confidence_b_to_a >= min_confidence and confidence_b_to_a != confidence_a_to_b:
                lift = confidence_b_to_a / (count_a / total_baskets) if count_a > 0 else 0

                ant_detail = product_details.get(str(product_b), {
                    'department': 'GENERAL', 'commodity': f'Product {product_b}',
                    'brand': 'Generic', 'size': 'N/A'
                })
                cons_detail = product_details.get(str(product_a), {
                    'department': 'GENERAL', 'commodity': f'Product {product_a}',
                    'brand': 'Generic', 'size': 'N/A'
                })

                rules.append({
                    'antecedent': [str(product_b)],
                    'consequent': [str(product_a)],
                    'antecedent_details': [ant_detail],
                    'consequent_details': [cons_detail],
                    'support': support,
                    'confidence': confidence_b_to_a,
                    'lift': lift,
                    'rule_type': 'product',
                    'min_support_threshold': min_support,
                    'min_confidence_threshold': min_confidence,
                    'min_lift_threshold': None,
                    'source_view': 'analysis.association_rules',
                    'metadata': {
                        'antecedent_details': [ant_detail],
                        'consequent_details': [cons_detail],
                    },
                })

        # Sort by lift and return top N results
        all_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
        logger.info(f"Generated {len(all_rules)} total association rules, returning top {max_results}")
        return all_rules[:max_results]

    except Exception as e:
        logger.error(f"Error in _generate_association_rules: {str(e)}")
        raise e  # Re-raise to be caught by the view function


def _generate_department_association_rules(min_support, min_confidence, transaction_period='all', max_results=100):
    """
    Generate association rules at department level using all transaction data
    """
    from django.db import connection
    import logging

    logger = logging.getLogger(__name__)
    rules = []

    try:
        # Calculate date filter based on transaction period
        start_day = None
        if transaction_period != 'all':
            with connection.cursor() as cursor:
                cursor.execute("SELECT MAX(day) FROM transactions")
                max_day_result = cursor.fetchone()
                max_day = max_day_result[0] if max_day_result and max_day_result[0] else 365

                period_days = {
                    '1_month': 30,
                    '3_months': 90,
                    '6_months': 180,
                    '12_months': 365
                }
                days_back = period_days.get(transaction_period, 365)
                start_day = max(1, max_day - days_back + 1)
                logger.info(f"Filtering transactions from day {start_day} to {max_day} ({transaction_period})")

        with connection.cursor() as cursor:
            # Get total number of unique baskets for support calculation
            logger.info("Starting department-level basket count query...")
            if start_day is not None:
                basket_count_query = f"SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= {start_day}"
            else:
                basket_count_query = "SELECT COUNT(DISTINCT basket_id) FROM transactions"
            cursor.execute(basket_count_query)
            result = cursor.fetchone()
            total_baskets = result[0] if result else 0

            if total_baskets == 0:
                logger.warning("No baskets found in transactions table")
                return rules

            # Calculate minimum basket count threshold
            min_basket_count = max(1, int(total_baskets * min_support))
            logger.info(f"Total baskets: {total_baskets}, Min basket count: {min_basket_count}")

            # Build the query with optional date filtering for departments
            if start_day is not None:
                date_filter_pairs = f" AND t1.day >= {start_day} AND t2.day >= {start_day}"
                date_filter_single = f" AND t.day >= {start_day}"
            else:
                date_filter_pairs = ""
                date_filter_single = ""

            # Find frequent department pairs
            pairs_query = f"""
            SELECT TOP 1000
                pairs.dept_a,
                pairs.dept_b,
                pairs.pair_count,
                counts_a.dept_count as count_a,
                counts_b.dept_count as count_b
            FROM (
                SELECT
                    p1.department as dept_a,
                    p2.department as dept_b,
                    COUNT(DISTINCT t1.basket_id) as pair_count
                FROM transactions t1
                JOIN transactions t2 ON t1.basket_id = t2.basket_id
                    AND t1.product_id != t2.product_id
                JOIN product p1 ON t1.product_id = p1.product_id
                JOIN product p2 ON t2.product_id = p2.product_id
                WHERE p1.department < p2.department
                    AND p1.department IS NOT NULL
                    AND p2.department IS NOT NULL
                    {date_filter_pairs}
                GROUP BY p1.department, p2.department
                HAVING COUNT(DISTINCT t1.basket_id) >= %s
            ) pairs
            JOIN (
                SELECT
                    p.department,
                    COUNT(DISTINCT t.basket_id) as dept_count
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                {date_filter_single}
                WHERE p.department IS NOT NULL
                GROUP BY p.department
            ) counts_a ON pairs.dept_a = counts_a.department
            JOIN (
                SELECT
                    p.department,
                    COUNT(DISTINCT t.basket_id) as dept_count
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                {date_filter_single}
                WHERE p.department IS NOT NULL
                GROUP BY p.department
            ) counts_b ON pairs.dept_b = counts_b.department
            ORDER BY pairs.pair_count DESC
            """

            logger.info(f"Executing department pairs query with min_basket_count: {min_basket_count}")
            cursor.execute(pairs_query, [min_basket_count])

            department_pairs = cursor.fetchall()
            logger.info(f"Found {len(department_pairs)} department pairs")

        # Process the pairs to generate rules
        for dept_a, dept_b, pair_count, count_a, count_b in department_pairs:
            # Calculate metrics
            support = pair_count / total_baskets
            confidence_a_to_b = pair_count / count_a if count_a > 0 else 0
            confidence_b_to_a = pair_count / count_b if count_b > 0 else 0

            # Generate rule A -> B
            if confidence_a_to_b >= min_confidence:
                lift = confidence_a_to_b / (count_b / total_baskets) if count_b > 0 else 0

                rules.append({
                    'antecedent': dept_a,
                    'consequent': dept_b,
                    'support': support,
                    'confidence': confidence_a_to_b,
                    'lift': lift,
                    'rule_type': 'department'
                })

            # Generate rule B -> A (if different from A -> B)
            if confidence_b_to_a >= min_confidence and confidence_b_to_a != confidence_a_to_b:
                lift = confidence_b_to_a / (count_a / total_baskets) if count_a > 0 else 0

                rules.append({
                    'antecedent': dept_b,
                    'consequent': dept_a,
                    'support': support,
                    'confidence': confidence_b_to_a,
                    'lift': lift,
                    'rule_type': 'department'
                })

        # Sort by lift and return top N results
        all_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
        logger.info(f"Generated {len(all_rules)} department association rules, returning top {max_results}")
        return all_rules[:max_results]

    except Exception as e:
        logger.error(f"Error in _generate_department_association_rules: {str(e)}")
        raise e


def _generate_commodity_association_rules(min_support, min_confidence, transaction_period='all', max_results=100):
    """
    Generate association rules at commodity level using all transaction data
    """
    from django.db import connection
    import logging

    logger = logging.getLogger(__name__)
    rules = []

    try:
        # Calculate date filter based on transaction period
        start_day = None
        if transaction_period != 'all':
            with connection.cursor() as cursor:
                cursor.execute("SELECT MAX(day) FROM transactions")
                max_day_result = cursor.fetchone()
                max_day = max_day_result[0] if max_day_result and max_day_result[0] else 365

                period_days = {
                    '1_month': 30,
                    '3_months': 90,
                    '6_months': 180,
                    '12_months': 365
                }
                days_back = period_days.get(transaction_period, 365)
                start_day = max(1, max_day - days_back + 1)
                logger.info(f"Filtering transactions from day {start_day} to {max_day} ({transaction_period})")

        with connection.cursor() as cursor:
            # Get total number of unique baskets for support calculation
            logger.info("Starting commodity-level basket count query...")
            if start_day is not None:
                basket_count_query = f"SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= {start_day}"
            else:
                basket_count_query = "SELECT COUNT(DISTINCT basket_id) FROM transactions"
            cursor.execute(basket_count_query)
            result = cursor.fetchone()
            total_baskets = result[0] if result else 0

            if total_baskets == 0:
                logger.warning("No baskets found in transactions table")
                return rules

            # Calculate minimum basket count threshold - use lower threshold for commodities
            min_basket_count = max(1, int(total_baskets * min_support))
            # For commodities, use even lower threshold due to higher granularity
            if min_basket_count > 50:
                min_basket_count = max(10, min_basket_count // 5)
            logger.info(f"Total baskets: {total_baskets}, Min basket count for commodities: {min_basket_count}")

            # Build the query with optional date filtering for commodities
            if start_day is not None:
                date_filter_pairs = f" AND t1.day >= {start_day} AND t2.day >= {start_day}"
                date_filter_single = f" AND t.day >= {start_day}"
            else:
                date_filter_pairs = ""
                date_filter_single = ""

            # Find frequent commodity pairs - use simplified approach for better performance
            pairs_query = f"""
            SELECT TOP 1000
                pairs.comm_a,
                pairs.comm_b,
                pairs.pair_count,
                counts_a.comm_count as count_a,
                counts_b.comm_count as count_b
            FROM (
                SELECT
                    p1.commodity_desc as comm_a,
                    p2.commodity_desc as comm_b,
                    COUNT(DISTINCT t1.basket_id) as pair_count
                FROM transactions t1
                JOIN transactions t2 ON t1.basket_id = t2.basket_id
                    AND t1.product_id != t2.product_id
                JOIN product p1 ON t1.product_id = p1.product_id
                JOIN product p2 ON t2.product_id = p2.product_id
                WHERE p1.commodity_desc IS NOT NULL
                    AND p2.commodity_desc IS NOT NULL
                    AND p1.commodity_desc != p2.commodity_desc
                    AND p1.commodity_desc < p2.commodity_desc
                    {date_filter_pairs}
                GROUP BY p1.commodity_desc, p2.commodity_desc
                HAVING COUNT(DISTINCT t1.basket_id) >= %s
            ) pairs
            JOIN (
                SELECT
                    p.commodity_desc,
                    COUNT(DISTINCT t.basket_id) as comm_count
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                {date_filter_single}
                WHERE p.commodity_desc IS NOT NULL
                GROUP BY p.commodity_desc
            ) counts_a ON pairs.comm_a = counts_a.commodity_desc
            JOIN (
                SELECT
                    p.commodity_desc,
                    COUNT(DISTINCT t.basket_id) as comm_count
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                {date_filter_single}
                WHERE p.commodity_desc IS NOT NULL
                GROUP BY p.commodity_desc
            ) counts_b ON pairs.comm_b = counts_b.commodity_desc
            ORDER BY pairs.pair_count DESC
            """

            logger.info(f"Executing commodity pairs query with min_basket_count: {min_basket_count}")
            cursor.execute(pairs_query, [min_basket_count])

            commodity_pairs = cursor.fetchall()
            logger.info(f"Found {len(commodity_pairs)} commodity pairs")

        # Process the pairs to generate rules
        for comm_a, comm_b, pair_count, count_a, count_b in commodity_pairs:
            # Calculate metrics
            support = pair_count / total_baskets
            confidence_a_to_b = pair_count / count_a if count_a > 0 else 0
            confidence_b_to_a = pair_count / count_b if count_b > 0 else 0

            # Generate rule A -> B
            if confidence_a_to_b >= min_confidence:
                lift = confidence_a_to_b / (count_b / total_baskets) if count_b > 0 else 0

                rules.append({
                    'antecedent': comm_a,
                    'consequent': comm_b,
                    'support': support,
                    'confidence': confidence_a_to_b,
                    'lift': lift,
                    'rule_type': 'commodity'
                })

            # Generate rule B -> A (if different from A -> B)
            if confidence_b_to_a >= min_confidence and confidence_b_to_a != confidence_a_to_b:
                lift = confidence_b_to_a / (count_a / total_baskets) if count_a > 0 else 0

                rules.append({
                    'antecedent': comm_b,
                    'consequent': comm_a,
                    'support': support,
                    'confidence': confidence_b_to_a,
                    'lift': lift,
                    'rule_type': 'commodity'
                })

        # Sort by lift and return top N results
        all_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
        logger.info(f"Generated {len(all_rules)} commodity association rules, returning top {max_results}")
        return all_rules[:max_results]

    except Exception as e:
        logger.error(f"Error in _generate_commodity_association_rules: {str(e)}")
        raise e


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
        { 'title': 'Customer Insights', 'description': 'Explore and manage your customer data in detail', 'url': reverse('customers:search'), 'icon': '👤' },
    ]
    return render(request, 'site/index.html', { 'analysis_tools': tools })


@login_required(login_url='/admin/login/')
def basket_analysis(request):
    basket_stats = Transaction.objects.values('basket_id', 'household_key').annotate(
        total_items=Sum('quantity'),
        total_value=Sum('sales_value'),
        unique_products=Count('product_id', distinct=True)
    ).order_by('-total_value')[:20]

    dept_analysis = Transaction.objects.values('product_id').annotate(
        total_sales=Sum('sales_value'),
        total_transactions=Count('product_id')
    ).order_by('-total_sales')[:10]

    product_stats = Transaction.objects.values('product_id').annotate(
        frequency=Count('product_id'),
        total_sales=Sum('sales_value')
    ).filter(product_id__isnull=False)

    top_products_frequency_raw = list(product_stats.order_by('-frequency')[:20])
    top_products_sales_raw = list(product_stats.order_by('-total_sales')[:20])

    product_ids = {item['product_id'] for item in top_products_frequency_raw + top_products_sales_raw}
    product_details = {
        item['product_id']: item for item in DunnhumbyProduct.objects.filter(product_id__in=product_ids).values(
            'product_id', 'brand', 'department', 'commodity_desc', 'sub_commodity_desc', 'manufacturer'
        )
    }

    def _enrich_product_records(records):
        enriched = []
        for record in records:
            details = product_details.get(record['product_id'], {})
            enriched.append({
                'product_id': record['product_id'],
                'frequency': record['frequency'],
                'total_sales': float(record['total_sales'] or 0),
                'brand': details.get('brand') or 'Unknown Brand',
                'department': details.get('department') or 'MISC. TRANS.',
                'commodity_desc': details.get('commodity_desc') or 'All Products',
                'sub_commodity_desc': details.get('sub_commodity_desc') or '',
                'manufacturer': details.get('manufacturer') or ''
            })
        return enriched

    top_products_frequency = _enrich_product_records(top_products_frequency_raw)
    top_products_sales = _enrich_product_records(top_products_sales_raw)

    return render(request, 'site/dunnhumby/basket_analysis.html', {
        'title': 'Shopping Basket Analysis',
        'basket_stats': basket_stats,
        'dept_analysis': dept_analysis,
        'top_products_frequency': top_products_frequency,
        'top_products_sales': top_products_sales,
    })


@login_required(login_url='/admin/login/')
def association_rules(request):
    if request.method == 'POST':
        try:
            min_support = float(request.POST.get('min_support', 0.0001))
            min_confidence = float(request.POST.get('min_confidence', 0.5))
            transaction_period = request.POST.get('transaction_period', 'all')
            max_results = int(request.POST.get('max_results', 100))

            # Validate parameters
            if min_support <= 0 or min_support > 1:
                min_support = 0.0001
            if min_confidence <= 0 or min_confidence > 1:
                min_confidence = 0.5
            if transaction_period not in ['all', '1_month', '3_months', '6_months', '12_months']:
                transaction_period = 'all'
            if max_results not in [50, 100, 200, 500, 1000]:
                max_results = 100

            rules = _generate_association_rules(min_support, min_confidence, transaction_period, max_results)

            # Get period display name
            period_names = {
                'all': 'all transactions',
                '1_month': 'last 1 month',
                '3_months': 'last 3 months',
                '6_months': 'last 6 months',
                '12_months': 'last 12 months'
            }
            period_display = period_names.get(transaction_period, 'all transactions')

            ctx = {
                'title': 'Association Rules',
                'rules': rules,
                'min_support': min_support,
                'min_confidence': min_confidence,
                'transaction_period': transaction_period,
                'max_results': max_results,
                'success_message': f'Generated {len(rules)} association rules from {period_display} successfully!'
            }
        except Exception as e:
            ctx = {
                'title': 'Association Rules',
                'rules': [],
                'error_message': f'Error generating rules: {str(e)}. Please try with higher support values.',
                'min_support': 0.0001,
                'min_confidence': 0.5,
                'transaction_period': request.POST.get('transaction_period', 'all'),
                'max_results': request.POST.get('max_results', 100),
            }
    else:
        ctx = {
            'title': 'Association Rules',
            'rules': AssociationRule.objects.all().order_by('-lift')[:100],
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
@require_POST
def api_insert_association_rule(request):
    if request.content_type == 'application/json':
        try:
            payload = json.loads(request.body.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({'success': False, 'error': 'Invalid JSON payload.'}, status=400)
    else:
        payload = request.POST

    def _parse_list(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            return [item.strip() for item in value.split(',') if item.strip()]
        return []

    def _parse_float(value, fallback=None):
        if value in (None, ''):
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    try:
        antecedent = _parse_list(payload.get('antecedent'))
        consequent = _parse_list(payload.get('consequent'))
        support = float(payload.get('support'))
        confidence = float(payload.get('confidence'))
        lift = float(payload.get('lift'))
    except (TypeError, ValueError):
        return JsonResponse({'success': False, 'error': 'Invalid numeric values supplied.'}, status=400)

    if not antecedent or not consequent:
        return JsonResponse({'success': False, 'error': 'Antecedent and consequent are required.'}, status=400)

    rule_type = payload.get('rule_type') or 'product'
    if rule_type not in {'product', 'category', 'commodity', 'department'}:
        rule_type = 'product'

    min_support = _parse_float(payload.get('min_support_threshold') or payload.get('min_support'), support)
    min_confidence = _parse_float(payload.get('min_confidence_threshold') or payload.get('min_confidence'))
    min_lift = _parse_float(payload.get('min_lift_threshold') or payload.get('min_lift'))
    source_view = payload.get('source_view') or payload.get('source') or 'manual.insert'

    metadata = payload.get('metadata')
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {'raw': metadata}
    if metadata is None:
        metadata = {}

    existing = AssociationRule.objects.filter(
        antecedent=antecedent,
        consequent=consequent,
        rule_type=rule_type
    ).first()

    if existing:
        existing.support = support
        existing.confidence = confidence
        existing.lift = lift
        existing.min_support_threshold = min_support
        existing.min_confidence_threshold = min_confidence
        existing.min_lift_threshold = min_lift
        existing.source_view = source_view
        existing.metadata = metadata
        existing.save()
        return JsonResponse({'success': True, 'message': 'Rule updated.'})
    else:
        AssociationRule.objects.create(
            antecedent=antecedent,
            consequent=consequent,
            support=support,
            confidence=confidence,
            lift=lift,
            rule_type=rule_type,
            min_support_threshold=min_support,
            min_confidence_threshold=min_confidence,
            min_lift_threshold=min_lift,
            source_view=source_view,
            metadata=metadata,
        )
        return JsonResponse({'success': True, 'message': 'Rule inserted.'}, status=201)

@login_required(login_url='/admin/login/')
def api_get_table_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
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

        # Define which fields to select for each table
        field_sets = {
            'transactions': ['id','basket_id', 'household_key', 'product_id', 'quantity', 'sales_value', 'day', 'week_no', 'store_id'],
            'products': ['product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'],
            'households': ['household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'],
        }
        
        # Always start with a values() queryset to get dictionaries
        if table_name in field_sets:
            queryset = model.objects.values(*field_sets[table_name])
        else:
            # For other tables, get all fields as values
            queryset = model.objects.values()

        # Generic Search Logic
        searchable_fields = {
            'transactions': ['basket_id', 'household_key', 'product_id'],
            'products': ['commodity_desc', 'brand', 'department', 'manufacturer'],
            'households': ['household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'],
            'campaigns': ['description'],
            'customer_segments': ['rfm_segment', 'household_key'],
            'basket_analysis': ['basket_id', 'household_key']
        }
        if search and table_name in searchable_fields:
            q_objects = Q()
            for field in searchable_fields.get(table_name, []):
                try:
                    # Check if field type is numeric before attempting numeric search
                    is_numeric_field = 'int' in model._meta.get_field(field).get_internal_type().lower()
                    if search.isnumeric() and is_numeric_field:
                         q_objects |= Q(**{f"{field}": search})
                    else:
                         q_objects |= Q(**{f"{field}__icontains": search})
                except (AttributeError, ValueError):
                    # Fallback for non-model fields or casting issues
                    q_objects |= Q(**{f"{field}__icontains": search})
            if q_objects:
                queryset = queryset.filter(q_objects)


        # Generic Filter Logic
        filter_mappings = {
            'household_key_min': 'household_key__gte',
            'household_key_max': 'household_key__lte',
            'product_id_min': 'product_id__gte',
            'product_id_max': 'product_id__lte',
            'sales_value_min': 'sales_value__gte',
            'sales_value_max': 'sales_value__lte',
            'day_min': 'day__gte',
            'day_max': 'day__lte',
            'department': 'department__icontains',
            'brand': 'brand__icontains',
            'age_desc': 'age_desc__icontains',
            'income_desc': 'income_desc__icontains',
            'description': 'description__icontains',
            'rfm_segment': 'rfm_segment__icontains',
            # ADDED: Filters for Basket Analysis
            'total_items_min': 'total_items__gte',
            'total_items_max': 'total_items__lte',
            'total_value_min': 'total_value__gte',
            'total_value_max': 'total_value__lte',
        }
        
        if filters:
            filter_kwargs = {}
            for key, value in filters.items():
                if key in filter_mappings and value:
                    filter_kwargs[filter_mappings[key]] = value
            if filter_kwargs:
                queryset = queryset.filter(**filter_kwargs)
        
        # Ordering
        ordering_fields = {
            'transactions': ('-day', 'basket_id'),
            'products': ('product_id',),
            'households': ('household_key',),
            'customer_segments': ('-total_spend',)
        }
        if table_name in ordering_fields:
            queryset = queryset.order_by(*ordering_fields[table_name])
        elif hasattr(model._meta, 'pk'):
             queryset = queryset.order_by(model._meta.pk.name)


        total_count = queryset.count()
        offset = (page - 1) * limit
        
        # Simple slicing now works because queryset is always a ValuesQuerySet
        data = list(queryset[offset:offset + limit])

        return JsonResponse({
            'data': data,
            'total': total_count,
            'page': page,
            'pages': (total_count + limit - 1) // limit,
            'has_next': offset + limit < total_count,
            'has_prev': page > 1
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_table_schema(request):
    """Return a schema for a table to build dynamic filters or forms client-side."""
    table = request.GET.get('table')
    purpose = request.GET.get('purpose', 'filter') # 'filter' or 'form'
    
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

    fields_to_include = []
    if purpose == 'form':
        # For forms, we usually want all non-pk, editable fields
        fields_to_include = [f.name for f in model._meta.fields if not f.primary_key and f.editable]
        # For specific tables, we might need to add the PK field for creation
        if table == 'products':
            fields_to_include.insert(0, 'product_id')
        elif table == 'households':
            fields_to_include.insert(0, 'household_key')
    else: # purpose == 'filter'
        filterable_fields = {
            'transactions': ['household_key', 'product_id', 'sales_value', 'day'],
            'products': ['department', 'brand'],
            'households': ['age_desc', 'income_desc'],
            'campaigns': ['description'],
            'customer_segments': ['rfm_segment'],
            # ADDED: Filterable fields for Basket Analysis
            'basket_analysis': ['total_items', 'total_value']
        }
        fields_to_include = filterable_fields.get(table, [])
    
    meta = {f.name: f for f in model._meta.fields}
    fields = []
    for name in fields_to_include:
        f = meta.get(name)
        if f:
            ftype = field_type(f)
            # Use verbose_name for a user-friendly label, fallback to name
            label = getattr(f, 'verbose_name', name).title()
            fields.append({'name': name, 'type': ftype, 'label': label})
            
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


@csrf_exempt
@login_required(login_url='/admin/login/')
def api_create_record(request):
    """
    Creates a new record in the specified table.
    For 'campaigns' and 'households', it auto-increments the primary key if not provided.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required'}, status=405)
    
    table_name = request.POST.get('table_name')
    field_data = json.loads(request.POST.get('field_data', '{}'))

    # Prevent creation for read-only analytical tables
    if table_name in READ_ONLY_ANALYTICAL_TABLES:
        return JsonResponse({'success': False, 'error': f'Creation is not allowed for the {table_name} table.'}, status=403)

    model_map = {
        'products': DunnhumbyProduct, 'households': Household, 'campaigns': Campaign,
        'association_rules': AssociationRule, 'transactions': Transaction,
    }
    model = model_map.get(table_name)
    if not model:
        return JsonResponse({'success': False, 'error': 'Unsupported table for creation'}, status=400)
        
    try:
        pk_field_map = {
            'products': 'product_id', 'households': 'household_key',
            'campaigns': 'campaign',
        }
        pk_field = pk_field_map.get(table_name, 'id')

        # Auto-increment logic for specified tables
        if table_name in ['households', 'campaigns'] and pk_field not in field_data:
            last_record = model.objects.order_by(f'-{pk_field}').first()
            if last_record:
                field_data[pk_field] = getattr(last_record, pk_field) + 1
            else:
                field_data[pk_field] = 1

        record = model.objects.create(**field_data)
        
        # Determine the primary key value to return
        pk_value = getattr(record, pk_field)
            
        return JsonResponse({'success': True, 'record_id': pk_value})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@login_required(login_url='/admin/login/')
@require_POST
def api_update_record(request):
    table_name = request.POST.get('table_name')
    record_id = request.POST.get('record_id')
    field_data = json.loads(request.POST.get('field_data', '{}'))

    # Prevent updates for read-only analytical tables
    if table_name in READ_ONLY_ANALYTICAL_TABLES:
        return JsonResponse({'success': False, 'error': f'Updates are not allowed for the {table_name} table.'}, status=403)

    model_map = {
        'products': DunnhumbyProduct, 'households': Household, 'campaigns': Campaign,
        'association_rules': AssociationRule,
    }
    model = model_map.get(table_name)
    if not model:
        return JsonResponse({'success': False, 'error': 'Unsupported table for updates'}, status=400)

    try:
        pk_field_map = {
            'products': 'product_id', 'households': 'household_key',
            'campaigns': 'campaign',
        }
        pk_field = pk_field_map.get(table_name, 'id')

        if pk_field in field_data:
            del field_data[pk_field]

        # Use .update() for a more direct and reliable update
        updated_count = model.objects.filter(**{pk_field: record_id}).update(**field_data)
        
        if updated_count > 0:
            return JsonResponse({'success': True, 'message': 'Record updated successfully.'})
        else:
            return JsonResponse({'success': False, 'error': 'Record not found or no changes detected.'}, status=404)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@login_required(login_url='/admin/login/')
@require_POST
def api_delete_record(request):
    """
    Deletes a record. For 'campaigns' and 'households', it first deletes all dependent records
    to maintain data integrity. It now also handles cascading deletes between
    Transaction and BasketAnalysis tables.
    """
    table_name = request.POST.get('table_name')
    record_id = request.POST.get('record_id')

    deletable_tables = {
        'products': DunnhumbyProduct, 'households': Household, 'campaigns': Campaign,
        'basket_analysis': BasketAnalysis, 'association_rules': AssociationRule,
        'customer_segments': CustomerSegment, 'transactions': Transaction,
    }
    model = deletable_tables.get(table_name)
    if not model:
        return JsonResponse({'success': False, 'error': 'Deletion from this table is not permitted'}, status=403)

    try:
        with db_transaction.atomic():
            pk_field_map = {
                'products': 'product_id', 'households': 'household_key',
                'campaigns': 'campaign',
            }
            pk_field = pk_field_map.get(table_name, 'id')
    
            if table_name == 'campaigns':
                campaign_id = int(record_id)
                coupons_to_delete = Coupon.objects.filter(campaign=campaign_id)
                coupon_upcs = [c.coupon_upc for c in coupons_to_delete]
                if coupon_upcs:
                    CouponRedemption.objects.filter(coupon_upc__in=coupon_upcs).delete()
                CouponRedemption.objects.filter(campaign=campaign_id).delete()
                coupons_to_delete.delete()
                CampaignMember.objects.filter(campaign=campaign_id).delete()

            elif table_name == 'households':
                household_id = int(record_id)
                # This can be very slow on large datasets. Use with caution.
                Transaction.objects.filter(household_key=household_id).delete()
                CampaignMember.objects.filter(household_key=household_id).delete()
                CouponRedemption.objects.filter(household_key=household_id).delete()
                CustomerSegment.objects.filter(household_key=household_id).delete()
            
            elif table_name == 'transactions':
                # Get the transaction to be deleted
                transaction_to_delete = get_object_or_404(Transaction, id=record_id)
                basket_id_to_update = transaction_to_delete.basket_id

                # Delete the specific transaction
                transaction_to_delete.delete()

                # Check for remaining transactions in the same basket
                remaining_transactions = Transaction.objects.filter(basket_id=basket_id_to_update)

                if remaining_transactions.exists():
                    # If transactions remain, update the basket analysis
                    new_totals = remaining_transactions.aggregate(
                        total_items=Sum('quantity'),
                        total_value=Sum('sales_value')
                    )
                    BasketAnalysis.objects.filter(basket_id=basket_id_to_update).update(
                        total_items=new_totals['total_items'] or 0,
                        total_value=new_totals['total_value'] or 0.0
                    )
                else:
                    # If no transactions remain, delete the basket analysis record
                    BasketAnalysis.objects.filter(basket_id=basket_id_to_update).delete()
            
            elif table_name == 'basket_analysis':
                # Get the basket analysis record to delete
                basket_analysis_to_delete = get_object_or_404(BasketAnalysis, id=record_id)
                basket_id_to_clear = basket_analysis_to_delete.basket_id

                # Delete all associated transactions
                Transaction.objects.filter(basket_id=basket_id_to_clear).delete()

                # Delete the basket analysis record itself
                basket_analysis_to_delete.delete()

            else:
                record = model.objects.get(**{pk_field: record_id})
                record.delete()
            
        return JsonResponse({'success': True, 'message': 'Record deleted successfully.'})
    except model.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Record not found'}, status=404)
    except Exception as e:
        # This will catch any other database errors, including other potential FK constraints
        return JsonResponse({'success': False, 'error': f'An unexpected error occurred: {str(e)}'}, status=500)
    
@login_required(login_url='/admin/login/')
def api_generate_department_rules(request):
    """API endpoint for generating department-level association rules"""
    if request.method == 'POST':
        try:
            min_support = float(request.POST.get('min_support', 0.001))
            min_confidence = float(request.POST.get('min_confidence', 0.6))
            transaction_period = request.POST.get('transaction_period', 'all')
            max_results = int(request.POST.get('max_results', 100))

            # Validate parameters
            if min_support <= 0 or min_support > 1:
                min_support = 0.001
            if min_confidence <= 0 or min_confidence > 1:
                min_confidence = 0.6
            if transaction_period not in ['all', '1_month', '3_months', '6_months', '12_months']:
                transaction_period = 'all'
            if max_results not in [50, 100, 200, 500]:
                max_results = 100

            rules = _generate_department_association_rules(min_support, min_confidence, transaction_period, max_results)

            return JsonResponse({
                'success': True,
                'rules': rules,
                'count': len(rules),
                'parameters': {
                    'min_support': min_support,
                    'min_confidence': min_confidence,
                    'transaction_period': transaction_period,
                    'max_results': max_results
                }
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method allowed'}, status=405)


@login_required(login_url='/admin/login/')
def api_generate_commodity_rules(request):
    """API endpoint for generating commodity-level association rules"""
    if request.method == 'POST':
        try:
            min_support = float(request.POST.get('min_support', 0.001))
            min_confidence = float(request.POST.get('min_confidence', 0.6))
            transaction_period = request.POST.get('transaction_period', 'all')
            max_results = int(request.POST.get('max_results', 100))

            # Validate parameters
            if min_support <= 0 or min_support > 1:
                min_support = 0.001
            if min_confidence <= 0 or min_confidence > 1:
                min_confidence = 0.6
            if transaction_period not in ['all', '1_month', '3_months', '6_months', '12_months']:
                transaction_period = 'all'
            if max_results not in [50, 100, 200, 500]:
                max_results = 100

            rules = _generate_commodity_association_rules(min_support, min_confidence, transaction_period, max_results)

            return JsonResponse({
                'success': True,
                'rules': rules,
                'count': len(rules),
                'parameters': {
                    'min_support': min_support,
                    'min_confidence': min_confidence,
                    'transaction_period': transaction_period,
                    'max_results': max_results
                }
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method allowed'}, status=405)


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
    """API endpoint for predictive market basket analysis - returns department predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        model_type = request.POST.get('model_type', 'neural_network')

        # Get time horizon parameter
        time_horizon_param = request.POST.get('time_horizon') or request.GET.get('time_horizon')
        try:
            time_horizon = int(time_horizon_param) if time_horizon_param else None
        except (TypeError, ValueError):
            time_horizon = None

        valid_horizons = {1, 3, 6, 12}
        selected_horizon = time_horizon if (time_horizon in valid_horizons) else 3

        # Get department predictions specifically
        department_predictions = ml_analyzer.get_department_predictions(model_type, selected_horizon)

        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'time_horizon_months': selected_horizon,
            'department_predictions': department_predictions
        })

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
    model_type = request.POST.get('model_type', 'neural_network')
    time_horizon_param = request.POST.get('time_horizon') or request.GET.get('time_horizon')
    try:
        time_horizon = int(time_horizon_param) if time_horizon_param else None
    except (TypeError, ValueError):
        time_horizon = None

    valid_horizons = {1, 3, 6, 12}
    selected_horizon = time_horizon if (time_horizon in valid_horizons) else 3

    try:
        predictions = ml_analyzer.get_department_predictions(model_type, selected_horizon)
        response_horizon = predictions[0].get('time_horizon_months', selected_horizon) if predictions else selected_horizon
        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'time_horizon_months': response_horizon,
            'predictions': predictions
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Prediction error: {str(e)}',
            'predictions': [],
            'time_horizon_months': selected_horizon
        })

@csrf_exempt
def get_recommendations(request):
    """Get AI-powered product recommendations"""
    model_type = request.POST.get('model_type', 'neural_network')
    top_n = int(request.POST.get('top_n', 10))
    time_horizon_param = request.POST.get('time_horizon') or request.GET.get('time_horizon')
    try:
        time_horizon = int(time_horizon_param) if time_horizon_param else None
    except (TypeError, ValueError):
        time_horizon = None

    valid_horizons = {1, 3, 6, 12}
    selected_horizon = time_horizon if (time_horizon in valid_horizons) else 3

    try:
        customer_id = request.POST.get('customer_id')

        recommendations = ml_analyzer.predict_customer_preferences(
            model_type, customer_id, top_n, selected_horizon
        )

        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'customer_id': customer_id,
            'time_horizon_months': selected_horizon,
            'recommendations': recommendations
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Recommendation error: {str(e)}',
            'time_horizon_months': selected_horizon,
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

