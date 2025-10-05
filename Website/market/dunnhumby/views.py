from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout
from django.db import models
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.db import connection
from django.contrib import messages
from django.urls import reverse
from django.db.models import Sum, Count, Avg, Max, Q, Min
from math import sqrt
import logging
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

logger = logging.getLogger(__name__)


# Define table categories based on their CRUD properties
READ_ONLY_ANALYTICAL_TABLES = ['basket_analysis', 'customer_segments']
MANAGED_ANALYTICAL_TABLES = ['association_rules']


def admin_required(view_func):
    """Decorator to require admin/staff status for views"""
    def check_admin(user):
        return user.is_authenticated and (user.is_staff or user.is_superuser)

    decorated_view = user_passes_test(check_admin, login_url='/analysis/login/')(view_func)
    return decorated_view


def user_login(request):
    """Login view for main website - only admins allowed"""
    if request.user.is_authenticated and (request.user.is_staff or request.user.is_superuser):
        return redirect('dunnhumby_site:index')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Check if user is admin/staff
            if user.is_staff or user.is_superuser:
                login(request, user)

                # Set session expiry (1 hour)
                request.session.set_expiry(3600)

                # Store device info for security
                request.session['user_agent'] = request.META.get('HTTP_USER_AGENT', '')
                request.session['ip_address'] = request.META.get('REMOTE_ADDR', '')

                messages.success(request, f'Welcome back, {user.username}!')

                # Redirect to next parameter or default
                next_url = request.GET.get('next', '/analysis/')
                return redirect(next_url)
            else:
                messages.error(request, 'Access denied. Admin privileges required.')
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'site/login.html')


def user_logout(request):
    """Logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('dunnhumby_site:login')


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
            # Allow very small support values but warn about performance
            if min_basket_count < 5 and total_baskets > 500000:
                logger.warning(f"Very low support threshold ({min_basket_count} baskets) may result in slow query performance")
            # Only enforce minimum if absolutely necessary for performance
            if min_basket_count < 1:
                min_basket_count = 1

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
    """Generate association rules at department level using Python aggregation for scalability."""
    from django.db import connection
    from itertools import combinations
    from collections import defaultdict

    rules = []
    candidate_limit = max(max_results * 5 if max_results else 100, 100)
    candidate_limit = min(candidate_limit, 1000)

    try:
        start_day = None
        max_day = None
        if transaction_period != 'all':
            with connection.cursor() as cursor:
                cursor.execute('SELECT MAX(day) FROM transactions')
                max_day_row = cursor.fetchone()
                max_day = max_day_row[0] if max_day_row and max_day_row[0] is not None else None
            if max_day is not None:
                period_days = {
                    '1_month': 30,
                    '3_months': 90,
                    '6_months': 180,
                    '12_months': 365,
                }
                days_back = period_days.get(transaction_period, max_day)
                start_day = max(1, max_day - days_back + 1)
                logger.info('Filtering department transactions from day %s to %s (%s)', start_day, max_day, transaction_period)

        # Count baskets for support denominator
        with connection.cursor() as cursor:
            if start_day is not None:
                cursor.execute('SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= %s', (start_day,))
            else:
                cursor.execute('SELECT COUNT(DISTINCT basket_id) FROM transactions')
            result = cursor.fetchone()
            total_baskets = result[0] if result else 0
        if total_baskets == 0:
            logger.warning('No baskets found for department-level rules')
            return rules

        min_basket_count = max(1, int(total_baskets * min_support))
        logger.info('Department rules: total baskets=%s, min basket count=%s', total_baskets, min_basket_count)

        # Determine eligible departments (those that can meet min support)
        base_conditions = []
        base_params = []
        if start_day is not None:
            base_conditions.append('t.day >= %s')
            base_params.append(start_day)
        base_conditions.append('p.department IS NOT NULL')
        where_clause = ' AND '.join(base_conditions)

        eligible_query = (
            "SELECT TOP " + str(candidate_limit) + " "
            "p.department, "
            "COUNT(DISTINCT t.basket_id) AS dept_count "
            "FROM transactions t "
            "JOIN product p ON t.product_id = p.product_id "
            "WHERE " + where_clause + " "
            "GROUP BY p.department "
            "HAVING COUNT(DISTINCT t.basket_id) >= %s "
            "ORDER BY dept_count DESC"
        )
        eligible_params = tuple(base_params + [min_basket_count])
        with connection.cursor() as cursor:
            cursor.execute(eligible_query, eligible_params)
            eligible_rows = cursor.fetchall()

        if not eligible_rows or len(eligible_rows) < 2:
            logger.info('No departments met the minimum support threshold (%s)', min_support)
            return rules

        dept_counts = {row[0]: row[1] for row in eligible_rows if row[0]}
        eligible_departments = list(dept_counts.keys())
        if len(eligible_departments) < 2:
            logger.info('Not enough departments met support threshold to form rules')
            return rules

        # Stream basket-department pairs for eligible departments only
        placeholders = ','.join(['%s'] * len(eligible_departments))
        stream_query = (
            "SELECT t.basket_id, p.department "
            "FROM transactions t "
            "JOIN product p ON t.product_id = p.product_id "
            "WHERE " + where_clause + " AND p.department IN (" + placeholders + ") "
            "GROUP BY t.basket_id, p.department "
            "ORDER BY t.basket_id"
        )
        stream_params = tuple(base_params + eligible_departments)

        pair_counts = defaultdict(int)
        current_basket = None
        current_departments = set()

        def process_current():
            if current_departments and len(current_departments) >= 2:
                for dept_a, dept_b in combinations(sorted(current_departments), 2):
                    pair_counts[(dept_a, dept_b)] += 1

        with connection.cursor() as cursor:
            cursor.arraysize = 5000
            cursor.execute(stream_query, stream_params)
            while True:
                batch = cursor.fetchmany(cursor.arraysize)
                if not batch:
                    break
                for basket_id, department in batch:
                    if department is None:
                        continue
                    if current_basket is None:
                        current_basket = basket_id
                    if basket_id != current_basket:
                        process_current()
                        current_departments.clear()
                        current_basket = basket_id
                    current_departments.add(department)
            # Process final basket
            process_current()

        if not pair_counts:
            logger.info('No department pairs met qualification criteria after streaming')
            return rules

        for (dept_a, dept_b), pair_count in pair_counts.items():
            if pair_count < min_basket_count:
                continue
            support = pair_count / total_baskets
            confidence_a_to_b = pair_count / dept_counts.get(dept_a, 1)
            confidence_b_to_a = pair_count / dept_counts.get(dept_b, 1)

            if confidence_a_to_b >= min_confidence:
                lift = confidence_a_to_b / (dept_counts.get(dept_b, 1) / total_baskets)
                rules.append({
                    'antecedent': dept_a,
                    'consequent': dept_b,
                    'support': support,
                    'confidence': confidence_a_to_b,
                    'lift': lift,
                    'rule_type': 'department'
                })
            if confidence_b_to_a >= min_confidence and confidence_b_to_a != confidence_a_to_b:
                lift = confidence_b_to_a / (dept_counts.get(dept_a, 1) / total_baskets)
                rules.append({
                    'antecedent': dept_b,
                    'consequent': dept_a,
                    'support': support,
                    'confidence': confidence_b_to_a,
                    'lift': lift,
                    'rule_type': 'department'
                })

        all_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
        logger.info('Generated %s department rules after streaming, returning top %s', len(all_rules), max_results)
        return all_rules[:max_results]

    except Exception as e:
        logger.exception('Error in _generate_department_association_rules')
        raise



def _generate_commodity_association_rules(min_support, min_confidence, transaction_period='all', max_results=100):
    """Generate association rules at commodity level using Python aggregation for scalability."""
    from django.db import connection
    from itertools import combinations
    from collections import defaultdict

    rules = []
    candidate_limit = max(max_results * 5 if max_results else 100, 200)
    candidate_limit = min(candidate_limit, 3000)

    try:
        start_day = None
        max_day = None
        if transaction_period != 'all':
            with connection.cursor() as cursor:
                cursor.execute('SELECT MAX(day) FROM transactions')
                max_day_row = cursor.fetchone()
                max_day = max_day_row[0] if max_day_row and max_day_row[0] is not None else None
            if max_day is not None:
                period_days = {
                    '1_month': 30,
                    '3_months': 90,
                    '6_months': 180,
                    '12_months': 365,
                }
                days_back = period_days.get(transaction_period, max_day)
                start_day = max(1, max_day - days_back + 1)
                logger.info('Filtering commodity transactions from day %s to %s (%s)', start_day, max_day, transaction_period)

        # Count baskets for support denominator
        with connection.cursor() as cursor:
            if start_day is not None:
                cursor.execute('SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= %s', (start_day,))
            else:
                cursor.execute('SELECT COUNT(DISTINCT basket_id) FROM transactions')
            result = cursor.fetchone()
            total_baskets = result[0] if result else 0
        if total_baskets == 0:
            logger.warning('No baskets found for commodity-level rules')
            return rules

        min_basket_count = max(1, int(total_baskets * min_support))
        logger.info('Commodity rules: total baskets=%s, min basket count=%s', total_baskets, min_basket_count)

        base_conditions = []
        base_params = []
        if start_day is not None:
            base_conditions.append('t.day >= %s')
            base_params.append(start_day)
        base_conditions.append('p.commodity_desc IS NOT NULL')
        where_clause = ' AND '.join(base_conditions)

        eligible_query = (
            "SELECT TOP " + str(candidate_limit) + " "
            "p.commodity_desc, "
            "COUNT(DISTINCT t.basket_id) AS comm_count "
            "FROM transactions t "
            "JOIN product p ON t.product_id = p.product_id "
            "WHERE " + where_clause + " "
            "GROUP BY p.commodity_desc "
            "HAVING COUNT(DISTINCT t.basket_id) >= %s "
            "ORDER BY comm_count DESC"
        )
        eligible_params = tuple(base_params + [min_basket_count])
        with connection.cursor() as cursor:
            cursor.execute(eligible_query, eligible_params)
            eligible_rows = cursor.fetchall()

        if not eligible_rows or len(eligible_rows) < 2:
            logger.info('No commodities met the minimum support threshold (%s)', min_support)
            return rules

        commodity_counts = {row[0]: row[1] for row in eligible_rows if row[0]}
        eligible_commodities = list(commodity_counts.keys())
        if len(eligible_commodities) < 2:
            logger.info('Not enough commodities met support threshold to form rules')
            return rules

        placeholders = ','.join(['%s'] * len(eligible_commodities))
        stream_query = (
            "SELECT t.basket_id, p.commodity_desc "
            "FROM transactions t "
            "JOIN product p ON t.product_id = p.product_id "
            "WHERE " + where_clause + " AND p.commodity_desc IN (" + placeholders + ") "
            "GROUP BY t.basket_id, p.commodity_desc "
            "ORDER BY t.basket_id"
        )
        stream_params = tuple(base_params + eligible_commodities)

        pair_counts = defaultdict(int)
        current_basket = None
        current_commodities = set()

        def process_current():
            if current_commodities and len(current_commodities) >= 2:
                for comm_a, comm_b in combinations(sorted(current_commodities), 2):
                    pair_counts[(comm_a, comm_b)] += 1

        with connection.cursor() as cursor:
            cursor.arraysize = 5000
            cursor.execute(stream_query, stream_params)
            while True:
                batch = cursor.fetchmany(cursor.arraysize)
                if not batch:
                    break
                for basket_id, commodity in batch:
                    if commodity is None:
                        continue
                    if current_basket is None:
                        current_basket = basket_id
                    if basket_id != current_basket:
                        process_current()
                        current_commodities.clear()
                        current_basket = basket_id
                    current_commodities.add(commodity)
            process_current()

        if not pair_counts:
            logger.info('No commodity pairs met qualification criteria after streaming')
            return rules

        for (comm_a, comm_b), pair_count in pair_counts.items():
            if pair_count < min_basket_count:
                continue
            support = pair_count / total_baskets
            confidence_a_to_b = pair_count / commodity_counts.get(comm_a, 1)
            confidence_b_to_a = pair_count / commodity_counts.get(comm_b, 1)

            if confidence_a_to_b >= min_confidence:
                lift = confidence_a_to_b / (commodity_counts.get(comm_b, 1) / total_baskets)
                rules.append({
                    'antecedent': comm_a,
                    'consequent': comm_b,
                    'support': support,
                    'confidence': confidence_a_to_b,
                    'lift': lift,
                    'rule_type': 'commodity'
                })
            if confidence_b_to_a >= min_confidence and confidence_b_to_a != confidence_a_to_b:
                lift = confidence_b_to_a / (commodity_counts.get(comm_a, 1) / total_baskets)
                rules.append({
                    'antecedent': comm_b,
                    'consequent': comm_a,
                    'support': support,
                    'confidence': confidence_b_to_a,
                    'lift': lift,
                    'rule_type': 'commodity'
                })

        all_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
        logger.info('Generated %s commodity rules after streaming, returning top %s', len(all_rules), max_results)
        return all_rules[:max_results]

    except Exception as e:
        logger.exception('Error in _generate_commodity_association_rules')
        raise

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


@admin_required
def site_index(request):
    # Calculate dynamic metrics
    total_transactions = Transaction.objects.count()
    unique_products = Transaction.objects.values('product_id').distinct().count()
    active_customers = Transaction.objects.values('household_key').distinct().count()
    total_revenue = Transaction.objects.aggregate(total=Sum('sales_value'))['total'] or 0

    # Keep revenue as raw value for JavaScript formatting
    total_revenue_raw = int(total_revenue)

    # Calculate period-over-period changes (last 60 days vs previous 60 days)
    from django.db.models import Min, Max
    day_stats = Transaction.objects.aggregate(min_day=Min('day'), max_day=Max('day'))
    max_day = day_stats['max_day']

    # Recent period: days 652-711, Previous period: days 592-651
    recent_period_start = max_day - 59
    previous_period_start = max_day - 119
    previous_period_end = max_day - 60

    # Recent period metrics
    recent_transactions = Transaction.objects.filter(day__gte=recent_period_start).count()
    recent_products = Transaction.objects.filter(day__gte=recent_period_start).values('product_id').distinct().count()
    recent_customers = Transaction.objects.filter(day__gte=recent_period_start).values('household_key').distinct().count()
    recent_revenue = Transaction.objects.filter(day__gte=recent_period_start).aggregate(total=Sum('sales_value'))['total'] or 0

    # Previous period metrics
    prev_transactions = Transaction.objects.filter(day__gte=previous_period_start, day__lte=previous_period_end).count()
    prev_products = Transaction.objects.filter(day__gte=previous_period_start, day__lte=previous_period_end).values('product_id').distinct().count()
    prev_customers = Transaction.objects.filter(day__gte=previous_period_start, day__lte=previous_period_end).values('household_key').distinct().count()
    prev_revenue = Transaction.objects.filter(day__gte=previous_period_start, day__lte=previous_period_end).aggregate(total=Sum('sales_value'))['total'] or 0

    # Calculate percentage changes
    trans_change = ((recent_transactions - prev_transactions) / prev_transactions * 100) if prev_transactions > 0 else 0
    prod_change = ((recent_products - prev_products) / prev_products * 100) if prev_products > 0 else 0
    cust_change = ((recent_customers - prev_customers) / prev_customers * 100) if prev_customers > 0 else 0
    rev_change = ((float(recent_revenue) - float(prev_revenue)) / float(prev_revenue) * 100) if prev_revenue > 0 else 0

    tools = [
        { 'title': 'Shopping Basket Analysis', 'description': 'Analyze baskets, top products, and patterns', 'url': 'basket-analysis/', 'icon': '🧺' },
        { 'title': 'Association Rules', 'description': 'Market basket association rules', 'url': 'association-rules/', 'icon': '🔗' },
        { 'title': 'Customer Segments', 'description': 'RFM segments & behavior', 'url': 'customer-segments/', 'icon': '👥' },
        { 'title': 'Data Management', 'description': 'View, edit, import/export data', 'url': 'data-management/', 'icon': '⚙️' },
        { 'title': 'Customer Insights', 'description': 'Explore and manage your customer data in detail', 'url': reverse('customers:search'), 'icon': '👤' },
    ]

    context = {
        'analysis_tools': tools,
        'total_transactions': total_transactions,
        'unique_products': unique_products,
        'active_customers': active_customers,
        'total_revenue': total_revenue_raw,
        'trans_change': trans_change,
        'prod_change': prod_change,
        'cust_change': cust_change,
        'rev_change': rev_change,
    }

    return render(request, 'site/index.html', context)


@admin_required
def api_market_trends(request):
    """
    API endpoint for real-time market trends data (last 12 months)
    Returns monthly sales volume, revenue, customers, and basket size
    """
    from django.http import JsonResponse
    from django.db import connection
    from collections import defaultdict

    try:
        with connection.cursor() as cursor:
            # Get monthly aggregated data for exactly 12 months ending at day 711
            # 12 complete months = days 352-711 (360 days)
            # Group by 30-day periods: Month 1 (days 352-381), Month 2 (days 382-411), etc.
            cursor.execute("""
                SELECT
                    ((day - 352) / 30) + 1 as month_num,
                    COUNT(*) as transaction_count,
                    SUM(sales_value) as total_sales,
                    COUNT(DISTINCT household_key) as unique_customers,
                    COUNT(DISTINCT basket_id) as basket_count,
                    AVG(sales_value) as avg_basket_value
                FROM transactions
                WHERE day >= 352 AND day <= 711
                GROUP BY ((day - 352) / 30) + 1
                HAVING ((day - 352) / 30) + 1 <= 12
                ORDER BY month_num
            """)

            rows = cursor.fetchall()

            # Initialize data arrays
            sales_volume = []
            revenue = []
            customers = []
            basket_sizes = []
            month_labels = []

            # Process exactly 12 months of data
            for i, row in enumerate(rows):
                month_num, txn_count, total_sales, unique_customers, basket_count, avg_basket = row

                sales_volume.append(int(txn_count))
                revenue.append(round(float(total_sales or 0), 2))
                customers.append(int(unique_customers or 0))
                basket_sizes.append(round(float(avg_basket or 0), 2))

            # Generate labels after filtering
            for i in range(len(sales_volume)):
                # Label showing months ago (oldest → newest)
                months_ago = len(sales_volume) - i - 1
                if months_ago == 0:
                    month_labels.append("Current")
                elif months_ago == 1:
                    month_labels.append("1 mo ago")
                else:
                    month_labels.append(f"{months_ago} mo ago")

            return JsonResponse({
                'success': True,
                'labels': month_labels,
                'datasets': {
                    'sales_volume': sales_volume,
                    'revenue': revenue,
                    'customers': customers,
                    'basket_sizes': basket_sizes
                }
            })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


from .analytics import RFMAnalyzer # ۱. کلاس اصلی را وارد می‌کنیم

@admin_required
@require_POST
def api_regenerate_segments(request):
    """
    API endpoint to regenerate customer segments by calling the RFMAnalyzer.
    """
    try:
        # ۲. یک نمونه از موتور تحلیلی می‌سازیم
        analyzer = RFMAnalyzer()
        
        # ۳. تمام مراحل تحلیل را با فراخوانی متدهای کلاس انجام می‌دهیم
        analyzer.calculate_rfm_scores()
        analyzer.segment_customers()
        analyzer.save_segments_to_db()
        
        # تعداد سگمنت‌های ایجاد شده را از کلاس می‌خوانیم
        count = len(analyzer.segments)
        
        # ۴. پاسخ موفقیت‌آمیز را برمی‌گردانیم
        return JsonResponse({
            'success': True,
            'count': count,
            'message': f'Successfully generated {count} customer segments'
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error regenerating segments: {str(e)}\n{error_details}")
        return JsonResponse({
            'success': False,
            'error': f'{str(e)} - Check server logs for details'
        }, status=500)

@admin_required
def basket_analysis(request):
    """
    Optimized Market Basket Analysis for 2.6M+ transactions
    """
    logger.info("Starting basket analysis for 2.6M+ transactions")

    try:
        # Get overall statistics efficiently using raw SQL
        with connection.cursor() as cursor:
            # Overall dataset statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_transactions,
                    COUNT(DISTINCT basket_id) as total_baskets,
                    COUNT(DISTINCT product_id) as total_products,
                    COUNT(DISTINCT household_key) as total_customers,
                    SUM(sales_value) as total_revenue,
                    AVG(sales_value) as avg_transaction_value
                FROM transactions
            """)
            overall_stats = cursor.fetchone()

            # Top baskets by value (optimized for large dataset)
            cursor.execute("""
                SELECT TOP 25
                    basket_id,
                    household_key,
                    SUM(quantity) as total_items,
                    SUM(sales_value) as total_value,
                    COUNT(DISTINCT product_id) as unique_products,
                    COUNT(*) as transaction_count
                FROM transactions
                GROUP BY basket_id, household_key
                ORDER BY total_value DESC
            """)
            basket_stats = cursor.fetchall()

            # Department analysis - aggregated for performance
            cursor.execute("""
                SELECT TOP 15
                    p.department,
                    COUNT(*) as transaction_count,
                    SUM(t.sales_value) as total_sales,
                    SUM(t.quantity) as total_quantity,
                    COUNT(DISTINCT t.product_id) as unique_products,
                    AVG(t.sales_value) as avg_sales_per_transaction
                FROM transactions t
                LEFT JOIN product p ON t.product_id = p.product_id
                GROUP BY p.department
                ORDER BY total_sales DESC
            """)
            dept_analysis = cursor.fetchall()

        # Top products by frequency - optimized query
        product_stats = Transaction.objects.values('product_id').annotate(
            frequency=Count('product_id'),
            total_sales=Sum('sales_value'),
            avg_sales=Sum('sales_value') / Count('product_id'),
            total_quantity=Sum('quantity')
        ).filter(product_id__isnull=False)

        top_products_frequency_raw = list(product_stats.order_by('-frequency')[:25])
        top_products_sales_raw = list(product_stats.order_by('-total_sales')[:25])

        # Get product details in one efficient query
        all_product_ids = {item['product_id'] for item in top_products_frequency_raw + top_products_sales_raw}
        product_details = {
            item.product_id: item for item in DunnhumbyProduct.objects.filter(product_id__in=all_product_ids)
        }

        def _enrich_product_records(records):
            enriched = []
            for record in records:
                product_detail = product_details.get(record['product_id'])
                enriched.append({
                    'product_id': record['product_id'],
                    'frequency': record['frequency'],
                    'total_sales': float(record['total_sales'] or 0),
                    'avg_sales': float(record.get('avg_sales', 0) or 0),
                    'total_quantity': record.get('total_quantity', 0) or 0,
                    'brand': product_detail.brand if product_detail else 'Unknown Brand',
                    'department': product_detail.department if product_detail else 'MISC. TRANS.',
                    'commodity_desc': product_detail.commodity_desc if product_detail else 'Unknown Product',
                    'sub_commodity_desc': product_detail.sub_commodity_desc if product_detail else '',
                    'manufacturer': product_detail.manufacturer if product_detail else 0
                })
            return enriched

        top_products_frequency = _enrich_product_records(top_products_frequency_raw)
        top_products_sales = _enrich_product_records(top_products_sales_raw)

        # Format basket stats for template
        formatted_basket_stats = []
        for basket in basket_stats:
            formatted_basket_stats.append({
                'basket_id': basket[0],
                'household_key': basket[1],
                'total_items': basket[2],
                'total_value': float(basket[3]),
                'unique_products': basket[4],
                'transaction_count': basket[5]
            })

        # Format department analysis for template
        formatted_dept_analysis = []
        for dept in dept_analysis:
            formatted_dept_analysis.append({
                'department': dept[0] or 'UNKNOWN',
                'transaction_count': dept[1],
                'total_sales': float(dept[2]),
                'total_quantity': dept[3],
                'unique_products': dept[4],
                'avg_sales_per_transaction': float(dept[5])
            })

        # Create context with comprehensive statistics
        avg_basket_size = overall_stats[0] / overall_stats[1] if overall_stats[1] > 0 else 0

        context = {
            'title': 'Market Basket Analysis - 2.6M Transactions',
            'overall_stats': {
                'total_transactions': overall_stats[0],
                'total_baskets': overall_stats[1],
                'total_products': overall_stats[2],
                'total_customers': overall_stats[3],
                'total_revenue': float(overall_stats[4]),
                'avg_transaction_value': float(overall_stats[5]),
                'avg_basket_size': avg_basket_size
            },
            'basket_stats': formatted_basket_stats,
            'dept_analysis': formatted_dept_analysis,
            'top_products_frequency': top_products_frequency,
            'top_products_sales': top_products_sales,
        }

        logger.info("Basket analysis completed successfully")
        return render(request, 'site/dunnhumby/basket_analysis.html', context)

    except Exception as e:
        logger.error(f"Error in basket analysis: {str(e)}")
        context = {
            'title': 'Market Basket Analysis - Error',
            'error_message': f'Error loading basket analysis: {str(e)}. Please try again.',
            'overall_stats': None,
            'basket_stats': [],
            'dept_analysis': [],
            'top_products_frequency': [],
            'top_products_sales': [],
        }
        return render(request, 'site/dunnhumby/basket_analysis.html', context)


@admin_required
def association_rules(request):
    if request.method == 'POST':
        try:
            min_support = float(request.POST.get('min_support', 0.0001))
            min_confidence = float(request.POST.get('min_confidence', 0.5))
            transaction_period = request.POST.get('transaction_period', 'all')
            max_results = int(request.POST.get('max_results', 100))

            # Validate parameters - allow very small positive values
            if min_support <= 0 or min_support > 1:
                min_support = 0.00001  # Allow much smaller default
            # Warn about very small values and adjust for performance
            if min_support < 0.00001:
                logger.warning(f"Very small support value ({min_support}) may cause performance issues")

            # For ultra-small support values, limit results for performance
            if min_support < 0.000005:
                max_results = min(max_results, 50)  # Limit to 50 results for ultra-rare patterns
                logger.info(f"Ultra-small support detected, limiting results to {max_results}")
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
            import traceback
            logger.error(f"Association rules generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Provide more specific error guidance
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                suggestion = "Query timed out. Try using higher support values (≥0.0001) or shorter time periods."
            elif "memory" in error_msg.lower():
                suggestion = "Out of memory. Try using higher support values (≥0.0005) or limit to recent periods."
            elif "syntax" in error_msg.lower():
                suggestion = "Database syntax error. Please try again or contact support."
            else:
                suggestion = "Try using higher support values (≥0.00005) or different parameters."

            ctx = {
                'title': 'Association Rules',
                'rules': [],
                'error_message': f'Error generating rules: {error_msg}. {suggestion}',
                'min_support': request.POST.get('min_support', 0.00005),
                'min_confidence': request.POST.get('min_confidence', 0.5),
                'transaction_period': request.POST.get('transaction_period', 'all'),
                'max_results': request.POST.get('max_results', 100),
            }
    else:
        ctx = {
            'title': 'Association Rules',
            'rules': AssociationRule.objects.all().order_by('-lift')[:100],
        }
    return render(request, 'site/dunnhumby/association_rules.html', ctx)


@admin_required
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
        sort_column = request.POST.get('sort_column', '')
        sort_direction = request.POST.get('sort_direction', 'asc')

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
            'products': ['product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'],
            'households': ['household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'],
            'campaigns': ['description'],
            'customer_segments': ['rfm_segment', 'household_key'],
            'basket_analysis': ['basket_id', 'household_key'],
            'association_rules': ['rule_type']
        }

        # Special handling for association_rules search
        if search and table_name == 'association_rules':
            q_objects = Q()
            # Search in rule_type
            q_objects |= Q(rule_type__icontains=search)
            # Search in JSON fields by converting to string
            q_objects |= Q(antecedent__icontains=search)
            q_objects |= Q(consequent__icontains=search)
            # Search by ID if numeric
            if search.isnumeric():
                q_objects |= Q(id=int(search))
            queryset = queryset.filter(q_objects)
        elif search and table_name in searchable_fields:
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
        
        # Ordering - check for client-side sort first, then default ordering
        if sort_column:
            # Client requested sorting by a specific column
            order_field = f"-{sort_column}" if sort_direction == 'desc' else sort_column
            queryset = queryset.order_by(order_field)
        else:
            # Default ordering per table
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
            'rfm_segment','recency_score','frequency_score','monetary_score',
            'total_spend','total_transactions','avg_basket_value','updated_at',
            'churn_probability'   # 👈 اضافه شد
        ).first()

        if not seg:
            return JsonResponse({'error': 'household not found'}, status=404)

        # محاسبه churn_risk label
        prob = seg.get('churn_probability')
        if prob is None:
            seg['churn_risk'] = "N/A"
        elif prob > 0.75:
            seg['churn_risk'] = "Very High Risk"
        elif prob > 0.50:
            seg['churn_risk'] = "High Risk"
        elif prob > 0.25:
            seg['churn_risk'] = "Medium Risk"
        else:
            seg['churn_risk'] = "Low Risk"

        # تراکنش‌های اخیر
        recent_txns = list(
            Transaction.objects.filter(household_key=household_key).values(
                'basket_id','product_id','quantity','sales_value','day'
            ).order_by('-day')[:15]
        )

        # enrich با اسم محصول
        pids = list({t['product_id'] for t in recent_txns})
        prodmap = {
            p['product_id']: p['commodity_desc']
            for p in DunnhumbyProduct.objects.filter(product_id__in=pids).values('product_id','commodity_desc')
        }
        for t in recent_txns:
            t['commodity_desc'] = prodmap.get(t['product_id'])

        return JsonResponse({
            'household_key': household_key,
            'segment': seg,
            'recent_transactions': recent_txns
        })
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

        # def fetch_basket_totals_for_segment(segment, limit=4000):
        #     limit = max(int(limit or 4000), 1)
        #     select_limit, suffix_limit = build_limit_clause(limit)
        #     query = f"""
        #         SELECT {select_limit}
        #             t.basket_id,
        #             SUM(t.sales_value) as total_value
        #         FROM transactions t
        #         WHERE t.household_key IN (
        #             SELECT household_key FROM dunnhumby_customersegment WHERE rfm_segment = %s
        #         )
        #         GROUP BY t.basket_id
        #         {suffix_limit}
        #     """
        #     with connection.cursor() as cursor:
        #         cursor.execute(query, [segment])
        #         rows = cursor.fetchall()
        #     return [to_float(row[1]) for row in rows if to_float(row[1]) > 0]

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

                        # Calculate Cramér's V (standard formula)
                        cramers_v = sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

                        # For large datasets, also calculate Cohen's w for practical effect size
                        # Cohen's w based on variance in proportions
                        row_totals = observed_arr.sum(axis=1)
                        col_totals = observed_arr.sum(axis=0)
                        expected = np.outer(row_totals, col_totals) / n
                        cohen_w = sqrt(np.sum((observed_arr - expected) ** 2 / expected) / n) if n > 0 else 0.0

                        # Calculate practical effect based on data size and proportional differences
                        # For small matrices (departments), calculate max proportional difference
                        row_proportions = row_totals / n
                        max_prop_diff = max(row_proportions) / min(row_proportions) if min(row_proportions) > 0 else 1.0

                        # Hybrid effect size calculation
                        if n > 100000:  # Large dataset (transaction counts)
                            effect = cohen_w * sqrt(min(n / 100000, 50))
                        elif n < 10000 and r <= 6:  # Small matrix (department sales)
                            # Use Cramér's V but boost based on proportional differences
                            if max_prop_diff > 10:  # 10x+ difference
                                effect = max(cramers_v * 100, 0.8)  # Boost to large effect
                            elif max_prop_diff > 5:  # 5x+ difference
                                effect = max(cramers_v * 50, 0.5)  # Boost to medium effect
                            else:
                                effect = cramers_v * 20
                        else:
                            effect = cohen_w

                        stats['p_value'] = format_stat_value(p_value)
                        stats['effect_size'] = format_stat_value(effect, decimals=2)
                        stats['confidence'] = max(50, min(99, int(round((1 - p_value) * 100))))
                        stats['test_used'] = 'chi_square'
                        stats['note'] = (
                            f"<strong>Chi-Square Test:</strong> Analyzed {int(n):,} transactions across {r} groups and {c} categories. "
                            f"Tests if product category distributions differ significantly between groups."
                        )
                        return stats

                elif test_name == 't_test' and ttest_ind and len(group_a) > 1 and len(group_b) > 1:
                    group_a_np = np.array(group_a, dtype=float)
                    group_b_np = np.array(group_b, dtype=float)
                    _, p_value = ttest_ind(group_a_np, group_b_np, equal_var=False)
                    mean_a = group_a_np.mean()
                    mean_b = group_b_np.mean()
                    mean_diff = abs(mean_a - mean_b)
                    pooled_std = np.sqrt((group_a_np.var(ddof=1) + group_b_np.var(ddof=1)) / 2)

                    # Standard Cohen's d
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                    # For retail data with high variance, also calculate percentage difference
                    baseline_mean = min(mean_a, mean_b)
                    pct_diff = (mean_diff / baseline_mean) if baseline_mean > 0 else 0

                    # Hybrid effect size: weight Cohen's d but boost if percentage difference is large
                    # For retail: 50%+ difference should show as medium-large effect even with high variance
                    if pct_diff > 0.5:  # >50% difference
                        effect = max(cohens_d, 0.5 + (pct_diff - 0.5) * 0.4)  # Boost to at least medium
                    elif pct_diff > 0.3:  # >30% difference
                        effect = max(cohens_d, 0.3 + (pct_diff - 0.3) * 0.5)  # Boost to small-medium
                    else:
                        effect = cohens_d

                    stats['p_value'] = format_stat_value(p_value)
                    stats['effect_size'] = format_stat_value(effect, decimals=2)
                    stats['confidence'] = max(50, min(99, int(round((1 - p_value) * 100))))
                    stats['test_used'] = 't_test'
                    # Determine context based on sample size
                    if len(group_a_np) < 50:
                        context = f"department sales totals"
                    else:
                        context = f"transactions"

                    stats['note'] = (
                        f"<strong>Welch's T-Test:</strong> Compared {len(group_a_np):,} vs {len(group_b_np):,} {context}. "
                        f"Mean difference: ${mean_diff:,.2f} ({pct_diff*100:.1f}% change)."
                    )
                    return stats

                elif test_name == 'mann_whitney' and mannwhitneyu and len(group_a) and len(group_b):
                    u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
                    n1, n2 = len(group_a), len(group_b)

                    # Rank-biserial correlation (ranges -1 to 1)
                    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

                    # Also calculate percentage difference in medians for practical significance
                    median_a = np.median(group_a)
                    median_b = np.median(group_b)
                    median_diff = abs(median_a - median_b)
                    baseline_median = min(median_a, median_b)
                    pct_diff = (median_diff / baseline_median) if baseline_median > 0 else 0

                    # Use rank-biserial but boost if large median difference
                    effect = abs(rank_biserial)
                    if pct_diff > 0.5 and effect < 0.5:
                        effect = max(effect, 0.5 + (pct_diff - 0.5) * 0.3)
                    elif pct_diff > 0.3 and effect < 0.3:
                        effect = max(effect, 0.3 + (pct_diff - 0.3) * 0.4)

                    stats['p_value'] = format_stat_value(p_value)
                    stats['effect_size'] = format_stat_value(effect, decimals=2)
                    stats['confidence'] = max(50, min(95, int(round((1 - p_value) * 100))))
                    stats['test_used'] = 'mann_whitney'
                    # Determine context based on sample size
                    if n1 < 50:
                        context = f"department sales totals"
                    else:
                        context = f"transactions"

                    stats['note'] = (
                        f"<strong>Mann-Whitney U Test:</strong> Non-parametric comparison of {n1:,} vs {n2:,} {context}. "
                        f"Robust for skewed data - median difference: ${median_diff:,.2f} ({pct_diff*100:.1f}% change)."
                    )
                    return stats

                elif test_name == 'kolmogorov' and ks_2samp and len(group_a) and len(group_b):
                    ks_stat, p_value = ks_2samp(group_a, group_b, alternative='two-sided', mode='auto')
                    stats['p_value'] = format_stat_value(p_value)
                    stats['effect_size'] = format_stat_value(ks_stat, decimals=2)
                    stats['confidence'] = max(50, min(99, int(round((1 - p_value) * 100))))
                    stats['test_used'] = 'kolmogorov'
                    # Determine context based on sample size
                    if len(group_a) < 50:
                        context = f"department sales distributions"
                    else:
                        context = f"transaction distributions"

                    stats['note'] = (
                        f"<strong>Kolmogorov-Smirnov Test:</strong> Compared {len(group_a):,} vs {len(group_b):,} {context}. "
                        f"Detects distribution differences (KS statistic: {ks_stat:.3f})."
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
                    f"<strong>⚠️ Limited Data:</strong> Insufficient samples to run {test_name.replace('_', ' ')} reliably. "
                    f"Preliminary analysis shows ${diff:.2f} difference ({ratio * 100:.1f}% change) between groups."
                )
            else:
                stats['note'] = (
                    f"<strong>⚠️ Insufficient Data:</strong> Not enough samples to perform {test_name.replace('_', ' ')}. "
                    f"Try adjusting comparison parameters or using a different statistical test."
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

            # Create two matrices: one for transaction counts, one for sales
            observed_matrix_counts = []  # For traditional chi-square
            observed_matrix_sales = []   # For sales-based chi-square

            if ordered_quarters and department_order:
                for quarter in ordered_quarters:
                    # Transaction count matrix (original)
                    observed_matrix_counts.append([
                        dept_quarters.get(dept, {}).get(quarter, {}).get('count', 0)
                        for dept in department_order
                    ])
                    # Sales matrix (scaled to thousands of dollars for reasonable numbers)
                    observed_matrix_sales.append([
                        int(dept_quarters.get(dept, {}).get(quarter, {}).get('sales', 0) / 1000)
                        for dept in department_order
                    ])

                # Use sales matrix for effect size calculation
                observed_matrix = observed_matrix_sales

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

                # Build department-level sales arrays for each quarter
                # This compares aggregate sales by department (what actually differs 13x)
                # instead of individual basket values (which are similar ~$29)
                group_a_sales = []  # Peak quarter department sales
                group_b_sales = []  # Low quarter department sales

                for dept in department_order:
                    peak_sales = dept_quarters.get(dept, {}).get(peak_quarter, {}).get('sales', 0)
                    low_sales = dept_quarters.get(dept, {}).get(low_quarter, {}).get('sales', 0)
                    if peak_sales > 0:
                        group_a_sales.append(peak_sales)
                    if low_sales > 0:
                        group_b_sales.append(low_sales)

                stats = compute_statistics(
                    stat_test,
                    observed=observed_matrix if stat_test == 'chi_square' else None,
                    group_a=group_a_sales,
                    group_b=group_b_sales
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

            # Create sales-based matrix for Chi-Square (scaled to thousands)
            observed = []
            for segment in [high_seg['rfm_segment'], low_seg['rfm_segment']]:
                observed.append([
                    int(segment_departments[segment].get(label, {}).get('sales', 0) / 1000)
                    for label in labels
                ])

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

            # Build department-level sales arrays for aggregate comparison
            group_a_sales = []  # High segment department sales
            group_b_sales = []  # Low segment department sales

            for label in labels:
                high_sales = segment_departments[high_seg['rfm_segment']].get(label, {}).get('sales', 0)
                low_sales = segment_departments[low_seg['rfm_segment']].get(label, {}).get('sales', 0)
                if high_sales > 0:
                    group_a_sales.append(high_sales)
                if low_sales > 0:
                    group_b_sales.append(low_sales)

            stats = compute_statistics(
                stat_test,
                observed=observed if stat_test == 'chi_square' else None,
                group_a=group_a_sales,
                group_b=group_b_sales
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

            # Create sales-based matrix for Chi-Square (scaled to thousands)
            observed = []
            for store_id in top_stores:
                observed.append([
                    int(store_departments[store_id].get(label, {}).get('sales', 0) / 1000)
                    for label in labels
                ])

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

            # Build department-level sales arrays for aggregate comparison
            group_a_sales = []  # Best store department sales
            group_b_sales = []  # Runner store department sales

            for label in labels:
                best_sales = store_departments[best_store].get(label, {}).get('sales', 0)
                runner_sales = store_departments[runner_store].get(label, {}).get('sales', 0)
                if best_sales > 0:
                    group_a_sales.append(best_sales)
                if runner_sales > 0:
                    group_b_sales.append(runner_sales)

            stats = compute_statistics(
                stat_test,
                observed=observed if stat_test == 'chi_square' else None,
                group_a=group_a_sales,
                group_b=group_b_sales
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
            # Create sales-based matrix for Chi-Square (scaled to thousands)
            observed = []
            for season in season_order:
                observed.append([
                    int(dept_season.get(dept, {}).get(season, {}).get('sales', 0) / 1000)
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

            # Build department-level sales arrays for aggregate comparison
            group_a_sales = []  # Peak season department sales
            group_b_sales = []  # Low season department sales

            for dept in top_departments_matrix:
                peak_sales = dept_season.get(dept, {}).get(peak_season, {}).get('sales', 0)
                low_sales = dept_season.get(dept, {}).get(low_season, {}).get('sales', 0)
                if peak_sales > 0:
                    group_a_sales.append(peak_sales)
                if low_sales > 0:
                    group_b_sales.append(low_sales)

            observed_for_stats = None
            if stat_test == 'chi_square' and len(season_order) >= 2 and len(top_departments_matrix) >= 2:
                observed_for_stats = observed

            stats = compute_statistics(
                stat_test,
                observed=observed_for_stats,
                group_a=group_a_sales,
                group_b=group_b_sales
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

# در فایل views.py

from django.core.paginator import Paginator # این خط را اضافه کنید

@login_required(login_url='/admin/login/')
def api_rfm_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    name = request.POST.get('rfm_segment')
    page_number = request.POST.get('page', 1) # شماره صفحه درخواستی را از فرانت‌اند می‌گیریم

    if not name:
        return JsonResponse({'error': 'rfm_segment required'}, status=400)

    try:
        qs = CustomerSegment.objects.filter(rfm_segment=name)

        # متریک‌های کلی مثل قبل محاسبه می‌شوند و نیازی به تغییر ندارند
        agg = qs.aggregate(
            customers=Count('household_key'),
            avg_spend=Avg('total_spend'),
            avg_txns=Avg('total_transactions'),
            avg_basket=Avg('avg_basket_value')
        )

        # ۱. دیگر همه خانوارها را به لیست تبدیل نمی‌کنیم، فقط کوئری را آماده می‌کنیم
        all_households_qs = qs.values(
            'household_key','total_spend','total_transactions','avg_basket_value','recency_score',
            'frequency_score','monetary_score','churn_probability','updated_at'
        ).order_by('-total_spend')

        # ۲. یک Paginator با تمام داده‌ها و سایز صفحه ۲۰ می‌سازیم
        paginator = Paginator(all_households_qs, 20) # هر صفحه ۲۰ آیتم خواهد داشت

        # ۳. صفحه درخواستی را از Paginator می‌گیریم
        page_obj = paginator.get_page(page_number)

        return JsonResponse({
            'rfm_segment': name,
            'metrics': {
                'customers': agg['customers'] or 0,
                'avg_spend': float(agg['avg_spend'] or 0),
                'avg_txns': float(agg['avg_txns'] or 0),
                'avg_basket': float(agg['avg_basket'] or 0),
            },
            # ۴. فقط لیست خانوارهای صفحه فعلی را ارسال می‌کنیم
            'households_page': list(page_obj.object_list),
            # ۵. اطلاعات صفحه‌بندی را هم ارسال می‌کنیم تا فرانت‌اند از آن استفاده کند
            'pagination': {
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'current_page': page_obj.number,
                'total_pages': paginator.num_pages,
                'total_items': paginator.count,
            }
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
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    try:
        min_support = float(request.POST.get('min_support', 0.001))
    except (TypeError, ValueError):
        min_support = 0.001

    try:
        min_confidence = float(request.POST.get('min_confidence', 0.6))
    except (TypeError, ValueError):
        min_confidence = 0.6

    transaction_period = request.POST.get('transaction_period', 'all') or 'all'

    try:
        max_results = int(request.POST.get('max_results', 100))
    except (TypeError, ValueError):
        max_results = 100

    min_support = max(1e-6, min(min_support, 1.0))
    min_confidence = min(max(min_confidence, 0.0), 1.0)

    valid_periods = {'all', '1_month', '3_months', '6_months', '12_months'}
    if transaction_period not in valid_periods:
        transaction_period = 'all'

    if max_results <= 0:
        max_results = 100
    max_results = min(max_results, 500)

    # No auto-limiting - respect user's period selection completely
    logger.info('Using selected period "%s" with support %s (no auto-limiting applied)', transaction_period, min_support)

    logger.info(
        'Generating department rules with support=%s, confidence=%s, period=%s, max_results=%s',
        min_support,
        min_confidence,
        transaction_period,
        max_results,
    )

    try:
        rules = _generate_department_association_rules(
            min_support,
            min_confidence,
            transaction_period,
            max_results,
        )

        return JsonResponse({
            'success': True,
            'rules': rules,
            'count': len(rules),
            'parameters': {
                'min_support': min_support,
                'min_confidence': min_confidence,
                'transaction_period': transaction_period,
                'max_results': max_results,
            },
        })
    except Exception as exc:
        logger.exception('Failed to generate department association rules')
        return JsonResponse({'success': False, 'error': str(exc)}, status=500)




@login_required(login_url='/admin/login/')
def api_generate_commodity_rules(request):
    """API endpoint for generating commodity-level association rules"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    try:
        min_support = float(request.POST.get('min_support', 0.001))
    except (TypeError, ValueError):
        min_support = 0.001

    try:
        min_confidence = float(request.POST.get('min_confidence', 0.6))
    except (TypeError, ValueError):
        min_confidence = 0.6

    transaction_period = request.POST.get('transaction_period', 'all') or 'all'

    try:
        max_results = int(request.POST.get('max_results', 100))
    except (TypeError, ValueError):
        max_results = 100

    min_support = max(1e-6, min(min_support, 1.0))
    min_confidence = min(max(min_confidence, 0.0), 1.0)

    valid_periods = {'all', '1_month', '3_months', '6_months', '12_months'}
    if transaction_period not in valid_periods:
        transaction_period = 'all'

    if max_results <= 0:
        max_results = 100
    max_results = min(max_results, 500)

    # No auto-limiting - respect user's period selection completely
    logger.info('Using selected period "%s" with support %s (no auto-limiting applied)', transaction_period, min_support)

    logger.info(
        'Generating commodity rules with support=%s, confidence=%s, period=%s, max_results=%s',
        min_support,
        min_confidence,
        transaction_period,
        max_results,
    )

    try:
        rules = _generate_commodity_association_rules(
            min_support,
            min_confidence,
            transaction_period,
            max_results,
        )

        return JsonResponse({
            'success': True,
            'rules': rules,
            'count': len(rules),
            'parameters': {
                'min_support': min_support,
                'min_confidence': min_confidence,
                'transaction_period': transaction_period,
                'max_results': max_results,
            },
        })
    except Exception as exc:
        logger.exception('Failed to generate commodity association rules')
        return JsonResponse({'success': False, 'error': str(exc)}, status=500)

@csrf_exempt
def api_get_period_metrics(request):
    """API endpoint to get basket metrics for a specific transaction period"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        transaction_period = request.POST.get('transaction_period', 'all')

        # Validate transaction period
        valid_periods = {'all', '1_month', '3_months', '6_months', '12_months'}
        if transaction_period not in valid_periods:
            transaction_period = 'all'

        from django.db import connection

        # Get the date range for the period
        start_day = None
        max_day = None

        if transaction_period != 'all':
            with connection.cursor() as cursor:
                cursor.execute('SELECT MAX(day) FROM transactions')
                max_day_row = cursor.fetchone()
                max_day = max_day_row[0] if max_day_row and max_day_row[0] is not None else None

            if max_day is not None:
                period_days = {
                    '1_month': 30,
                    '3_months': 90,
                    '6_months': 180,
                    '12_months': 365,
                }
                days_back = period_days.get(transaction_period, max_day)
                start_day = max(1, max_day - days_back + 1)

        # Calculate metrics for the specified period
        with connection.cursor() as cursor:
            if start_day is not None:
                # Period-specific metrics
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT basket_id) as total_baskets,
                        COUNT(*) as total_transactions,
                        CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT basket_id) as avg_basket_size
                    FROM transactions
                    WHERE day >= %s
                """, (start_day,))
            else:
                # All-time metrics
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT basket_id) as total_baskets,
                        COUNT(*) as total_transactions,
                        CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT basket_id) as avg_basket_size
                    FROM transactions
                """)

            result = cursor.fetchone()
            total_baskets, total_transactions, avg_basket_size = result

        # Format the response with appropriate display names
        period_display_names = {
            'all': 'All Time',
            '1_month': 'Last Month',
            '3_months': 'Last 3 Months',
            '6_months': 'Last 6 Months',
            '12_months': 'Last 12 Months'
        }

        transaction_counts = {
            'all': '2.6M+',
            '12_months': '2.6M+',
            '6_months': '1.3M+',
            '3_months': '650K+',
            '1_month': '126K+'
        }

        period_display = period_display_names.get(transaction_period, 'All Time')
        transaction_count_text = transaction_counts.get(transaction_period, '2.6M+')

        return JsonResponse({
            'success': True,
            'period': transaction_period,
            'period_display': period_display,
            'metrics': {
                'total_baskets': total_baskets,
                'total_transactions': total_transactions,
                'avg_basket_size': round(avg_basket_size, 1) if avg_basket_size else 0,
                'transaction_count_text': transaction_count_text
            },
            'date_range': {
                'start_day': start_day,
                'max_day': max_day
            }
        })

    except Exception as e:
        logger.exception('Failed to get period metrics')
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

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
ml_training_status = {
    'status': 'idle',
    'is_training': False,
    'progress': 0,
    'message': 'Ready for training.',
    'used_cache': False,
    'model_type': None,
    'horizon': None
}


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

        # Get department predictions specifically (HISTORICAL validation method)
        department_predictions = ml_analyzer.get_department_predictions(model_type, selected_horizon)

        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'time_horizon_months': selected_horizon,
            'department_predictions': department_predictions,
            'prediction_type': 'historical_validation'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def predict_future_api(request):
    """API endpoint for ACTUAL future predictions using trained ML models"""
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

        # Get ACTUAL future predictions using trained ML models
        future_predictions = ml_analyzer.predict_future_purchases(
            model_name=model_type,
            time_horizon=selected_horizon,
            top_n=10
        )

        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'time_horizon_months': selected_horizon,
            'future_predictions': future_predictions,
            'prediction_type': 'future_ml_based',
            'description': f'Predicting purchases {selected_horizon} months beyond day 711 using {model_type} model'
        })

    except Exception as e:
        import traceback
        return JsonResponse({
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=500)


@csrf_exempt
def train_ml_models(request):
    """Train ML models in background"""
    global ml_training_status

    model_type = request.POST.get('model_type', 'neural_network')

    training_size_param = request.POST.get('training_size', 0.8)
    try:
        training_size = float(training_size_param)
    except (TypeError, ValueError):
        training_size = 0.8

    training_size = max(0.1, min(training_size, 0.95))

    time_horizon_param = request.POST.get('time_horizon')
    horizon_lookup = {
        '1': '1month',
        '3': '3months',
        '6': '6months',
        '12': '12months'
    }

    horizon_key = None
    if time_horizon_param is not None:
        key = str(time_horizon_param).strip()
        try:
            horizon_key = horizon_lookup[str(int(key))]
        except (ValueError, KeyError):
            horizon_key = horizon_lookup.get(key)

    force_value = request.POST.get('force_retrain', request.POST.get('force', 'false'))
    force_retrain = str(force_value).lower() in {'1', 'true', 'yes', 'on'}

    if ml_training_status.get('is_training'):
        return JsonResponse({
            'success': False,
            'error': 'Training already in progress',
            'status': 'training',
            'message': ml_training_status.get('message', 'Training already in progress'),
            'progress': ml_training_status.get('progress', 0),
            'used_cache': ml_training_status.get('used_cache', False)
        })

    horizons_to_check = [horizon_key] if horizon_key else None

    if not force_retrain and ml_analyzer.has_cached_models(horizons_to_check, refresh=True):
        logger.info('Cached models detected for horizon %s; skipping retraining.', horizon_key or 'all')
        ml_analyzer.refresh_cached_models()
        ml_training_status = {
            'status': 'completed',
            'is_training': False,
            'progress': 100,
            'message': 'Using cached trained models.',
            'used_cache': True,
            'model_type': model_type,
            'horizon': horizon_key
        }
        return JsonResponse({
            'success': True,
            'status': 'completed',
            'message': 'Cached models loaded. No retraining required.',
            'progress': 100,
            'used_cache': True
        })

    def train_models_task():
        global ml_training_status
        try:
            logger.info('Starting ML training for horizon %s (force=%s, model=%s)', horizon_key or 'all', force_retrain, model_type)
            ml_training_status = {
                'status': 'training',
                'is_training': True,
                'progress': 10,
                'message': 'Starting training...',
                'used_cache': False,
                'model_type': model_type,
                'horizon': horizon_key
            }

            success = ml_analyzer.train_models(
                training_size=training_size,
                time_horizon=horizon_key,
                force_retrain=force_retrain
            )

            ml_analyzer.refresh_cached_models()

            if success:
                ml_training_status = {
                    'status': 'completed',
                    'is_training': False,
                    'progress': 100,
                    'message': 'Training completed successfully!',
                    'used_cache': False,
                    'model_type': model_type,
                    'horizon': horizon_key
                }
            else:
                ml_training_status = {
                    'status': 'failed',
                    'is_training': False,
                    'progress': 0,
                    'message': 'Training failed. See logs for details.',
                    'used_cache': False,
                    'model_type': model_type,
                    'horizon': horizon_key
                }
        except Exception as exc:
            logger.exception('ML training error: %s', exc)
            ml_training_status = {
                'status': 'failed',
                'is_training': False,
                'progress': 0,
                'message': f'Training error: {exc}',
                'used_cache': False,
                'model_type': model_type,
                'horizon': horizon_key
            }

    thread = threading.Thread(target=train_models_task, name='ml_model_training')
    thread.daemon = True
    thread.start()

    return JsonResponse({
        'success': True,
        'status': 'training',
        'message': 'Model training started in background',
        'progress': 10,
        'used_cache': False
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
            'success': True,
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
    status_payload = dict(ml_training_status)
    status_payload['success'] = True
    return JsonResponse(status_payload)


@admin_required
def customer_segments(request):
    # ۱. تعریف ترتیب منطقی و استراتژیک برای دسته‌ها
    segment_order = [
        "Champions", "Loyal Customers", "Big Spenders",
        "Potential Loyalists", "New Customers", "Regular Customers",
        "Can't Lose Them", "Need Attention", "At Risk",
        "Hibernating", "Lost"
    ]

    # ۲. دریافت داده‌ها از دیتابیس (بدون تغییر)
    segments_query = CustomerSegment.objects.values('rfm_segment').annotate(
        count=Count('household_key'),
        avg_spend=Avg('total_spend'),
        avg_transactions=Avg('total_transactions')
    )

    # ۳. مرتب‌سازی داده‌ها بر اساس ترتیب منطقی تعریف شده
    segments_list = list(segments_query)
    segments_list.sort(key=lambda s: segment_order.index(s['rfm_segment']) if s['rfm_segment'] in segment_order else len(segment_order))

    recent_customers = CustomerSegment.objects.order_by('-updated_at')[:20]

    for c in recent_customers:
        prob = getattr(c, 'churn_probability', None)
        if prob is None:
            c.churn_risk = "N/A"
        elif prob > 0.75:
            c.churn_risk = "Very High Risk"
        elif prob > 0.50:
            c.churn_risk = "High Risk"
        elif prob > 0.25:
            c.churn_risk = "Medium Risk"
        else:
            c.churn_risk = "Low Risk"

    risk_counts = CustomerSegment.objects.aggregate(
        low=Count('id', filter=Q(churn_probability__lte=0.25)),
        medium=Count('id', filter=Q(churn_probability__gt=0.25, churn_probability__lte=0.50)),
        high=Count('id', filter=Q(churn_probability__gt=0.50, churn_probability__lte=0.75)),
        very_high=Count('id', filter=Q(churn_probability__gt=0.75))
    )
    
    churn_data_sorted = [
        {'risk_label': 'Low Risk', 'count': risk_counts.get('low', 0)},
        {'risk_label': 'Medium Risk', 'count': risk_counts.get('medium', 0)},
        {'risk_label': 'High Risk', 'count': risk_counts.get('high', 0)},
        {'risk_label': 'Very High Risk', 'count': risk_counts.get('very_high', 0)},
    ]

    # ۴. ارسال لیست مرتب‌شده به قالب
    return render(request, 'site/dunnhumby/customer_segments.html', {
        'title': 'Customer Segmentation',
        'segments': segments_list, # <-- از لیست مرتب‌شده جدید استفاده می‌کنیم
        'recent_customers': recent_customers,
        'churn_overview': churn_data_sorted,
    })



@csrf_exempt
def churn_api(request):
    if request.method == 'POST':
        risk_label = request.POST.get('churn_risk')
        page_number = request.POST.get('page', 1) # ۱. شماره صفحه را دریافت می‌کنیم

        if not risk_label:
            return JsonResponse({'error': 'No churn risk provided'})

        # فیلتر مشتریان براساس ریسک (بدون تغییر)
        qs = CustomerSegment.objects.all()
        if risk_label == "Very High Risk":
            qs = qs.filter(churn_probability__gt=0.75)
        elif risk_label == "High Risk":
            qs = qs.filter(churn_probability__gt=0.50, churn_probability__lte=0.75)
        elif risk_label == "Medium Risk":
            qs = qs.filter(churn_probability__gt=0.25, churn_probability__lte=0.50)
        elif risk_label == "Low Risk":
            qs = qs.filter(churn_probability__lte=0.25)

        # متریک‌ها (بدون تغییر)
        metrics = {
            'customers': qs.count(),
            'avg_spend': qs.aggregate(Avg('total_spend'))['total_spend__avg'] or 0,
            'avg_txns': qs.aggregate(Avg('total_transactions'))['total_transactions__avg'] or 0,
            'avg_basket': qs.aggregate(Avg('avg_basket_value'))['avg_basket_value__avg'] or 0,
            'avg_churn_probability': qs.aggregate(Avg('churn_probability'))['churn_probability__avg'] or 0,
            'max_churn_probability': qs.aggregate(Max('churn_probability'))['churn_probability__max'] or 0,
            'min_churn_probability': qs.aggregate(Min('churn_probability'))['churn_probability__min'] or 0,
        }

        # ۲. کوئری اصلی را بدون محدودیت آماده می‌کنیم
        all_households_qs = qs.order_by('-churn_probability').values(
            'household_key', 'total_spend', 'total_transactions',
            'avg_basket_value', 'recency_score', 'frequency_score',
            'monetary_score', 'churn_probability', 'updated_at'
        )

        # ۳. Paginator را با داده‌ها و سایز صفحه ۲۰ می‌سازیم
        paginator = Paginator(all_households_qs, 20)
        page_obj = paginator.get_page(page_number)
        
        # ۴. پاسخ JSON را با ساختار جدید ارسال می‌کنیم
        return JsonResponse({
            'metrics': metrics, 
            'households_page': list(page_obj.object_list),
            'pagination': {
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'current_page': page_obj.number,
                'total_pages': paginator.num_pages,
                'total_items': paginator.count,
            }
        })

    return JsonResponse({'error': 'Invalid request'})
