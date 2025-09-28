#!/usr/bin/env python3
"""
Validation script to trace and verify association rules calculations
"""
import sys
import os

# Add the Django project to the path
sys.path.append(r'C:\Local Disk D\UNI Files\Final Project\Website')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')

import django
django.setup()

from market.dunnhumby.models import Transaction, DunnhumbyProduct
from django.db import connection
from collections import defaultdict, Counter

def validate_association_rules():
    """Validate the association rules by tracing through actual calculations"""

    print("=== ASSOCIATION RULES VALIDATION ===\n")

    # Get latest day to determine 1-month period
    with connection.cursor() as cursor:
        cursor.execute('SELECT MAX(day) FROM transactions')
        max_day = cursor.fetchone()[0]

    start_day = max(1, max_day - 30 + 1)  # 1 month period
    print(f"Dataset Period: Days {start_day} to {max_day} (1 month)")

    # Count total baskets in this period
    with connection.cursor() as cursor:
        cursor.execute('SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= %s', (start_day,))
        total_baskets = cursor.fetchone()[0]

    print(f"Total Baskets in Period: {total_baskets:,}")

    # Get top departments by frequency
    dept_query = """
        SELECT p.department, COUNT(DISTINCT t.basket_id) as basket_count
        FROM transactions t
        JOIN product p ON t.product_id = p.product_id
        WHERE t.day >= %s AND p.department IS NOT NULL
        GROUP BY p.department
        ORDER BY basket_count DESC
        LIMIT 10
    """

    with connection.cursor() as cursor:
        cursor.execute(dept_query, (start_day,))
        dept_results = cursor.fetchall()

    print(f"\nTop 10 Departments by Basket Count:")
    print("Department".ljust(20) + "Baskets".rjust(10) + "Support %".rjust(12))
    print("-" * 45)

    for dept, count in dept_results:
        support_pct = (count / total_baskets) * 100
        print(f"{dept[:18].ljust(20)}{count:>8,}{support_pct:>10.2f}%")

    # Trace specific rule: TRAVEL & LEISURE → MEAT
    print(f"\n=== TRACING: TRAVEL & LEISURE → MEAT ===")

    # Count baskets with TRAVEL & LEISURE
    travel_query = """
        SELECT COUNT(DISTINCT t.basket_id)
        FROM transactions t
        JOIN product p ON t.product_id = p.product_id
        WHERE t.day >= %s AND p.department = 'TRAVEL & LEISUR'
    """

    with connection.cursor() as cursor:
        cursor.execute(travel_query, (start_day,))
        travel_baskets = cursor.fetchone()[0]

    # Count baskets with both TRAVEL & LEISURE and MEAT
    both_query = """
        SELECT COUNT(DISTINCT t1.basket_id)
        FROM transactions t1
        JOIN product p1 ON t1.product_id = p1.product_id
        WHERE t1.day >= %s
        AND p1.department = 'TRAVEL & LEISUR'
        AND EXISTS (
            SELECT 1 FROM transactions t2
            JOIN product p2 ON t2.product_id = p2.product_id
            WHERE t2.basket_id = t1.basket_id
            AND p2.department = 'MEAT'
        )
    """

    with connection.cursor() as cursor:
        cursor.execute(both_query, (start_day,))
        both_baskets = cursor.fetchone()[0]

    # Count baskets with MEAT
    meat_query = """
        SELECT COUNT(DISTINCT t.basket_id)
        FROM transactions t
        JOIN product p ON t.product_id = p.product_id
        WHERE t.day >= %s AND p.department = 'MEAT'
    """

    with connection.cursor() as cursor:
        cursor.execute(meat_query, (start_day,))
        meat_baskets = cursor.fetchone()[0]

    # Calculate metrics
    support = both_baskets / total_baskets
    confidence = both_baskets / travel_baskets if travel_baskets > 0 else 0
    lift = confidence / (meat_baskets / total_baskets) if meat_baskets > 0 else 0

    print(f"Baskets with TRAVEL & LEISUR: {travel_baskets:,}")
    print(f"Baskets with MEAT: {meat_baskets:,}")
    print(f"Baskets with BOTH: {both_baskets:,}")
    print(f"Support: {both_baskets}/{total_baskets} = {support:.4f} ({support*100:.2f}%)")
    print(f"Confidence: {both_baskets}/{travel_baskets} = {confidence:.4f} ({confidence*100:.1f}%)")
    print(f"Expected MEAT rate: {meat_baskets}/{total_baskets} = {meat_baskets/total_baskets:.4f}")
    print(f"Lift: {confidence:.4f} / {meat_baskets/total_baskets:.4f} = {lift:.2f}")

    # Check if this matches our screenshot results
    expected_support = 0.002  # 0.2%
    expected_confidence = 0.649  # 64.9%
    expected_lift = 3.11

    print(f"\n=== COMPARISON WITH DISPLAYED RESULTS ===")
    print(f"Expected Support: {expected_support:.1%} | Calculated: {support:.1%} | Match: {abs(support - expected_support) < 0.001}")
    print(f"Expected Confidence: {expected_confidence:.1%} | Calculated: {confidence:.1%} | Match: {abs(confidence - expected_confidence) < 0.01}")
    print(f"Expected Lift: {expected_lift:.2f} | Calculated: {lift:.2f} | Match: {abs(lift - expected_lift) < 0.1}")

    # Check data quality
    print(f"\n=== DATA QUALITY CHECKS ===")

    # Check for reasonable department names
    weird_depts_query = """
        SELECT DISTINCT p.department
        FROM product p
        WHERE p.department IS NOT NULL
        AND LEN(p.department) < 3
    """

    with connection.cursor() as cursor:
        cursor.execute(weird_depts_query)
        weird_depts = cursor.fetchall()

    if weird_depts:
        print(f"⚠️  Found {len(weird_depts)} departments with < 3 characters: {weird_depts}")
    else:
        print("✅ All department names are reasonable length")

    # Check for null/empty data
    null_check_query = """
        SELECT
            COUNT(*) as total_transactions,
            COUNT(CASE WHEN p.department IS NULL THEN 1 END) as null_depts,
            COUNT(CASE WHEN t.basket_id IS NULL THEN 1 END) as null_baskets
        FROM transactions t
        JOIN product p ON t.product_id = p.product_id
        WHERE t.day >= %s
    """

    with connection.cursor() as cursor:
        cursor.execute(null_check_query, (start_day,))
        total_trans, null_depts, null_baskets = cursor.fetchone()

    print(f"✅ Total transactions in period: {total_trans:,}")
    print(f"✅ Null departments: {null_depts:,} ({null_depts/total_trans*100:.1f}%)")
    print(f"✅ Null baskets: {null_baskets:,}")

    return True

if __name__ == "__main__":
    try:
        validate_association_rules()
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()