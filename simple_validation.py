#!/usr/bin/env python3
"""
Simple validation script using direct SQL Server connection
"""
import pyodbc
from collections import defaultdict

def connect_to_database():
    """Connect to SQL Server database"""
    try:
        # Try to connect using Windows Authentication
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=.\\SQLEXPRESS;"
            "DATABASE=marketdb;"
            "Trusted_Connection=yes;"
        )
        conn = pyodbc.connect(conn_str)
        return conn
    except:
        try:
            # Alternative connection string
            conn_str = (
                "DRIVER={SQL Server};"
                "SERVER=.\\SQLEXPRESS;"
                "DATABASE=marketdb;"
                "Trusted_Connection=yes;"
            )
            conn = pyodbc.connect(conn_str)
            return conn
        except Exception as e:
            print(f"❌ Cannot connect to database: {e}")
            return None

def validate_association_rules():
    """Validate the association rules by tracing through actual calculations"""

    print("=== ASSOCIATION RULES VALIDATION ===\n")

    conn = connect_to_database()
    if not conn:
        print("❌ Database connection failed")
        return False

    cursor = conn.cursor()

    try:
        # Get latest day to determine 1-month period
        cursor.execute('SELECT MAX(day) FROM transactions')
        max_day = cursor.fetchone()[0]

        start_day = max(1, max_day - 30 + 1)  # 1 month period
        print(f"Dataset Period: Days {start_day} to {max_day} (1 month)")

        # Count total baskets in this period
        cursor.execute('SELECT COUNT(DISTINCT basket_id) FROM transactions WHERE day >= ?', (start_day,))
        total_baskets = cursor.fetchone()[0]

        print(f"Total Baskets in Period: {total_baskets:,}")

        # Get department names to verify the ones we saw
        dept_query = """
            SELECT p.department, COUNT(DISTINCT t.basket_id) as basket_count
            FROM transactions t
            JOIN product p ON t.product_id = p.product_id
            WHERE t.day >= ? AND p.department IS NOT NULL
            GROUP BY p.department
            ORDER BY basket_count DESC
        """

        cursor.execute(dept_query, (start_day,))
        dept_results = cursor.fetchall()

        print(f"\nTop 10 Departments by Basket Count:")
        print("Department".ljust(25) + "Baskets".rjust(10) + "Support %".rjust(12))
        print("-" * 50)

        dept_data = {}
        for i, (dept, count) in enumerate(dept_results[:10]):
            support_pct = (count / total_baskets) * 100
            dept_data[dept] = {'baskets': count, 'support': support_pct}
            print(f"{dept[:23].ljust(25)}{count:>8,}{support_pct:>10.2f}%")

        # Look for the specific departments we saw in results
        target_depts = ['TRAVEL & LEISUR', 'MEAT', 'CHEF SHOPPE', 'PRODUCE']
        print(f"\n=== CHECKING TARGET DEPARTMENTS ===")

        for dept in target_depts:
            cursor.execute("""
                SELECT COUNT(DISTINCT t.basket_id)
                FROM transactions t
                JOIN product p ON t.product_id = p.product_id
                WHERE t.day >= ? AND p.department = ?
            """, (start_day, dept))

            result = cursor.fetchone()
            count = result[0] if result else 0
            support = (count / total_baskets) * 100 if count > 0 else 0
            print(f"{dept.ljust(20)}: {count:>6,} baskets ({support:>5.2f}%)")

        # Trace specific rule: TRAVEL & LEISUR → MEAT
        print(f"\n=== TRACING: TRAVEL & LEISUR → MEAT ===")

        # Count baskets with TRAVEL & LEISUR
        cursor.execute("""
            SELECT COUNT(DISTINCT t.basket_id)
            FROM transactions t
            JOIN product p ON t.product_id = p.product_id
            WHERE t.day >= ? AND p.department = 'TRAVEL & LEISUR'
        """, (start_day,))
        travel_baskets = cursor.fetchone()[0]

        # Count baskets with MEAT
        cursor.execute("""
            SELECT COUNT(DISTINCT t.basket_id)
            FROM transactions t
            JOIN product p ON t.product_id = p.product_id
            WHERE t.day >= ? AND p.department = 'MEAT'
        """, (start_day,))
        meat_baskets = cursor.fetchone()[0]

        # Count baskets with both TRAVEL & LEISUR and MEAT
        cursor.execute("""
            SELECT COUNT(DISTINCT t1.basket_id)
            FROM transactions t1
            JOIN product p1 ON t1.product_id = p1.product_id
            WHERE t1.day >= ?
            AND p1.department = 'TRAVEL & LEISUR'
            AND EXISTS (
                SELECT 1 FROM transactions t2
                JOIN product p2 ON t2.product_id = p2.product_id
                WHERE t2.basket_id = t1.basket_id
                AND t2.day >= ?
                AND p2.department = 'MEAT'
            )
        """, (start_day, start_day))
        both_baskets = cursor.fetchone()[0]

        # Calculate metrics
        support = both_baskets / total_baskets if total_baskets > 0 else 0
        confidence = both_baskets / travel_baskets if travel_baskets > 0 else 0
        expected_meat_rate = meat_baskets / total_baskets if total_baskets > 0 else 0
        lift = confidence / expected_meat_rate if expected_meat_rate > 0 else 0

        print(f"Baskets with TRAVEL & LEISUR: {travel_baskets:,}")
        print(f"Baskets with MEAT: {meat_baskets:,}")
        print(f"Baskets with BOTH: {both_baskets:,}")
        print(f"Support: {both_baskets}/{total_baskets} = {support:.4f} ({support*100:.2f}%)")
        print(f"Confidence: {both_baskets}/{travel_baskets} = {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"Expected MEAT rate: {meat_baskets}/{total_baskets} = {expected_meat_rate:.4f}")
        print(f"Lift: {confidence:.4f} / {expected_meat_rate:.4f} = {lift:.2f}")

        # Check if this matches our screenshot results
        expected_support = 0.002  # 0.2%
        expected_confidence = 0.649  # 64.9%
        expected_lift = 3.11

        print(f"\n=== COMPARISON WITH DISPLAYED RESULTS ===")
        support_match = abs(support - expected_support) < 0.001
        confidence_match = abs(confidence - expected_confidence) < 0.05
        lift_match = abs(lift - expected_lift) < 0.5

        print(f"Expected Support: {expected_support:.1%} | Calculated: {support:.1%} | Match: {'✅' if support_match else '❌'}")
        print(f"Expected Confidence: {expected_confidence:.1%} | Calculated: {confidence:.1%} | Match: {'✅' if confidence_match else '❌'}")
        print(f"Expected Lift: {expected_lift:.2f} | Calculated: {lift:.2f} | Match: {'✅' if lift_match else '❌'}")

        # Sample some actual baskets to verify
        print(f"\n=== SAMPLE VERIFICATION ===")
        cursor.execute("""
            SELECT TOP 5 t1.basket_id, t1.household_key
            FROM transactions t1
            JOIN product p1 ON t1.product_id = p1.product_id
            WHERE t1.day >= ?
            AND p1.department = 'TRAVEL & LEISUR'
            AND EXISTS (
                SELECT 1 FROM transactions t2
                JOIN product p2 ON t2.product_id = p2.product_id
                WHERE t2.basket_id = t1.basket_id
                AND p2.department = 'MEAT'
            )
        """, (start_day,))

        sample_baskets = cursor.fetchall()
        print(f"Sample baskets with both TRAVEL & LEISUR and MEAT:")
        for basket_id, household_key in sample_baskets:
            print(f"  Basket {basket_id} (Household {household_key})")

        # Check data reasonableness
        print(f"\n=== DATA QUALITY ASSESSMENT ===")

        if total_baskets < 50000:
            print("⚠️  Low basket count - may indicate data filtering is too aggressive")
        elif total_baskets > 200000:
            print("⚠️  High basket count - may not be using 1-month filter correctly")
        else:
            print("✅ Reasonable basket count for 1-month period")

        if both_baskets == 0:
            print("❌ No baskets found with both departments - rule may be invalid")
        elif both_baskets < 10:
            print("⚠️  Very few baskets with both departments - rule may be unstable")
        else:
            print("✅ Sufficient basket count for reliable rule")

        if confidence > 0.9:
            print("⚠️  Very high confidence - may indicate data bias or small sample")
        elif confidence < 0.1:
            print("⚠️  Very low confidence - rule may not be meaningful")
        else:
            print("✅ Reasonable confidence level")

        if lift < 1.1:
            print("⚠️  Low lift - weak association")
        elif lift > 10:
            print("⚠️  Very high lift - may indicate data anomaly")
        else:
            print("✅ Good lift value indicating strong association")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        conn.close()

if __name__ == "__main__":
    validate_association_rules()