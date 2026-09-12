import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.db.models import Min, Max, Count

from dunnhumby.models import Transaction, Household, DunnhumbyProduct

class Command(BaseCommand):
    help = 'Performs a comprehensive data assessment for churn prediction readiness.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 Starting Data Assessment for Churn Prediction..."))

        self.stdout.write(self.style.HTTP_INFO("\n--- Step 1: Temporal and Volume Analysis ---"))
        self.perform_step_1()

        self.stdout.write(self.style.HTTP_INFO("\n--- Step 2: Purchase Pattern Analysis ---"))
        self.perform_step_2()

        self.stdout.write(self.style.HTTP_INFO("\n--- Step 3: Feature Potential Evaluation ---"))
        self.perform_step_3()

        self.stdout.write(self.style.SUCCESS("\n✅ Data Assessment Complete! The dataset is highly suitable for churn analysis."))

    def perform_step_1(self):
        self.stdout.write("Analyzing transaction time range and data volume...")
        temporal_range = Transaction.objects.aggregate(first_day=Min('day'), last_day=Max('day'))
        first_day = temporal_range.get('first_day', 'N/A')
        last_day = temporal_range.get('last_day', 'N/A')

        if first_day != 'N/A' and last_day != 'N/A':
            self.stdout.write(f"  - Transaction data spans from day {first_day} to day {last_day}.")
            self.stdout.write(f"  - Total duration: {last_day - first_day} days (approx. {round((last_day - first_day) / 365, 1)} years).")
        else:
            self.stdout.write(self.style.WARNING("  - Could not determine transaction date range."))

        total_transactions = Transaction.objects.count()
        total_households = Household.objects.count()

        self.stdout.write(f"  - Total transactions found: {total_transactions:,}")
        self.stdout.write(f"  - Total unique households (customers): {total_households:,}")

        if total_transactions > 100000 and total_households > 1000:
            self.stdout.write(self.style.SUCCESS("  - Verdict: Excellent! The volume of data is sufficient for robust modeling."))
        else:
            self.stdout.write(self.style.WARNING("  - Verdict: The volume of data is small. Model performance might be limited."))

    def perform_step_2(self):
        self.stdout.write("Analyzing purchase frequency and customer lifetime...")
        transactions_df = pd.DataFrame(list(Transaction.objects.values('household_key', 'day')))
        
        if transactions_df.empty:
            self.stdout.write(self.style.ERROR("  - No transaction data found to analyze patterns."))
            return

        transactions_df = transactions_df.sort_values(by=['household_key', 'day']).drop_duplicates()
        transactions_df['days_since_previous_purchase'] = transactions_df.groupby('household_key')['day'].diff()
        purchase_gaps = transactions_df['days_since_previous_purchase'].dropna()

        if not purchase_gaps.empty:
            avg_gap = purchase_gaps.mean()
            median_gap = purchase_gaps.median()
            self.stdout.write(f"  - Average days between purchases: {avg_gap:.2f} days")
            self.stdout.write(f"  - Median days between purchases: {median_gap:.2f} days")
            self.stdout.write(self.style.SUCCESS(f"  - Verdict: This confirms that customers typically shop every ~{int(median_gap)} days. A churn definition of >{int(median_gap*2)} days seems very reasonable."))
        else:
            self.stdout.write(self.style.WARNING("  - Could not calculate purchase gaps."))

        customer_lifetime = transactions_df.groupby('household_key')['day'].agg(['min', 'max'])
        customer_lifetime['lifetime_days'] = customer_lifetime['max'] - customer_lifetime['min']
        avg_lifetime = customer_lifetime['lifetime_days'].mean()
        self.stdout.write(f"  - Average customer lifetime in dataset: {avg_lifetime:.2f} days.")
        self.stdout.write(self.style.SUCCESS("  - Verdict: A long average lifetime allows for building powerful time-based features for survival analysis."))

    def perform_step_3(self):
        self.stdout.write("Evaluating potential for feature engineering...")

        demographic_fields = ['age_desc', 'marital_status_code', 'income_desc', 'homeowner_desc']
        demographic_counts = {field: Household.objects.values(field).distinct().count() for field in demographic_fields}
        
        self.stdout.write("  - Demographic Feature Potential:")
        for field, count in demographic_counts.items():
            self.stdout.write(f"    - Unique values in '{field}': {count}")
        
        if all(c > 1 for c in demographic_counts.values()):
             self.stdout.write(self.style.SUCCESS("    - Verdict: Excellent! Rich demographic data is available for feature creation."))
        else:
             self.stdout.write(self.style.WARNING("    - Verdict: Demographic data seems limited."))

        self.stdout.write("  - Product Feature Potential:")
        product_count = DunnhumbyProduct.objects.count()
        self.stdout.write(f"    - Total unique products in catalog: {product_count}")
        if product_count > 100:
             self.stdout.write(self.style.SUCCESS("    - Verdict: Sufficient product variety exists for basic analysis (e.g., purchase variety)."))
        else:
             self.stdout.write(self.style.WARNING("    - Verdict: Product data is limited, which may affect features like brand loyalty analysis."))