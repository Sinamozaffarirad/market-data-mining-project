"""
Django management command to clean transactions table to match CSV exactly
"""

from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Clean transactions to keep only unique basket_id, product_id, day combinations'

    def handle(self, *args, **options):
        self.stdout.write("Cleaning transactions table to match CSV exactly...")

        with connection.cursor() as cursor:
            # Check current state
            cursor.execute("SELECT COUNT(*) FROM transactions")
            before_count = cursor.fetchone()[0]
            self.stdout.write(f"Before cleanup: {before_count:,} transactions")

            # Delete duplicates keeping only the one with the highest ID for each unique combination
            self.stdout.write("Removing duplicates based on basket_id, product_id, day...")

            cursor.execute("""
                DELETE FROM transactions
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM transactions
                    GROUP BY basket_id, product_id, day
                )
            """)

            deleted_count = cursor.rowcount
            self.stdout.write(f"Deleted {deleted_count:,} duplicate transactions")

            # Check final state
            cursor.execute("SELECT COUNT(*) FROM transactions")
            after_count = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT basket_id, product_id, day
                    FROM transactions
                ) t
            """)
            unique_combinations = cursor.fetchone()[0]

            self.stdout.write(f"After cleanup: {after_count:,} transactions")
            self.stdout.write(f"Unique combinations: {unique_combinations:,}")

            # Compare with CSV
            csv_rows = 2595732
            self.stdout.write(f"\nCSV file has: {csv_rows:,} data rows")

            if after_count == csv_rows:
                self.stdout.write(self.style.SUCCESS("✅ Perfect match with CSV!"))
            elif after_count == unique_combinations:
                self.stdout.write(self.style.SUCCESS("✅ All duplicates removed!"))
                difference = csv_rows - after_count
                self.stdout.write(f"Difference from CSV: {difference:,} rows (likely due to data validation)")
            else:
                remaining_duplicates = after_count - unique_combinations
                self.stdout.write(self.style.WARNING(f"⚠️  Still {remaining_duplicates:,} duplicates remaining"))