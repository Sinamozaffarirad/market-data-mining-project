from django.core.management.base import BaseCommand
from django.db import connection, transaction as db_transaction


class Command(BaseCommand):
    help = 'Remove duplicate transactions keeping only unique combinations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        if dry_run:
            self.stdout.write("DRY RUN MODE - No data will be deleted")

        self.show_current_counts()

        if not dry_run:
            self.remove_duplicates()

            self.stdout.write("\n" + "="*50)
            self.stdout.write("AFTER CLEANUP:")
            self.show_current_counts()
        else:
            self.show_duplicate_preview()

    def show_current_counts(self):
        with connection.cursor() as cursor:
       
            cursor.execute("SELECT COUNT(*) FROM transactions")
            total = cursor.fetchone()[0]

           
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT basket_id, product_id, day
                    FROM transactions
                ) t
            """)
            unique = cursor.fetchone()[0]

            duplicates = total - unique

            self.stdout.write(f"Total transactions: {total:,}")
            self.stdout.write(f"Unique combinations: {unique:,}")
            self.stdout.write(f"Duplicates to remove: {duplicates:,}")

    def show_duplicate_preview(self):
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT TOP 10 basket_id, product_id, day, COUNT(*) as count
                FROM transactions
                GROUP BY basket_id, product_id, day
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
            """)

            self.stdout.write("\nSample duplicates (basket_id, product_id, day, count):")
            for row in cursor.fetchall():
                self.stdout.write(f"  {row[0]}, {row[1]}, {row[2]} -> {row[3]} copies")

    def remove_duplicates(self):

        self.stdout.write("Removing duplicate transactions...")

        with connection.cursor() as cursor:
            self.stdout.write("Step 1: Creating temporary table with unique transactions...")

            cursor.execute("""
                -- Create temporary table with unique transactions
                SELECT basket_id, product_id, day, household_key, quantity, sales_value,
                       store_id, retail_disc, trans_time, week_no, coupon_disc, coupon_match_disc,
                       MAX(id) as max_id
                INTO #unique_transactions
                FROM transactions
                GROUP BY basket_id, product_id, day, household_key, quantity, sales_value,
                         store_id, retail_disc, trans_time, week_no, coupon_disc, coupon_match_disc
            """)

            cursor.execute("SELECT COUNT(*) FROM #unique_transactions")
            unique_count = cursor.fetchone()[0]
            self.stdout.write(f"Found {unique_count:,} unique transactions")

            self.stdout.write("Step 2: Removing duplicate transactions...")

            cursor.execute("""
                DELETE FROM transactions
                WHERE id NOT IN (SELECT max_id FROM #unique_transactions)
            """)

            deleted_count = cursor.rowcount
            self.stdout.write(f"Deleted {deleted_count:,} duplicate transactions")

            cursor.execute("DROP TABLE #unique_transactions")

        self.stdout.write(self.style.SUCCESS("Duplicate removal completed!"))

    def verify_csv_match(self):
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM transactions")
            db_count = cursor.fetchone()[0]

            csv_count = 2595732

            self.stdout.write(f"\nVerification:")
            self.stdout.write(f"Database count: {db_count:,}")
            self.stdout.write(f"CSV count: {csv_count:,}")

            if db_count == csv_count:
                self.stdout.write(self.style.SUCCESS("✅ Counts match perfectly!"))
            else:
                difference = abs(db_count - csv_count)
                self.stdout.write(self.style.WARNING(f"⚠️  Difference: {difference:,} rows"))