"""
Django management command to import transaction data from CSV
"""

import os
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
from django.core.management.base import BaseCommand
from django.db import connection, transaction as db_transaction
from dunnhumby.models import Transaction, Household


class Command(BaseCommand):
    help = 'Import transaction data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-path',
            type=str,
            default=r"C:\Local Disk D\UNI Files\Final Project\transaction_data.csv",
            help='Path to the CSV file'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10000,
            help='Batch size for bulk operations'
        )
        parser.add_argument(
            '--skip-rows',
            type=int,
            default=0,
            help='Number of rows to skip from the beginning'
        )

    def handle(self, *args, **options):
        csv_path = options['csv_path']
        batch_size = options['batch_size']
        skip_rows = options['skip_rows']

        self.stdout.write(f"Starting import from {csv_path}")

        if not os.path.exists(csv_path):
            self.stderr.write(f"CSV file not found: {csv_path}")
            return

        # Check current database count
        initial_count = self.check_current_db_count()

        # Import data
        success = self.import_transactions_batch(csv_path, batch_size, skip_rows)

        if success:
            final_count = self.check_current_db_count()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Import completed! Database rows: {initial_count:,} → {final_count:,} (+"
                    f"{final_count - initial_count:,})"
                )
            )
        else:
            self.stderr.write("Import failed!")

    def check_current_db_count(self):
        """Check current number of rows in database"""
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM transactions")
            count = cursor.fetchone()[0]
            self.stdout.write(f"Current database transaction count: {count:,}")
            return count

    def validate_and_clean_row(self, row):
        """Validate and clean a single row of data"""
        try:
            # Handle missing/null values
            for col in ['QUANTITY', 'STORE_ID', 'WEEK_NO']:
                if pd.isna(row[col]) or row[col] == '':
                    row[col] = None

            # Convert numeric fields
            cleaned_row = {}
            cleaned_row['household_key'] = int(row['household_key'])
            cleaned_row['basket_id'] = int(row['BASKET_ID'])
            cleaned_row['day'] = int(row['DAY'])
            cleaned_row['product_id'] = int(row['PRODUCT_ID'])

            # Handle quantity (can be null)
            if row['QUANTITY'] is not None and not pd.isna(row['QUANTITY']):
                cleaned_row['quantity'] = int(row['QUANTITY'])
            else:
                cleaned_row['quantity'] = None

            # Handle store_id (can be null)
            if row['STORE_ID'] is not None and not pd.isna(row['STORE_ID']):
                cleaned_row['store_id'] = int(row['STORE_ID'])
            else:
                cleaned_row['store_id'] = None

            # Handle week_no (can be null)
            if row['WEEK_NO'] is not None and not pd.isna(row['WEEK_NO']):
                cleaned_row['week_no'] = int(row['WEEK_NO'])
            else:
                cleaned_row['week_no'] = None

            # Handle coupon_disc (now contains old TRANS_TIME data - integer)
            if row['COUPON_DISC'] is not None and not pd.isna(row['COUPON_DISC']):
                cleaned_row['coupon_disc'] = int(row['COUPON_DISC'])
            else:
                cleaned_row['coupon_disc'] = None

            # Convert decimal fields - note the column swap!
            cleaned_row['sales_value'] = Decimal(str(row['SALES_VALUE']))
            cleaned_row['retail_disc'] = Decimal(str(row['RETAIL_DISC']))
            cleaned_row['trans_time'] = Decimal(str(row['TRANS_TIME']))  # CSV TRANS_TIME goes to DB trans_time
            cleaned_row['coupon_match_disc'] = Decimal(str(row['COUPON_MATCH_DISC']))

            return True, cleaned_row
        except (ValueError, InvalidOperation, TypeError) as e:
            self.stdout.write(f"Invalid row data: {e}")
            return False, None

    def import_transactions_batch(self, csv_file_path, batch_size, skip_rows=0):
        """Import transactions in batches with validation"""
        chunk_count = 0
        total_processed = 0
        total_inserted = 0
        total_errors = 0

        try:
            # Skip rows if specified
            if skip_rows > 0:
                self.stdout.write(f"Skipping first {skip_rows} rows...")
                skip_chunks = skip_rows // batch_size
                chunk_count = skip_chunks
                total_processed = skip_rows

            for chunk in pd.read_csv(csv_file_path, chunksize=batch_size, skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None):
                chunk_count += 1
                self.stdout.write(f"Processing chunk {chunk_count} ({len(chunk)} rows)")

                batch_transactions = []

                for index, row in chunk.iterrows():
                    total_processed += 1

                    # Validate and clean row
                    is_valid, cleaned_row = self.validate_and_clean_row(row)

                    if not is_valid:
                        total_errors += 1
                        continue

                    # Validate household exists to avoid foreign key constraint violation
                    if not Household.objects.filter(household_key=cleaned_row['household_key']).exists():
                        total_errors += 1
                        continue

                    # Create transaction object
                    transaction_obj = Transaction(
                        household_key=cleaned_row['household_key'],
                        basket_id=cleaned_row['basket_id'],
                        day=cleaned_row['day'],
                        product_id=cleaned_row['product_id'],
                        quantity=cleaned_row['quantity'],
                        sales_value=cleaned_row['sales_value'],
                        store_id=cleaned_row['store_id'],
                        retail_disc=cleaned_row['retail_disc'],
                        trans_time=cleaned_row['trans_time'],
                        week_no=cleaned_row['week_no'],
                        coupon_disc=cleaned_row['coupon_disc'],
                        coupon_match_disc=cleaned_row['coupon_match_disc']
                    )

                    batch_transactions.append(transaction_obj)

                # Bulk insert batch
                if batch_transactions:
                    try:
                        with db_transaction.atomic():
                            Transaction.objects.bulk_create(batch_transactions)
                        total_inserted += len(batch_transactions)
                        self.stdout.write(f"Inserted {len(batch_transactions)} transactions from chunk {chunk_count}")
                    except Exception as e:
                        self.stderr.write(f"Error inserting batch: {e}")
                        total_errors += len(batch_transactions)

                # Progress update
                if chunk_count % 10 == 0:
                    self.stdout.write(f"Progress: {total_processed:,} processed, {total_inserted:,} inserted, {total_errors:,} errors")

        except Exception as e:
            self.stderr.write(f"Error during import: {e}")
            return False

        self.stdout.write("=" * 50)
        self.stdout.write("IMPORT SUMMARY")
        self.stdout.write("=" * 50)
        self.stdout.write(f"CSV total rows: 2,595,732")
        self.stdout.write(f"Rows processed: {total_processed:,}")
        self.stdout.write(f"Rows inserted: {total_inserted:,}")
        self.stdout.write(f"Errors/Skipped: {total_errors:,}")
        self.stdout.write("=" * 50)

        return True