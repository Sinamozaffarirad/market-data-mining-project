#!/usr/bin/env python3
"""
Transaction Data Import Script
Imports transaction_data.csv into the database with proper column mapping and data validation
"""

import os
import sys
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
import logging

# Add Django project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')

import django
django.setup()

from market.dunnhumby.models import Transaction
from django.db import connection, transaction as db_transaction

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_current_db_count():
    """Check current number of rows in database"""
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        logger.info(f"Current database transaction count: {count:,}")
        return count

def validate_and_clean_row(row):
    """Validate and clean a single row of data"""
    try:
        # Handle missing/null values
        for col in ['quantity', 'store_id', 'week_no']:
            if pd.isna(row[col]) or row[col] == '':
                row[col] = None

        # Convert numeric fields
        row['household_key'] = int(row['household_key'])
        row['basket_id'] = int(row['BASKET_ID'])
        row['day'] = int(row['DAY'])
        row['product_id'] = int(row['PRODUCT_ID'])

        # Handle quantity (can be null)
        if row['QUANTITY'] is not None and not pd.isna(row['QUANTITY']):
            row['quantity'] = int(row['QUANTITY'])
        else:
            row['quantity'] = None

        # Handle store_id (can be null)
        if row['STORE_ID'] is not None and not pd.isna(row['STORE_ID']):
            row['store_id'] = int(row['STORE_ID'])
        else:
            row['store_id'] = None

        # Handle week_no (can be null)
        if row['WEEK_NO'] is not None and not pd.isna(row['WEEK_NO']):
            row['week_no'] = int(row['WEEK_NO'])
        else:
            row['week_no'] = None

        # Handle coupon_disc (now contains old TRANS_TIME data - integer)
        if row['COUPON_DISC'] is not None and not pd.isna(row['COUPON_DISC']):
            row['coupon_disc'] = int(row['COUPON_DISC'])
        else:
            row['coupon_disc'] = None

        # Convert decimal fields - note the column swap!
        row['sales_value'] = Decimal(str(row['SALES_VALUE']))
        row['retail_disc'] = Decimal(str(row['RETAIL_DISC']))
        row['trans_time'] = Decimal(str(row['TRANS_TIME']))  # CSV TRANS_TIME goes to DB trans_time
        row['coupon_match_disc'] = Decimal(str(row['COUPON_MATCH_DISC']))

        return True, row
    except (ValueError, InvalidOperation, TypeError) as e:
        logger.warning(f"Invalid row data: {e} - Row: {row}")
        return False, None

def import_transactions_batch(csv_file_path, batch_size=10000):
    """Import transactions in batches with validation"""
    logger.info(f"Starting import from {csv_file_path}")

    # Count current database rows
    initial_count = check_current_db_count()

    # Read CSV in chunks for memory efficiency
    chunk_count = 0
    total_processed = 0
    total_inserted = 0
    total_errors = 0

    try:
        for chunk in pd.read_csv(csv_file_path, chunksize=batch_size):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count} ({len(chunk)} rows)")

            batch_transactions = []

            for index, row in chunk.iterrows():
                total_processed += 1

                # Validate and clean row
                is_valid, cleaned_row = validate_and_clean_row(row)

                if not is_valid:
                    total_errors += 1
                    continue

                # Check if transaction already exists (by basket_id, product_id, day)
                existing = Transaction.objects.filter(
                    basket_id=cleaned_row['basket_id'],
                    product_id=cleaned_row['product_id'],
                    day=cleaned_row['day']
                ).exists()

                if existing:
                    continue  # Skip duplicate

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
                        Transaction.objects.bulk_create(batch_transactions, ignore_conflicts=True)
                    total_inserted += len(batch_transactions)
                    logger.info(f"Inserted {len(batch_transactions)} transactions from chunk {chunk_count}")
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    total_errors += len(batch_transactions)

            # Progress update
            if chunk_count % 10 == 0:
                logger.info(f"Progress: {total_processed:,} processed, {total_inserted:,} inserted, {total_errors:,} errors")

    except Exception as e:
        logger.error(f"Error during import: {e}")
        return False

    # Final count verification
    final_count = check_current_db_count()

    logger.info("=" * 50)
    logger.info("IMPORT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"CSV total rows: 2,595,732")
    logger.info(f"Rows processed: {total_processed:,}")
    logger.info(f"Rows inserted: {total_inserted:,}")
    logger.info(f"Errors/Skipped: {total_errors:,}")
    logger.info(f"Database before: {initial_count:,}")
    logger.info(f"Database after: {final_count:,}")
    logger.info(f"Net increase: {final_count - initial_count:,}")
    logger.info("=" * 50)

    return True

if __name__ == "__main__":
    csv_path = r"C:\Local Disk D\UNI Files\Final Project\transaction_data.csv"

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    logger.info("Starting transaction data import...")
    success = import_transactions_batch(csv_path)

    if success:
        logger.info("Import completed successfully!")
    else:
        logger.error("Import failed!")
        sys.exit(1)