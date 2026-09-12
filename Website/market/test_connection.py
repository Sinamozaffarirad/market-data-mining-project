import os
import sys
import django
from django.conf import settings


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')
django.setup()


try:
    from django.db import connection
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]
    print("Database connection successful!")
    print(f"Found {count:,} transactions in the database")
    

    cursor.execute("SELECT COUNT(*) FROM product")
    product_count = cursor.fetchone()[0]
    print(f"Found {product_count:,} products")
    
    cursor.execute("SELECT COUNT(*) FROM household")
    household_count = cursor.fetchone()[0]
    print(f"Found {household_count:,} households")
    
except Exception as e:
    print("Database connection failed:", str(e))
    sys.exit(1)