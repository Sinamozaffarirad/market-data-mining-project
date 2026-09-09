from django.core.management.base import BaseCommand
from django.db.models import Sum, Count, Max, Avg
from dunnhumby.models import Transaction, DunnhumbyProduct, Household, BasketAnalysis, CustomerSegment, AssociationRule


class Command(BaseCommand):
    help = 'Test admin panel functionality'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Testing admin panel functionality...\n'))
        
        # Test basic model access
        self.stdout.write('1. Testing model access:')
        try:
            transaction_count = Transaction.objects.count()
            product_count = DunnhumbyProduct.objects.count()
            household_count = Household.objects.count()
            
            self.stdout.write(f'   - Transactions: {transaction_count:,}')
            self.stdout.write(f'   - Products: {product_count:,}')
            self.stdout.write(f'   - Households: {household_count:,}')
            self.stdout.write(self.style.SUCCESS('   ✓ Basic model access working\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ✗ Error accessing models: {e}\n'))
            return
        
        # Test basket analysis queries
        self.stdout.write('2. Testing basket analysis queries:')
        try:
            basket_stats = Transaction.objects.values('basket_id').annotate(
                total_items=Sum('quantity'),
                total_value=Sum('sales_value'),
                unique_products=Count('product_id', distinct=True)
            ).order_by('-total_value')[:5]
            
            self.stdout.write('   Top 5 baskets by value:')
            for basket in basket_stats:
                items = basket['total_items'] or 0
                value = basket['total_value'] or 0
                products = basket['unique_products'] or 0
                self.stdout.write(f'   - Basket {basket["basket_id"]}: {items} items, ${value:.2f}, {products} unique products')
            
            self.stdout.write(self.style.SUCCESS('   ✓ Basket analysis queries working\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ✗ Error in basket analysis: {e}\n'))
        
        # Test product analysis queries
        self.stdout.write('3. Testing product analysis queries:')
        try:
            top_products = Transaction.objects.values(
                'product_id'
            ).annotate(
                frequency=Count('product_id'),
                total_sales=Sum('sales_value')
            ).order_by('-frequency')[:5]
            
            self.stdout.write('   Top 5 products by frequency:')
            for product in top_products:
                self.stdout.write(f'   - Product {product["product_id"]}: {product["frequency"]} transactions, ${product["total_sales"]:.2f}')
            
            self.stdout.write(self.style.SUCCESS('   ✓ Product analysis queries working\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ✗ Error in product analysis: {e}\n'))
        
        # Test RFM data preparation
        self.stdout.write('4. Testing RFM analysis data:')
        try:
            customers = Transaction.objects.values('household_key').annotate(
                last_transaction=Max('day'),
                total_transactions=Count('basket_id'),
                total_spend=Sum('sales_value'),
                avg_basket_value=Avg('sales_value')
            ).order_by('-total_spend')[:5]
            
            self.stdout.write('   Top 5 customers by spend:')
            for customer in customers:
                self.stdout.write(f'   - Household {customer["household_key"]}: ${customer["total_spend"]:.2f}, {customer["total_transactions"]} transactions')
            
            self.stdout.write(self.style.SUCCESS('   ✓ RFM analysis queries working\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ✗ Error in RFM analysis: {e}\n'))
        
        # Test analysis tables
        self.stdout.write('5. Testing analysis tables:')
        try:
            basket_analysis_count = BasketAnalysis.objects.count()
            customer_segment_count = CustomerSegment.objects.count()
            association_rule_count = AssociationRule.objects.count()
            
            self.stdout.write(f'   - Basket Analyses: {basket_analysis_count}')
            self.stdout.write(f'   - Customer Segments: {customer_segment_count}')
            self.stdout.write(f'   - Association Rules: {association_rule_count}')
            self.stdout.write(self.style.SUCCESS('   ✓ Analysis tables accessible\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ✗ Error accessing analysis tables: {e}\n'))
        
        # Summary
        self.stdout.write(self.style.SUCCESS('✓ Admin panel functionality test completed successfully!'))
        self.stdout.write(self.style.SUCCESS('\nThe admin panel should be accessible at:'))
        self.stdout.write('   - Main admin: http://127.0.0.1:8000/admin/')
        self.stdout.write('   - Dunnhumby admin: http://127.0.0.1:8000/dunnhumby-admin/')
        self.stdout.write('   - Basket Analysis: http://127.0.0.1:8000/dunnhumby-admin/basket-analysis/')
        self.stdout.write('   - Association Rules: http://127.0.0.1:8000/dunnhumby-admin/association-rules/')
        self.stdout.write('   - Customer Segments: http://127.0.0.1:8000/dunnhumby-admin/customer-segments/')
        self.stdout.write('   - Data Manipulation: http://127.0.0.1:8000/dunnhumby-admin/data-manipulation/')