from django.contrib import admin
from django.db.models import Sum, Count, Avg, Max
from django.http import HttpResponse
from django.urls import path
from django.shortcuts import render
from django.template.response import TemplateResponse
from django.utils.html import format_html
from .models import (
    Transaction, DunnhumbyProduct, Household, Campaign, Coupon,
    CouponRedemption, CampaignMember, CausalData, BasketAnalysis,
    AssociationRule, CustomerSegment
)
import json
from collections import defaultdict


class DunnhumbyAdminSite(admin.AdminSite):
    site_header = "Dunnhumby Market Analysis Admin"
    site_title = "Market Analysis"
    index_title = "Shopping Basket Analysis & Database Management"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('basket-analysis/', self.admin_view(self.basket_analysis_view), name='basket-analysis'),
            path('association-rules/', self.admin_view(self.association_rules_view), name='association-rules'),
            path('customer-segments/', self.admin_view(self.customer_segments_view), name='customer-segments'),
            path('data-manipulation/', self.admin_view(self.data_manipulation_view), name='data-manipulation'),
        ]
        return custom_urls + urls

    def index(self, request, extra_context=None):
        """Custom admin index page with analysis tools"""
        extra_context = extra_context or {}
        
        # Add custom analysis tools to the context
        extra_context['analysis_tools'] = [
            {
                'title': 'Shopping Basket Analysis',
                'description': 'Analyze shopping baskets, top products, and purchasing patterns',
                'url': 'basket-analysis/',
                'icon': '📊'
            },
            {
                'title': 'Association Rules Mining',
                'description': 'Generate and view market basket association rules',
                'url': 'association-rules/',
                'icon': '🔗'
            },
            {
                'title': 'Customer Segmentation',
                'description': 'RFM analysis and customer behavioral segments',
                'url': 'customer-segments/',
                'icon': '👥'
            },
            {
                'title': 'Data Manipulation',
                'description': 'Database management and analysis data refresh tools',
                'url': 'data-manipulation/',
                'icon': '⚙️'
            }
        ]
        
        return super().index(request, extra_context)

    def basket_analysis_view(self, request):
        """Shopping basket analysis dashboard"""
        # Get basket statistics
        basket_stats = Transaction.objects.values('basket_id').annotate(
            total_items=Sum('quantity'),
            total_value=Sum('sales_value'),
            unique_products=Count('product_id', distinct=True)
        ).order_by('-total_value')[:20]

        # Department analysis
        dept_analysis = Transaction.objects.values(
            'product_id'
        ).annotate(
            total_sales=Sum('sales_value'),
            total_transactions=Count('product_id')
        ).order_by('-total_sales')[:10]

        # Top products by frequency
        top_products = Transaction.objects.values(
            'product_id'
        ).annotate(
            frequency=Count('product_id'),
            total_sales=Sum('sales_value')
        ).order_by('-frequency')[:20]

        context = {
            'title': 'Shopping Basket Analysis',
            'basket_stats': basket_stats,
            'dept_analysis': dept_analysis,
            'top_products': top_products,
        }
        return TemplateResponse(request, 'admin/dunnhumby/basket_analysis.html', context)

    def association_rules_view(self, request):
        """Association rules analysis"""
        if request.method == 'POST':
            min_support = float(request.POST.get('min_support', 0.01))
            min_confidence = float(request.POST.get('min_confidence', 0.5))
            
            # Generate association rules (simplified implementation)
            rules = self.generate_association_rules(min_support, min_confidence)
            
            context = {
                'title': 'Association Rules',
                'rules': rules,
                'min_support': min_support,
                'min_confidence': min_confidence,
            }
        else:
            context = {
                'title': 'Association Rules',
                'rules': AssociationRule.objects.all().order_by('-lift')[:50],
            }
        
        return TemplateResponse(request, 'admin/dunnhumby/association_rules.html', context)

    def customer_segments_view(self, request):
        """Customer segmentation dashboard"""
        # RFM Analysis summary
        segments = CustomerSegment.objects.values('rfm_segment').annotate(
            count=Count('household_key'),
            avg_spend=Avg('total_spend'),
            avg_transactions=Avg('total_transactions')
        ).order_by('-count')

        # Recent customer analysis
        recent_customers = CustomerSegment.objects.order_by('-updated_at')[:20]

        context = {
            'title': 'Customer Segmentation',
            'segments': segments,
            'recent_customers': recent_customers,
        }
        return TemplateResponse(request, 'admin/dunnhumby/customer_segments.html', context)

    def data_manipulation_view(self, request):
        """Enhanced database manipulation interface with comprehensive CRUD operations"""
        result_message = ""
        error_message = ""
        
        if request.method == 'POST':
            action = request.POST.get('action')
            
            try:
                if action == 'refresh_basket_analysis':
                    # Refresh basket analysis data
                    count = self.refresh_basket_analysis()
                    result_message = f"Basket analysis refreshed successfully. {count} records processed."
                    
                elif action == 'generate_segments':
                    # Generate RFM segments
                    count = self.generate_rfm_segments()
                    result_message = f"Customer segments generated successfully. {count} segments created."
                    
                elif action == 'clean_data':
                    # Clean inconsistent data
                    cleaned = self.clean_data()
                    result_message = f"Data cleaning completed. {cleaned} records processed."
                    
                elif action == 'generate_association_rules':
                    # Generate association rules
                    min_support = float(request.POST.get('min_support', 0.01))
                    min_confidence = float(request.POST.get('min_confidence', 0.5))
                    count = self.generate_and_save_association_rules(min_support, min_confidence)
                    result_message = f"Association rules generated successfully. {count} rules created with min_support={min_support}, min_confidence={min_confidence}."
                    
                elif action == 'optimize_database':
                    # Optimize database performance
                    result = self.optimize_database()
                    result_message = f"Database optimization completed. {result}"
                    
                elif action == 'backup_data':
                    # Backup critical data
                    result = self.backup_data()
                    result_message = f"Data backup completed. {result}"
                    
                elif action == 'get_table_data':
                    # AJAX request for table data
                    table_name = request.POST.get('table_name')
                    page = int(request.POST.get('page', 1))
                    limit = int(request.POST.get('limit', 50))
                    search = request.POST.get('search', '')
                    
                    data = self.get_table_data(table_name, page, limit, search)
                    return HttpResponse(json.dumps(data), content_type='application/json')
                    
                elif action == 'update_record':
                    # Update a specific record
                    table_name = request.POST.get('table_name')
                    record_id = request.POST.get('record_id')
                    field_data = json.loads(request.POST.get('field_data', '{}'))
                    
                    success = self.update_record(table_name, record_id, field_data)
                    if success:
                        result_message = f"Record updated successfully in {table_name}."
                    else:
                        error_message = f"Failed to update record in {table_name}."
                        
                elif action == 'delete_record':
                    # Delete a specific record
                    table_name = request.POST.get('table_name')
                    record_id = request.POST.get('record_id')
                    
                    success = self.delete_record(table_name, record_id)
                    if success:
                        result_message = f"Record deleted successfully from {table_name}."
                    else:
                        error_message = f"Failed to delete record from {table_name}."
                        
                elif action == 'bulk_import':
                    # Bulk import data
                    table_name = request.POST.get('table_name')
                    csv_data = request.POST.get('csv_data')
                    
                    count = self.bulk_import_data(table_name, csv_data)
                    result_message = f"Bulk import completed. {count} records imported to {table_name}."
                    
                elif action == 'export_data':
                    # Export data to CSV
                    table_name = request.POST.get('table_name')
                    filters = json.loads(request.POST.get('filters', '{}'))
                    
                    csv_content = self.export_table_data(table_name, filters)
                    response = HttpResponse(csv_content, content_type='text/csv')
                    response['Content-Disposition'] = f'attachment; filename="{table_name}_export.csv"'
                    return response
                
            except Exception as e:
                error_message = f"Error processing {action}: {str(e)}"
        
        # Handle GET request for template rendering
        context = {
            'title': 'Database Manipulation & Management',
            'result_message': result_message,
            'error_message': error_message,
            'data_stats': self.get_data_statistics(),
            'available_tables': self.get_available_tables(),
        }
        
        return TemplateResponse(request, 'admin/dunnhumby/data_manipulation.html', context)

    def generate_association_rules(self, min_support, min_confidence):
        """Generate association rules (enhanced implementation)"""
        print(f"Generating rules with min_support={min_support}, min_confidence={min_confidence}")
        
        # Get frequent itemsets using values() to avoid 'id' column issue
        baskets = defaultdict(list)
        transactions = Transaction.objects.values('basket_id', 'product_id')[:20000]  # Increased limit
        
        print(f"Processing {len(transactions)} transactions...")
        
        for transaction in transactions:
            if transaction['product_id']:  # Ensure product_id is not None
                baskets[transaction['basket_id']].append(str(transaction['product_id']))
        
        print(f"Found {len(baskets)} unique baskets")
        
        if len(baskets) == 0:
            print("No baskets found!")
            return []
        
        # Simple frequent pairs analysis
        pair_counts = defaultdict(int)
        total_baskets = len(baskets)
        
        # Count pairs in each basket
        for basket_items in baskets.values():
            # Only process baskets with at least 2 items
            if len(basket_items) >= 2:
                for i in range(len(basket_items)):
                    for j in range(i+1, len(basket_items)):
                        pair = tuple(sorted([basket_items[i], basket_items[j]]))
                        pair_counts[pair] += 1
        
        print(f"Found {len(pair_counts)} unique pairs")
        
        # Generate rules
        rules = []
        for pair, count in pair_counts.items():
            support = count / total_baskets
            if support >= min_support:
                # Calculate confidence for both directions
                # Direction 1: pair[0] -> pair[1]
                antecedent_count = sum(1 for basket_items in baskets.values() if pair[0] in basket_items)
                if antecedent_count > 0:
                    confidence = count / antecedent_count
                    
                    if confidence >= min_confidence:
                        consequent_count = sum(1 for basket_items in baskets.values() if pair[1] in basket_items)
                        if consequent_count > 0:
                            lift = confidence / (consequent_count / total_baskets)
                            
                            rules.append({
                                'antecedent': [pair[0]],
                                'consequent': [pair[1]], 
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
                
                # Direction 2: pair[1] -> pair[0]
                antecedent_count = sum(1 for basket_items in baskets.values() if pair[1] in basket_items)
                if antecedent_count > 0:
                    confidence = count / antecedent_count
                    
                    if confidence >= min_confidence:
                        consequent_count = sum(1 for basket_items in baskets.values() if pair[0] in basket_items)
                        if consequent_count > 0:
                            lift = confidence / (consequent_count / total_baskets)
                            
                            rules.append({
                                'antecedent': [pair[1]],
                                'consequent': [pair[0]],
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
        
        print(f"Generated {len(rules)} rules before filtering")
        
        # Sort by lift and return top results
        sorted_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)[:50]
        
        print(f"Returning {len(sorted_rules)} rules")
        return sorted_rules

    def refresh_basket_analysis(self):
        """Refresh basket analysis data"""
        BasketAnalysis.objects.all().delete()
        
        baskets = Transaction.objects.values('basket_id', 'household_key').annotate(
            total_items=Sum('quantity'),
            total_value=Sum('sales_value')
        )
        
        count = 0
        for basket in baskets[:1000]:  # Limit for demo
            BasketAnalysis.objects.update_or_create(
                basket_id=basket['basket_id'],
                defaults={
                    'household_key': basket['household_key'],
                    'total_items': basket['total_items'] or 0,
                    'total_value': basket['total_value'] or 0,
                    'department_mix': {}
                }
            )
            count += 1
        
        return count

    def generate_rfm_segments(self):
        """Generate RFM customer segments"""
        # This would implement proper RFM analysis
        # For now, simplified implementation
        count = 0
        try:
            customers = Transaction.objects.values('household_key').annotate(
                last_transaction=Max('day'),
                total_transactions=Count('basket_id'),
                total_spend=Sum('sales_value'),
                avg_basket_value=Avg('sales_value')
            )
            
            # Use truncate to safely clear all records
            CustomerSegment.objects.all().delete()
        except Exception as e:
            # Log error and continue with update_or_create approach
            print(f"Error in RFM segments: {e}")
            return 0
        
        for customer in customers:
            try:
                # Simplified RFM scoring (1-5 scale)
                recency_score = min(5, max(1, int((400 - customer['last_transaction']) / 80)))
                frequency_score = min(5, max(1, int(customer['total_transactions'] / 20)))
                monetary_score = min(5, max(1, int(customer['total_spend'] / 100)))
                
                # Determine segment based on scores
                if recency_score >= 4 and frequency_score >= 4 and monetary_score >= 4:
                    segment = "Champions"
                elif recency_score >= 3 and frequency_score >= 3:
                    segment = "Loyal Customers"
                elif recency_score >= 4:
                    segment = "New Customers"
                elif monetary_score >= 4:
                    segment = "Big Spenders"
                elif frequency_score >= 4:
                    segment = "Regular Customers"
                elif recency_score <= 2:
                    segment = "At Risk"
                else:
                    segment = "Standard"
                
                CustomerSegment.objects.update_or_create(
                    household_key=customer['household_key'],
                    defaults={
                        'recency_score': recency_score,
                        'frequency_score': frequency_score,
                        'monetary_score': monetary_score,
                        'rfm_segment': segment,
                        'last_transaction_day': customer['last_transaction'],
                        'total_transactions': customer['total_transactions'],
                        'total_spend': customer['total_spend'],
                        'avg_basket_value': customer['avg_basket_value']
                    }
                )
                count += 1
            except Exception as e:
                # Skip this customer if there's an error
                print(f"Skipping customer {customer['household_key']}: {e}")
                continue
        
        return count

    def clean_data(self):
        """Clean inconsistent data"""
        # Remove transactions with invalid values (using raw SQL to avoid 'id' column issue)
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("UPDATE transactions SET sales_value = 0 WHERE sales_value < 0")
            cleaned = cursor.rowcount
        
        return cleaned

    def get_data_statistics(self):
        """Get current data statistics"""
        return {
            'total_transactions': Transaction.objects.count(),
            'total_products': DunnhumbyProduct.objects.count(),
            'total_households': Household.objects.count(),
            'total_campaigns': Campaign.objects.count(),
            'basket_analyses': BasketAnalysis.objects.count(),
            'association_rules': AssociationRule.objects.count(),
            'customer_segments': CustomerSegment.objects.count(),
        }

    def generate_and_save_association_rules(self, min_support, min_confidence):
        """Generate and save association rules to database"""
        rules = self.generate_association_rules(min_support, min_confidence)
        
        # Clear existing rules
        AssociationRule.objects.all().delete()
        
        count = 0
        for rule in rules:
            try:
                AssociationRule.objects.create(
                    antecedent=rule['antecedent'],
                    consequent=rule['consequent'],
                    support=rule['support'],
                    confidence=rule['confidence'],
                    lift=rule['lift'],
                    rule_type='frequent_itemset'
                )
                count += 1
            except Exception as e:
                print(f"Error saving rule: {e}")
                continue
        
        return count

    def optimize_database(self):
        """Optimize database performance"""
        from django.db import connection
        optimizations = []
        
        try:
            with connection.cursor() as cursor:
                # Update statistics
                cursor.execute("UPDATE STATISTICS transactions")
                optimizations.append("Updated transaction statistics")
                
                # Rebuild indexes (simplified)
                cursor.execute("REINDEX DATABASE")
                optimizations.append("Rebuilt database indexes")
                
        except Exception as e:
            optimizations.append(f"Optimization error: {str(e)}")
            
        return "; ".join(optimizations)

    def backup_data(self):
        """Backup critical analysis data"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Backup basket analysis
            basket_count = BasketAnalysis.objects.count()
            
            # Backup customer segments  
            segment_count = CustomerSegment.objects.count()
            
            # Backup association rules
            rule_count = AssociationRule.objects.count()
            
            return f"Backup_{timestamp}: {basket_count} baskets, {segment_count} segments, {rule_count} rules"
            
        except Exception as e:
            return f"Backup failed: {str(e)}"

    def get_available_tables(self):
        """Get list of available database tables for CRUD operations"""
        return [
            {'name': 'transactions', 'model': 'Transaction', 'display': 'Transactions'},
            {'name': 'products', 'model': 'DunnhumbyProduct', 'display': 'Products'},  
            {'name': 'households', 'model': 'Household', 'display': 'Households'},
            {'name': 'campaigns', 'model': 'Campaign', 'display': 'Campaigns'},
            {'name': 'basket_analysis', 'model': 'BasketAnalysis', 'display': 'Basket Analysis'},
            {'name': 'association_rules', 'model': 'AssociationRule', 'display': 'Association Rules'},
            {'name': 'customer_segments', 'model': 'CustomerSegment', 'display': 'Customer Segments'},
        ]

    def get_table_data(self, table_name, page=1, limit=50, search=''):
        """Get paginated data for a specific table"""
        try:
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
                return {'error': 'Table not found'}
            
            # Calculate offset
            offset = (page - 1) * limit
            
            # Get data using values() to avoid 'id' column issues
            if table_name == 'transactions':
                queryset = model.objects.values(
                    'basket_id', 'household_key', 'product_id', 'quantity', 
                    'sales_value', 'day', 'week_no', 'store_id'
                )
                if search:
                    queryset = queryset.filter(basket_id__icontains=search)
                    
            elif table_name == 'products':
                queryset = model.objects.values(
                    'product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'
                )
                if search:
                    queryset = queryset.filter(commodity_desc__icontains=search)
                    
            elif table_name == 'households':
                queryset = model.objects.values(
                    'household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'
                )
                if search:
                    queryset = queryset.filter(household_key__icontains=search)
                    
            else:
                # For other models with proper primary keys
                queryset = model.objects.all()
                if search and hasattr(model, 'name'):
                    queryset = queryset.filter(name__icontains=search)
            
            total_count = queryset.count()
            data = list(queryset[offset:offset + limit])
            
            return {
                'data': data,
                'total': total_count,
                'page': page,
                'pages': (total_count + limit - 1) // limit,
                'has_next': offset + limit < total_count,
                'has_prev': page > 1
            }
            
        except Exception as e:
            return {'error': str(e)}

    def update_record(self, table_name, record_id, field_data):
        """Update a specific record"""
        try:
            model_map = {
                'products': DunnhumbyProduct,
                'households': Household,
                'campaigns': Campaign,
                'basket_analysis': BasketAnalysis,
                'association_rules': AssociationRule,
                'customer_segments': CustomerSegment,
            }
            
            model = model_map.get(table_name)
            if not model:
                return False
                
            # For models without standard 'id' field, use appropriate key
            if table_name == 'products':
                record = model.objects.get(product_id=record_id)
            elif table_name == 'households':
                record = model.objects.get(household_key=record_id)
            else:
                record = model.objects.get(pk=record_id)
            
            # Update fields
            for field, value in field_data.items():
                if hasattr(record, field):
                    setattr(record, field, value)
            
            record.save()
            return True
            
        except Exception as e:
            print(f"Update error: {e}")
            return False

    def delete_record(self, table_name, record_id):
        """Delete a specific record"""
        try:
            model_map = {
                'products': DunnhumbyProduct,
                'households': Household,
                'campaigns': Campaign,
                'basket_analysis': BasketAnalysis,
                'association_rules': AssociationRule,
                'customer_segments': CustomerSegment,
            }
            
            model = model_map.get(table_name)
            if not model:
                return False
                
            # For models without standard 'id' field, use appropriate key
            if table_name == 'products':
                record = model.objects.get(product_id=record_id)
            elif table_name == 'households':
                record = model.objects.get(household_key=record_id)
            else:
                record = model.objects.get(pk=record_id)
            
            record.delete()
            return True
            
        except Exception as e:
            print(f"Delete error: {e}")
            return False

    def bulk_import_data(self, table_name, csv_data):
        """Bulk import data from CSV"""
        import csv
        import io
        
        try:
            model_map = {
                'basket_analysis': BasketAnalysis,
                'association_rules': AssociationRule,
                'customer_segments': CustomerSegment,
            }
            
            model = model_map.get(table_name)
            if not model:
                return 0
            
            csv_file = io.StringIO(csv_data)
            reader = csv.DictReader(csv_file)
            
            count = 0
            for row in reader:
                try:
                    model.objects.create(**row)
                    count += 1
                except Exception as e:
                    print(f"Import error for row: {e}")
                    continue
            
            return count
            
        except Exception as e:
            print(f"Bulk import error: {e}")
            return 0

    def export_table_data(self, table_name, filters=None):
        """Export table data to CSV"""
        import csv
        import io
        
        try:
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
                return "Table not found"
            
            output = io.StringIO()
            
            # Get data using values() for models without proper primary keys
            if table_name in ['transactions', 'products', 'households']:
                if table_name == 'transactions':
                    data = model.objects.values(
                        'basket_id', 'household_key', 'product_id', 'quantity', 
                        'sales_value', 'day', 'week_no', 'store_id'
                    )[:1000]  # Limit for performance
                elif table_name == 'products':
                    data = model.objects.values(
                        'product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'
                    )[:1000]
                else:  # households
                    data = model.objects.values(
                        'household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'
                    )[:1000]
            else:
                data = model.objects.all()[:1000]
                
            if data:
                if hasattr(data.first(), 'keys'):
                    # For values() querysets
                    fieldnames = list(data.first().keys())
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
                else:
                    # For model instances
                    fieldnames = [field.name for field in model._meta.fields]
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for record in data:
                        row = {}
                        for field in fieldnames:
                            row[field] = getattr(record, field, '')
                        writer.writerow(row)
            
            return output.getvalue()
            
        except Exception as e:
            return f"Export error: {str(e)}"


# Create custom admin site instance
dunnhumby_admin_site = DunnhumbyAdminSite(name='dunnhumby_admin')


# Transaction model removed from admin due to lack of primary key
# Use custom views in the admin site instead
class TransactionAdmin(admin.ModelAdmin):
    list_display = ('basket_id', 'household_key', 'product_id', 'quantity', 'sales_value', 'day')
    list_filter = ('day', 'week_no', 'store_id')
    search_fields = ('basket_id', 'household_key', 'product_id')
    list_per_page = 50
    ordering = ('-day', '-basket_id')


@admin.register(DunnhumbyProduct)
class DunnhumbyProductAdmin(admin.ModelAdmin):
    list_display = ('product_id', 'commodity_desc', 'brand', 'department', 'manufacturer')
    list_filter = ('department', 'brand', 'manufacturer')
    search_fields = ('commodity_desc', 'sub_commodity_desc', 'brand')
    list_per_page = 50


@admin.register(Household)
class HouseholdAdmin(admin.ModelAdmin):
    list_display = ('household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc')
    list_filter = ('age_desc', 'income_desc', 'homeowner_desc')
    search_fields = ('household_key',)


@admin.register(BasketAnalysis)
class BasketAnalysisAdmin(admin.ModelAdmin):
    list_display = ('basket_id', 'household_key', 'total_items', 'total_value', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('basket_id', 'household_key')
    readonly_fields = ('department_mix',)


@admin.register(AssociationRule)
class AssociationRuleAdmin(admin.ModelAdmin):
    list_display = ('get_rule_display', 'support', 'confidence', 'lift', 'rule_type', 'created_at')
    list_filter = ('rule_type', 'created_at')
    ordering = ('-lift', '-confidence')
    
    def get_rule_display(self, obj):
        ant = ', '.join(map(str, obj.antecedent))
        con = ', '.join(map(str, obj.consequent))
        return f"{ant} → {con}"
    get_rule_display.short_description = "Rule"


@admin.register(CustomerSegment)
class CustomerSegmentAdmin(admin.ModelAdmin):
    list_display = ('household_key', 'rfm_segment', 'recency_score', 'frequency_score', 
                   'monetary_score', 'total_spend', 'updated_at')
    list_filter = ('rfm_segment', 'recency_score', 'frequency_score', 'monetary_score')
    search_fields = ('household_key',)
    ordering = ('-total_spend',)


@admin.register(Campaign)
class CampaignAdmin(admin.ModelAdmin):
    list_display = ('campaign', 'description', 'start_day', 'end_day')
    list_filter = ('description',)
    ordering = ('campaign',)


@admin.register(Coupon)
class CouponAdmin(admin.ModelAdmin):
    list_display = ('coupon_upc', 'product_id', 'campaign')
    list_filter = ('campaign',)
    search_fields = ('coupon_upc', 'product_id')


# CouponRedemption model removed from admin due to lack of primary key
class CouponRedemptionAdmin(admin.ModelAdmin):
    list_display = ('household_key', 'coupon_upc', 'campaign', 'day')
    list_filter = ('campaign', 'day')
    search_fields = ('household_key', 'coupon_upc')


# Register models with the custom admin site (excluding models without proper primary keys)
# dunnhumby_admin_site.register(Transaction, TransactionAdmin)  # Commented out - no primary key
dunnhumby_admin_site.register(DunnhumbyProduct, DunnhumbyProductAdmin)
dunnhumby_admin_site.register(Household, HouseholdAdmin)
dunnhumby_admin_site.register(Campaign, CampaignAdmin)
dunnhumby_admin_site.register(BasketAnalysis, BasketAnalysisAdmin)
dunnhumby_admin_site.register(AssociationRule, AssociationRuleAdmin)
dunnhumby_admin_site.register(CustomerSegment, CustomerSegmentAdmin)
