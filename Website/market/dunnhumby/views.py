from django.contrib.auth.decorators import login_required
from django.db import models
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.db.models import Sum, Count, Avg, Max, Q
from .models import (
    Transaction, DunnhumbyProduct, Household, Campaign, Coupon,
    CouponRedemption, CampaignMember, CausalData, BasketAnalysis,
    AssociationRule, CustomerSegment
)
from .ml_models import ml_analyzer
import json
from collections import defaultdict
import threading
from django.views.decorators.csrf import csrf_exempt


def _generate_association_rules(min_support, min_confidence):
    baskets = defaultdict(list)
    transactions = Transaction.objects.values('basket_id', 'product_id')[:10000]
    for t in transactions:
        baskets[t['basket_id']].append(str(t['product_id']))

    pair_counts = defaultdict(int)
    total_baskets = len(baskets)
    for items in baskets.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = tuple(sorted([items[i], items[j]]))
                pair_counts[pair] += 1

    rules = []
    for pair, count in pair_counts.items():
        support = count / (total_baskets or 1)
        if support >= min_support:
            ant_count = sum(1 for b in baskets.values() if pair[0] in b)
            confidence = count / (ant_count or 1)
            if confidence >= min_confidence:
                cons_count = sum(1 for b in baskets.values() if pair[1] in b)
                lift = confidence / (cons_count / (total_baskets or 1) or 1)
                rules.append({
                    'antecedent': [pair[0]],
                    'consequent': [pair[1]],
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'rule_type': 'frequent_itemset',
                })
    return sorted(rules, key=lambda x: x['lift'], reverse=True)[:50]


def _get_data_statistics():
    return {
        'total_transactions': Transaction.objects.count(),
        'total_products': DunnhumbyProduct.objects.count(),
        'total_households': Household.objects.count(),
        'total_campaigns': Campaign.objects.count(),
        'basket_analyses': BasketAnalysis.objects.count(),
        'association_rules': AssociationRule.objects.count(),
        'customer_segments': CustomerSegment.objects.count(),
    }


@login_required(login_url='/admin/login/')
def site_index(request):
    tools = [
        { 'title': 'Shopping Basket Analysis', 'description': 'Analyze baskets, top products, and patterns', 'url': 'basket-analysis/', 'icon': '📊' },
        { 'title': 'Association Rules', 'description': 'Market basket association rules', 'url': 'association-rules/', 'icon': '🔗' },
        { 'title': 'Customer Segments', 'description': 'RFM segments & behavior', 'url': 'customer-segments/', 'icon': '👥' },
        { 'title': 'Data Management', 'description': 'View, edit, import/export data', 'url': 'data-management/', 'icon': '⚙️' },
    ]
    return render(request, 'site/index.html', { 'analysis_tools': tools })


@login_required(login_url='/admin/login/')
def basket_analysis(request):
    basket_stats = Transaction.objects.values('basket_id', 'household_key').annotate(
        total_items=Sum('quantity'),
        total_value=Sum('sales_value'),
        unique_products=Count('product_id', distinct=True)
    ).order_by('-total_value')[:20]

    dept_analysis = Transaction.objects.values('product_id').annotate(
        total_sales=Sum('sales_value'),
        total_transactions=Count('product_id')
    ).order_by('-total_sales')[:10]

    top_products = Transaction.objects.values('product_id').annotate(
        frequency=Count('product_id'),
        total_sales=Sum('sales_value')
    ).order_by('-frequency')[:20]

    return render(request, 'site/dunnhumby/basket_analysis.html', {
        'title': 'Shopping Basket Analysis',
        'basket_stats': basket_stats,
        'dept_analysis': dept_analysis,
        'top_products': top_products,
    })


@login_required(login_url='/admin/login/')
def association_rules(request):
    if request.method == 'POST':
        min_support = float(request.POST.get('min_support', 0.01))
        min_confidence = float(request.POST.get('min_confidence', 0.5))
        rules = _generate_association_rules(min_support, min_confidence)
        ctx = {
            'title': 'Association Rules',
            'rules': rules,
            'min_support': min_support,
            'min_confidence': min_confidence,
        }
    else:
        ctx = {
            'title': 'Association Rules',
            'rules': AssociationRule.objects.all().order_by('-lift')[:50],
        }
    return render(request, 'site/dunnhumby/association_rules.html', ctx)


@login_required(login_url='/admin/login/')
def customer_segments(request):
    segments = CustomerSegment.objects.values('rfm_segment').annotate(
        count=Count('household_key'),
        avg_spend=Avg('total_spend'),
        avg_transactions=Avg('total_transactions')
    ).order_by('-count')

    recent_customers = CustomerSegment.objects.order_by('-updated_at')[:20]

    return render(request, 'site/dunnhumby/customer_segments.html', {
        'title': 'Customer Segmentation',
        'segments': segments,
        'recent_customers': recent_customers,
    })


@login_required(login_url='/admin/login/')
def data_management(request):
    return render(request, 'site/dunnhumby/data_management.html', {
        'title': 'Database Manipulation & Management',
        'data_stats': _get_data_statistics(),
    })


@login_required(login_url='/admin/login/')
def api_get_table_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        table_name = request.POST.get('table_name')
        page = int(request.POST.get('page', 1))
        limit = int(request.POST.get('limit', 50))
        search = request.POST.get('search', '')
        filters = json.loads(request.POST.get('filters', '{}')) if request.POST.get('filters') else {}

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
            return JsonResponse({'error': 'Table not found'}, status=400)

        # Define which fields to select for each table
        field_sets = {
            'transactions': ['basket_id', 'household_key', 'product_id', 'quantity', 'sales_value', 'day', 'week_no', 'store_id'],
            'products': ['product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'],
            'households': ['household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'],
        }
        
        # Always start with a values() queryset to get dictionaries
        if table_name in field_sets:
            queryset = model.objects.values(*field_sets[table_name])
        else:
            # For other tables, get all fields as values
            queryset = model.objects.values()

        # Generic Search Logic
        searchable_fields = {
            'transactions': ['basket_id', 'household_key', 'product_id'],
            'products': ['commodity_desc', 'brand', 'department', 'manufacturer'],
            'households': ['household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'],
            'campaigns': ['description'],
            'customer_segments': ['rfm_segment', 'household_key']
        }
        if search and table_name in searchable_fields:
            q_objects = Q()
            for field in searchable_fields.get(table_name, []):
                try:
                    # Check if field type is numeric before attempting numeric search
                    is_numeric_field = 'int' in model._meta.get_field(field).get_internal_type().lower()
                    if search.isnumeric() and is_numeric_field:
                         q_objects |= Q(**{f"{field}": search})
                    else:
                         q_objects |= Q(**{f"{field}__icontains": search})
                except (AttributeError, ValueError):
                    # Fallback for non-model fields or casting issues
                    q_objects |= Q(**{f"{field}__icontains": search})
            if q_objects:
                queryset = queryset.filter(q_objects)


        # Generic Filter Logic
        filter_mappings = {
            'household_key_min': 'household_key__gte',
            'household_key_max': 'household_key__lte',
            'product_id_min': 'product_id__gte',
            'product_id_max': 'product_id__lte',
            'sales_value_min': 'sales_value__gte',
            'sales_value_max': 'sales_value__lte',
            'day_min': 'day__gte',
            'day_max': 'day__lte',
            'department': 'department__icontains',
            'brand': 'brand__icontains',
            'age_desc': 'age_desc__icontains',
            'income_desc': 'income_desc__icontains',
            'description': 'description__icontains',
            'rfm_segment': 'rfm_segment__icontains'
        }
        
        if filters:
            filter_kwargs = {}
            for key, value in filters.items():
                if key in filter_mappings and value:
                    filter_kwargs[filter_mappings[key]] = value
            if filter_kwargs:
                queryset = queryset.filter(**filter_kwargs)
        
        # Ordering
        ordering_fields = {
            'transactions': ('-day', 'basket_id'),
            'products': ('product_id',),
            'households': ('household_key',),
            'customer_segments': ('-total_spend',)
        }
        if table_name in ordering_fields:
            queryset = queryset.order_by(*ordering_fields[table_name])
        elif hasattr(model._meta, 'pk'):
             queryset = queryset.order_by(model._meta.pk.name)


        total_count = queryset.count()
        offset = (page - 1) * limit
        
        # Simple slicing now works because queryset is always a ValuesQuerySet
        data = list(queryset[offset:offset + limit])

        return JsonResponse({
            'data': data,
            'total': total_count,
            'page': page,
            'pages': (total_count + limit - 1) // limit,
            'has_next': offset + limit < total_count,
            'has_prev': page > 1
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_table_schema(request):
    """Return a schema for a table to build dynamic filters or forms client-side."""
    table = request.GET.get('table')
    purpose = request.GET.get('purpose', 'filter') # 'filter' or 'form'
    
    model_map = {
        'transactions': Transaction,
        'products': DunnhumbyProduct,
        'households': Household,
        'campaigns': Campaign,
        'basket_analysis': BasketAnalysis,
        'association_rules': AssociationRule,
        'customer_segments': CustomerSegment,
    }
    model = model_map.get(table)
    if not model:
        return JsonResponse({'error': 'table not found'}, status=404)

    def field_type(f):
        t = getattr(f, 'get_internal_type', lambda: 'TextField')()
        if t in ('IntegerField','BigIntegerField','SmallIntegerField','PositiveIntegerField','FloatField','DecimalField'): return 'number'
        return 'text'

    fields_to_include = []
    if purpose == 'form':
        # For forms, we usually want all non-pk, editable fields
        fields_to_include = [f.name for f in model._meta.fields if not f.primary_key and f.editable]
    else: # purpose == 'filter'
        filterable_fields = {
            'transactions': ['household_key', 'product_id', 'sales_value', 'day'],
            'products': ['department', 'brand'],
            'households': ['age_desc', 'income_desc'],
            'campaigns': ['description'],
            'customer_segments': ['rfm_segment']
        }
        fields_to_include = filterable_fields.get(table, [])
    
    meta = {f.name: f for f in model._meta.fields}
    fields = []
    for name in fields_to_include:
        f = meta.get(name)
        if f:
            ftype = field_type(f)
            # Use verbose_name for a user-friendly label, fallback to name
            label = getattr(f, 'verbose_name', name).title()
            fields.append({'name': name, 'type': ftype, 'label': label})
            
    return JsonResponse({'fields': fields})


@login_required(login_url='/admin/login/')
def api_basket_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    basket_id = request.POST.get('basket_id')
    if not basket_id:
        return JsonResponse({'error': 'basket_id required'}, status=400)
    try:
        qs = Transaction.objects.values('product_id','quantity','sales_value','day','store_id','household_key').filter(basket_id=basket_id).order_by('day')
        items = list(qs[:200])
        total_sales = sum(float(i.get('sales_value') or 0) for i in items)
        unique_products = list({i['product_id'] for i in items})
        prod_info = {}
        if unique_products:
            for p in DunnhumbyProduct.objects.filter(product_id__in=unique_products).values('product_id','brand','department','commodity_desc'):
                prod_info[p['product_id']] = {'brand': p['brand'], 'department': p['department'], 'commodity_desc': p['commodity_desc']}
        for i in items:
            meta = prod_info.get(i['product_id'], {})
            i['brand'] = meta.get('brand')
            i['department'] = meta.get('department')
            i['commodity_desc'] = meta.get('commodity_desc')
        resp = {
            'basket_id': basket_id,
            'total_sales': total_sales,
            'item_count': len(items),
            'unique_products': len(unique_products),
            'items': items,
        }
        return JsonResponse(resp)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_product_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    product_id = request.POST.get('product_id')
    if not product_id:
        return JsonResponse({'error': 'product_id required'}, status=400)
    try:
        agg = Transaction.objects.filter(product_id=product_id).aggregate(
            total_sales=Sum('sales_value'),
            total_txns=Count('product_id'),
            unique_households=Count('household_key', distinct=True)
        )
        top_households = list(
            Transaction.objects.filter(product_id=product_id).values('household_key').annotate(spend=Sum('sales_value')).order_by('-spend')[:5]
        )
        prod = DunnhumbyProduct.objects.filter(product_id=product_id).values('commodity_desc','brand','department','manufacturer').first() or {}
        return JsonResponse({
            'product_id': product_id,
            'meta': prod,
            'stats': {k: float(v) if v is not None and k=='total_sales' else v for k,v in agg.items()},
            'top_households': top_households,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_household_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    household_key = request.POST.get('household_key')
    if not household_key:
        return JsonResponse({'error': 'household_key required'}, status=400)
    try:
        seg = CustomerSegment.objects.filter(household_key=household_key).values(
            'rfm_segment','recency_score','frequency_score','monetary_score','total_spend','total_transactions','avg_basket_value','updated_at'
        ).first()
        recent_txns = list(Transaction.objects.filter(household_key=household_key).values(
            'basket_id','product_id','quantity','sales_value','day'
        ).order_by('-day')[:15])
        # enrich with product names
        pids = list({t['product_id'] for t in recent_txns})
        prodmap = {p['product_id']: p['commodity_desc'] for p in DunnhumbyProduct.objects.filter(product_id__in=pids).values('product_id','commodity_desc')}
        for t in recent_txns:
            t['commodity_desc'] = prodmap.get(t['product_id'])
        return JsonResponse({'household_key': household_key, 'segment': seg, 'recent_transactions': recent_txns})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_segment_details(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    name = request.POST.get('rfm_segment')
    if not name:
        return JsonResponse({'error': 'rfm_segment required'}, status=400)
    try:
        qs = CustomerSegment.objects.filter(rfm_segment=name)
        agg = qs.aggregate(
            customers=Count('household_key'),
            avg_spend=Avg('total_spend'),
            avg_txns=Avg('total_transactions'),
            avg_basket=Avg('avg_basket_value')
        )
        top_households = list(
            qs.values('household_key','total_spend','total_transactions','avg_basket_value','recency_score','frequency_score','monetary_score','updated_at')
              .order_by('-total_spend')[:12]
        )
        return JsonResponse({
            'rfm_segment': name,
            'metrics': {
                'customers': agg['customers'] or 0,
                'avg_spend': float(agg['avg_spend'] or 0),
                'avg_txns': float(agg['avg_txns'] or 0),
                'avg_basket': float(agg['avg_basket'] or 0),
            },
            'top_households': top_households,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_update_record(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    table_name = request.POST.get('table_name')
    record_id = request.POST.get('record_id')
    field_data = json.loads(request.POST.get('field_data', '{}'))

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
        return JsonResponse({'success': False, 'error': 'Unsupported table'}, status=400)

    try:
        if table_name == 'products':
            record = model.objects.get(product_id=record_id)
        elif table_name == 'households':
            record = model.objects.get(household_key=record_id)
        else:
            record = model.objects.get(pk=record_id)
        for k, v in field_data.items():
            if hasattr(record, k):
                setattr(record, k, v)
        record.save()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@login_required(login_url='/admin/login/')
def api_delete_record(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
        
    table_name = request.POST.get('table_name')
    record_id = request.POST.get('record_id')

    # Important: Do not allow deleting from core tables like transactions
    editable_tables = {
        'products': DunnhumbyProduct,
        'households': Household,
        'campaigns': Campaign,
        'basket_analysis': BasketAnalysis,
        'association_rules': AssociationRule,
        'customer_segments': CustomerSegment,
    }
    model = editable_tables.get(table_name)
    if not model:
        return JsonResponse({'success': False, 'error': 'Deletion from this table is not permitted'}, status=403)

    try:
        if table_name == 'products':
            record = model.objects.get(product_id=record_id)
        elif table_name == 'households':
            record = model.objects.get(household_key=record_id)
        else:
            record = model.objects.get(pk=record_id)
        
        record.delete()
        return JsonResponse({'success': True, 'message': 'Record deleted successfully.'})
    except model.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Record not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required(login_url='/admin/login/')
def api_export_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    import csv, io
    table_name = request.POST.get('table_name')
    filters = json.loads(request.POST.get('filters', '{}'))

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
        return JsonResponse({'error': 'Table not found'}, status=400)

    output = io.StringIO()
    writer = csv.writer(output)

    # values selection for frequently used sets
    if table_name in ('transactions', 'products', 'households'):
        if table_name == 'transactions':
            data = model.objects.values(
                'basket_id', 'household_key', 'product_id', 'quantity',
                'sales_value', 'day', 'week_no', 'store_id'
            )[:1000]
        elif table_name == 'products':
            data = model.objects.values(
                'product_id', 'commodity_desc', 'brand', 'department', 'manufacturer'
            )[:1000]
        else:
            data = model.objects.values(
                'household_key', 'age_desc', 'income_desc', 'homeowner_desc', 'hh_comp_desc'
            )[:1000]
        if not data:
            return HttpResponse('', content_type='text/csv')
        fieldnames = list(data.first().keys())
        writer.writerow(fieldnames)
        for row in data:
            writer.writerow([row.get(f, '') for f in fieldnames])
        csv_content = output.getvalue()
    else:
        qs = model.objects.all()[:1000]
        fieldnames = [f.name for f in model._meta.fields]
        writer.writerow(fieldnames)
        for rec in qs:
            writer.writerow([getattr(rec, f, '') for f in fieldnames])
        csv_content = output.getvalue()

    resp = HttpResponse(csv_content, content_type='text/csv')
    resp['Content-Disposition'] = f'attachment; filename="{table_name}_export.csv"'
    return resp


# Global variable to track training status
ml_training_status = {'is_training': False, 'progress': 0, 'message': ''}


@csrf_exempt
def predictive_analysis_api(request):
    """API endpoint for predictive market basket analysis"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        action = request.POST.get('action', 'predict')
        model_type = request.POST.get('model_type', 'neural_network')
        training_size = float(request.POST.get('training_size', 0.8))
        
        if action == 'train':
            return train_ml_models(request, model_type, training_size)
        elif action == 'predict':
            return get_predictions(request, model_type)
        elif action == 'recommendations':
            return get_recommendations(request, model_type)
        elif action == 'performance':
            return get_model_performance(request)
        else:
            return JsonResponse({'error': 'Invalid action'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def train_ml_models(request):
    """Train ML models in background"""
    global ml_training_status
    
    # Get parameters from POST data
    model_type = request.POST.get('model_type', 'neural_network')
    training_size = float(request.POST.get('training_size', 0.8))
    
    if ml_training_status['is_training']:
        return JsonResponse({
            'success': False,
            'error': 'Training already in progress',
            'status': 'training',
            'message': 'Training already in progress',
            'progress': ml_training_status['progress']
        })
    
    def train_models():
        global ml_training_status
        try:
            ml_training_status = {'is_training': True, 'progress': 10, 'message': 'Starting training...'}
            
            # Train models
            success = ml_analyzer.train_models(training_size)
            
            if success:
                ml_training_status = {'is_training': False, 'progress': 100, 'message': 'Training completed successfully!'}
            else:
                ml_training_status = {'is_training': False, 'progress': 0, 'message': 'Training failed'}
                
        except Exception as e:
            ml_training_status = {'is_training': False, 'progress': 0, 'message': f'Training error: {str(e)}'}
    
    # Start training in background thread
    thread = threading.Thread(target=train_models)
    thread.daemon = True
    thread.start()
    
    return JsonResponse({
        'success': True,
        'status': 'started',
        'message': 'Model training started in background',
        'progress': 10
    })


@csrf_exempt
def get_predictions(request):
    """Get department-level predictions"""
    # Get parameters from POST data
    model_type = request.POST.get('model_type', 'neural_network')
    try:
        predictions = ml_analyzer.get_department_predictions(model_type)
        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'predictions': predictions
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Prediction error: {str(e)}',
            'predictions': []
        })


@csrf_exempt
def get_recommendations(request):
    """Get AI-powered product recommendations"""
    # Get parameters from POST data
    model_type = request.POST.get('model_type', 'neural_network')
    top_n = int(request.POST.get('top_n', 10))
    try:
        customer_id = request.POST.get('customer_id')
        
        recommendations = ml_analyzer.predict_customer_preferences(
            model_type, customer_id, top_n
        )
        
        return JsonResponse({
            'success': True,
            'status': 'success',
            'model_type': model_type,
            'customer_id': customer_id,
            'recommendations': recommendations
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Recommendation error: {str(e)}',
            'recommendations': []
        })


@csrf_exempt
def get_model_performance(request):
    """Get model performance metrics"""
    try:
        performance = ml_analyzer.get_model_performance()
        return JsonResponse({
            'status': 'success',
            'performance': performance,
            'training_status': ml_training_status
        })
    except Exception as e:
        return JsonResponse({'error': f'Performance error: {str(e)}'}, status=500)


@csrf_exempt  
def training_status_api(request):
    """Get current training status"""
    global ml_training_status
    return JsonResponse(ml_training_status)

