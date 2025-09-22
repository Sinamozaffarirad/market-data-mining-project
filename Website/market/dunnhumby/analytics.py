"""
Advanced analytics module for Dunnhumby data analysis
Includes association rules, RFM analysis, and market basket analysis
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from django.db import connection
from .models import Transaction, Household, DunnhumbyProduct, AssociationRule, CustomerSegment, BasketAnalysis



class AssociationRulesMiner:
    """
    Advanced Association Rules Mining using Apriori algorithm
    """
    
    def __init__(self, min_support=0.01, min_confidence=0.5, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.frequent_itemsets = {}
        self.association_rules = []
        self.transaction_data = None
        
    def load_transaction_data(self, limit=None):
        """Load transaction data from database"""
        if limit:
            query = f"""
            SELECT TOP {limit} t.basket_id, t.product_id, p.commodity_desc, p.department
            FROM transactions t
            LEFT JOIN product p ON t.product_id = p.product_id
            """
        else:
            query = """
            SELECT t.basket_id, t.product_id, p.commodity_desc, p.department
            FROM transactions t
            LEFT JOIN product p ON t.product_id = p.product_id
            """
            
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            
        # Group by basket
        baskets = defaultdict(list)
        for basket_id, product_id, commodity_desc, department in results:
            item = commodity_desc or f"Product_{product_id}"
            baskets[basket_id].append(item)
            
        self.transaction_data = list(baskets.values())
        return len(self.transaction_data)
    
    def get_item_support(self, itemset):
        """Calculate support for an itemset"""
        count = sum(1 for basket in self.transaction_data 
                   if all(item in basket for item in itemset))
        return count / len(self.transaction_data)
    
    def find_frequent_1_itemsets(self):
        """Find frequent 1-itemsets"""
        item_counts = Counter()
        for basket in self.transaction_data:
            for item in basket:
                item_counts[item] += 1
        
        total_transactions = len(self.transaction_data)
        frequent_items = {}
        
        for item, count in item_counts.items():
            support = count / total_transactions
            if support >= self.min_support:
                frequent_items[frozenset([item])] = support
                
        return frequent_items
    
    def apriori_gen(self, frequent_k_minus_1):
        """Generate candidate k-itemsets from frequent (k-1)-itemsets"""
        candidates = set()
        items = list(frequent_k_minus_1.keys())
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                # Join step
                union = items[i] | items[j]
                if len(union) == len(items[i]) + 1:
                    # Prune step
                    valid = True
                    for subset in combinations(union, len(union) - 1):
                        if frozenset(subset) not in frequent_k_minus_1:
                            valid = False
                            break
                    if valid:
                        candidates.add(union)
        
        return candidates
    
    def find_frequent_itemsets(self):
        """Find all frequent itemsets using Apriori algorithm"""
        # Find frequent 1-itemsets
        self.frequent_itemsets[1] = self.find_frequent_1_itemsets()
        
        k = 2
        while self.frequent_itemsets.get(k-1):
            candidates = self.apriori_gen(self.frequent_itemsets[k-1])
            frequent_k = {}
            
            for candidate in candidates:
                support = self.get_item_support(candidate)
                if support >= self.min_support:
                    frequent_k[candidate] = support
            
            if frequent_k:
                self.frequent_itemsets[k] = frequent_k
                k += 1
            else:
                break
                
        return self.frequent_itemsets
    
    def generate_rules(self):
        """Generate association rules from frequent itemsets"""
        rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset, support in self.frequent_itemsets[k].items():
                # Generate all possible rules
                for r in range(1, len(itemset)):
                    for antecedent in combinations(itemset, r):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        antecedent_support = self.frequent_itemsets[len(antecedent)].get(antecedent, 0)
                        if antecedent_support > 0:
                            confidence = support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_support = 0
                                if len(consequent) == 1:
                                    consequent_support = self.frequent_itemsets[1].get(consequent, 0)
                                else:
                                    consequent_support = self.frequent_itemsets.get(len(consequent), {}).get(consequent, 0)
                                
                                lift = confidence / consequent_support if consequent_support > 0 else 0
                                
                                if lift >= self.min_lift:
                                    rules.append({
                                        'antecedent': list(antecedent),
                                        'consequent': list(consequent),
                                        'support': support,
                                        'confidence': confidence,
                                        'lift': lift
                                    })
        
        self.association_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
        return self.association_rules
    
    def save_rules_to_db(self, rule_type='product'):
        """Save association rules to database"""
        # Clear existing rules of this type
        AssociationRule.objects.filter(rule_type=rule_type).delete()
        
        for rule in self.association_rules:
            AssociationRule.objects.create(
                antecedent=rule['antecedent'],
                consequent=rule['consequent'],
                support=rule['support'],
                confidence=rule['confidence'],
                lift=rule['lift'],
                rule_type=rule_type,
                min_support_threshold=self.min_support,
                min_confidence_threshold=self.min_confidence,
                min_lift_threshold=self.min_lift,
                source_view=rule.get('source_view', 'analysis.miner'),
                metadata=rule.get('metadata', {})
            )
    
    def get_top_rules(self, n=20):
        """Get top N rules by lift"""
        return self.association_rules[:n]


class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) Analysis for customer segmentation
    """
    
    def __init__(self):
        self.rfm_data = None
        self.segments = None
        
    def calculate_rfm_scores(self, quantiles=5):
        """Calculate RFM scores for all customers"""
        query = """
        SELECT 
            household_key,
            MAX(day) as last_transaction_day,
            COUNT(DISTINCT basket_id) as frequency,
            SUM(sales_value) as monetary
        FROM transactions
        GROUP BY household_key
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(results, columns=['customer_id', 'recency', 'frequency', 'monetary'])
        
		# Convert monetary from Decimal to float for calculations
        df['monetary'] = df['monetary'].astype(float) # <-- ADD THIS LINE
        
        # Calculate recency (days since last purchase, assuming max day is reference)
        max_day = df['recency'].max()
        df['recency'] = max_day - df['recency']
        
        # Calculate RFM scores (1-5 scale)
        df['R'] = pd.qcut(df['recency'], quantiles, labels=[5,4,3,2,1])  # Lower recency = higher score
        df['F'] = pd.qcut(df['frequency'].rank(method='first'), quantiles, labels=[1,2,3,4,5])  # Higher frequency = higher score
        df['M'] = pd.qcut(df['monetary'], quantiles, labels=[1,2,3,4,5])  # Higher monetary = higher score
        
        # Convert to numeric
        df['R'] = df['R'].astype(int)
        df['F'] = df['F'].astype(int)
        df['M'] = df['M'].astype(int)
        
        self.rfm_data = df
        return df
    
    def segment_customers(self):
        """Segment customers based on RFM scores"""
        if self.rfm_data is None:
            self.calculate_rfm_scores()
        
        df = self.rfm_data.copy()
        
        # Create RFM segments based on scores
        def assign_segment(row):
            r, f, m = row['R'], row['F'], row['M']
            
            # Champions: High value, high frequency, recent
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            
            # Loyal Customers: High frequency, good monetary
            elif f >= 4 and m >= 3:
                return "Loyal Customers"
            
            # Potential Loyalists: Recent customers with good frequency
            elif r >= 4 and f >= 3:
                return "Potential Loyalists"
            
            # New Customers: Recent but low frequency/monetary
            elif r >= 4 and f <= 2:
                return "New Customers"
            
            # Big Spenders: High monetary regardless of recency/frequency
            elif m >= 4:
                return "Big Spenders"
            
            # Regular Customers: Consistent but not exceptional
            elif f >= 3 and r >= 3:
                return "Regular Customers"
            
            # Need Attention: Good customers who haven't purchased recently
            elif r <= 2 and f >= 3 and m >= 3:
                return "Need Attention"
            
            # At Risk: Were good customers but haven't purchased recently
            elif r <= 2 and f >= 2 and m >= 2:
                return "At Risk"
            
            # Can't Lose: High value customers who haven't purchased recently
            elif r <= 2 and f >= 4 and m >= 4:
                return "Can't Lose Them"
            
            # Hibernating: Low recency, were customers before
            elif r <= 2:
                return "Hibernating"
            
            # Lost: Lowest recency, frequency, and monetary
            else:
                return "Lost"
        
        df['Segment'] = df.apply(assign_segment, axis=1)
        self.segments = df
        return df
    
    def save_segments_to_db(self):
        """Save customer segments to database"""
        if self.segments is None:
            self.segment_customers()
        
        # Clear existing segments
        CustomerSegment.objects.all().delete()
        
        for _, row in self.segments.iterrows():
            CustomerSegment.objects.create(
                household_key=row['customer_id'],
                recency_score=row['R'],
                frequency_score=row['F'],
                monetary_score=row['M'],
                rfm_segment=row['Segment'],
                last_transaction_day=row['recency'],
                total_transactions=row['frequency'],
                total_spend=row['monetary'],
                avg_basket_value=row['monetary'] / row['frequency'] if row['frequency'] > 0 else 0
            )
    
    def get_segment_summary(self):
        """Get summary statistics for each segment"""
        if self.segments is None:
            self.segment_customers()
        
        summary = self.segments.groupby('Segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['mean', 'sum']
        }).round(2)
        
        summary.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue']
        return summary.reset_index()


class MarketBasketAnalyzer:
    """
    Comprehensive market basket analysis
    """
    
    def __init__(self):
        self.basket_data = None
        
    def analyze_baskets(self):
        """Analyze shopping baskets"""
        query = """
        SELECT 
            t.basket_id,
            t.household_key,
            COUNT(t.product_id) as total_items,
            SUM(t.sales_value) as total_value,
            COUNT(DISTINCT p.department) as unique_departments,
            STRING_AGG(DISTINCT p.department, ',') as departments
        FROM transactions t
        LEFT JOIN product p ON t.product_id = p.product_id
        GROUP BY t.basket_id, t.household_key
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=[
            'basket_id', 'household_key', 'total_items', 'total_value',
            'unique_departments', 'departments'
        ])
        
        self.basket_data = df
        return df
    
    def get_basket_statistics(self):
        """Get comprehensive basket statistics"""
        if self.basket_data is None:
            self.analyze_baskets()
        
        stats = {
            'total_baskets': len(self.basket_data),
            'avg_basket_size': self.basket_data['total_items'].mean(),
            'avg_basket_value': self.basket_data['total_value'].mean(),
            'avg_departments_per_basket': self.basket_data['unique_departments'].mean(),
            'max_basket_value': self.basket_data['total_value'].max(),
            'max_basket_size': self.basket_data['total_items'].max(),
        }
        
        return stats
    
    def save_basket_analysis(self):
        """Save basket analysis to database"""
        if self.basket_data is None:
            self.analyze_baskets()
        
        # Clear existing analysis
        BasketAnalysis.objects.all().delete()
        
        for _, row in self.basket_data.iterrows():
            dept_mix = {}
            if row['departments']:
                departments = row['departments'].split(',')
                for dept in departments:
                    dept_mix[dept.strip()] = dept_mix.get(dept.strip(), 0) + 1
            
            BasketAnalysis.objects.create(
                basket_id=row['basket_id'],
                household_key=row['household_key'],
                total_items=row['total_items'],
                total_value=row['total_value'],
                department_mix=dept_mix
            )


def run_complete_analysis(transaction_limit=None):
    """
    Run complete analysis pipeline
    """
    results = {}

    print("Starting Association Rules Mining...")
    arm = AssociationRulesMiner(min_support=0.0001, min_confidence=0.3)  # Lowered support for large dataset
    transactions_loaded = arm.load_transaction_data(limit=transaction_limit)
    results['transactions_loaded'] = transactions_loaded
    
    if transactions_loaded > 0:
        arm.find_frequent_itemsets()
        rules = arm.generate_rules()
        arm.save_rules_to_db()
        results['association_rules_found'] = len(rules)
    
    print("Starting RFM Analysis...")
    rfm = RFMAnalyzer()
    rfm_data = rfm.calculate_rfm_scores()
    segments = rfm.segment_customers()
    rfm.save_segments_to_db()
    results['customers_segmented'] = len(segments)
    
    print("Starting Market Basket Analysis...")
    mba = MarketBasketAnalyzer()
    basket_data = mba.analyze_baskets()
    mba.save_basket_analysis()
    results['baskets_analyzed'] = len(basket_data)
    
    return results


def build_churn_feature_set(prediction_point_offset=30):
    """
    یک مجموعه ویژگی جامع برای پیش‌بینی ریزش مشتری بدون نشت داده ایجاد می‌کند.
    ویژگی‌ها بر اساس یک نقطه زمانی در گذشته محاسبه شده و برچسب ریزش بر اساس
    رفتار مشتری در آینده تعیین می‌شود.
    """
    print("🚀 Starting CORRECTED churn feature engineering process (time-aware)...")

    # --- ۱. بارگذاری داده و تعیین پنجره‌های زمانی ---
    print("  - Step 1: Loading data and setting time windows...")
    transactions_df = pd.DataFrame(list(Transaction.objects.all().values()))
    households_df = pd.DataFrame(list(Household.objects.all().values()))

    if transactions_df.empty:
        print("Error: No transaction data found.")
        return pd.DataFrame()

    # تعیین "امروز" و "نقطه پیش‌بینی" در گذشته
    last_day_in_data = transactions_df['day'].max()
    prediction_date = last_day_in_data - prediction_point_offset

    # تقسیم داده به "تاریخچه" (برای ساخت ویژگی) و "آینده" (برای برچسب‌گذاری)
    history_df = transactions_df[transactions_df['day'] <= prediction_date]
    future_df = transactions_df[transactions_df['day'] > prediction_date]

    print(f"  - Data available until day: {last_day_in_data}")
    print(f"  - Building features based on data up to day: {prediction_date}")
    print(f"  - Labeling churn based on activity after day: {prediction_date}")


    # --- ۲. محاسبه ویژگی‌ها بر اساس داده‌های تاریخی ---
    print("  - Step 2: Calculating features from HISTORICAL data...")

    if history_df.empty:
        print("Error: Not enough historical data to build features.")
        return pd.DataFrame()
        
    # محاسبه RFM بر اساس تاریخچه
    customer_features = history_df.groupby('household_key').agg(
        recency=('day', lambda date: (prediction_date - date.max())), # Recency نسبت به نقطه پیش‌بینی
        frequency=('day', 'nunique'),
        monetary=('sales_value', 'sum')
    ).reset_index()

    # محاسبه ویژگی‌های رفتاری بر اساس تاریخچه
    temp_df = history_df[['household_key', 'day']].drop_duplicates().sort_values(['household_key', 'day'])
    temp_df['purchase_gap'] = temp_df.groupby('household_key')['day'].diff()
    avg_purchase_gap = temp_df.groupby('household_key')['purchase_gap'].mean().reset_index()
    avg_purchase_gap.rename(columns={'purchase_gap': 'avg_purchase_gap'}, inplace=True)

    product_variety = history_df.groupby('household_key')['product_id'].nunique().reset_index()
    product_variety.rename(columns={'product_id': 'product_variety'}, inplace=True)


    # --- ۳. ساخت برچسب Churn بر اساس داده‌های آینده ---
    print("  - Step 3: Creating churn label from FUTURE data...")
    
    # مشتریانی که در دوره آینده خرید کرده‌اند را پیدا می‌کنیم
    customers_who_returned = future_df['household_key'].unique()
    
    # برچسب Churn حالا به رفتار آینده بستگی دارد، نه Recency گذشته
    customer_features['is_churn'] = 1 # فرض می‌کنیم همه ریزش کرده‌اند
    customer_features.loc[customer_features['household_key'].isin(customers_who_returned), 'is_churn'] = 0 # آنهایی که برگشتند، ریزش نکرده‌اند


    # --- ۴. ترکیب تمام ویژگی‌ها ---
    print("  - Step 4: Merging all features...")
    df = pd.merge(customer_features, avg_purchase_gap, on='household_key', how='left')
    df = pd.merge(df, product_variety, on='household_key', how='left')
    df = pd.merge(df, households_df, on='household_key', how='inner')

    df.fillna(0, inplace=True)

    print("✅ Time-aware feature engineering complete!")
    return df