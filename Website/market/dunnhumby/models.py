from django.db import models
from decimal import Decimal


class Transaction(models.Model):
    """Transaction data from Dunnhumby dataset"""
    id = models.BigAutoField(primary_key=True) 
    household_key = models.IntegerField()
    basket_id = models.BigIntegerField()
    day = models.IntegerField()
    product_id = models.IntegerField()
    quantity = models.IntegerField(null=True)
    sales_value = models.DecimalField(max_digits=10, decimal_places=2)
    store_id = models.IntegerField(null=True)
    retail_disc = models.DecimalField(max_digits=10, decimal_places=2)
    coupon_disc = models.DecimalField(max_digits=10, decimal_places=2)
    coupon_match_disc = models.DecimalField(max_digits=10, decimal_places=2)
    week_no = models.IntegerField(null=True)
    trans_time = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        managed = False
        db_table = 'transactions'

    def __str__(self):
        return f"Transaction {self.basket_id} - Product {self.product_id}"


class DunnhumbyProduct(models.Model):
    """Product catalog from Dunnhumby dataset"""
    product_id = models.BigIntegerField(primary_key=True)
    manufacturer = models.IntegerField()
    department = models.CharField(max_length=50)
    brand = models.CharField(max_length=50)
    commodity_desc = models.CharField(max_length=100)
    sub_commodity_desc = models.CharField(max_length=100)
    curr_size_of_product = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'product'

    def __str__(self):
        return f"{self.commodity_desc} - {self.brand}"


class Household(models.Model):
    """Customer household data from Dunnhumby dataset"""
    household_key = models.BigIntegerField(primary_key=True)
    age_desc = models.CharField(max_length=10, null=True, blank=True)
    marital_status_code = models.CharField(max_length=5, null=True, blank=True)
    income_desc = models.CharField(max_length=15, null=True, blank=True)
    homeowner_desc = models.CharField(max_length=20, null=True, blank=True)
    hh_comp_desc = models.CharField(max_length=30, null=True, blank=True)
    household_size_desc = models.CharField(max_length=5, null=True, blank=True)
    kid_category_desc = models.CharField(max_length=10, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'household'

    def __str__(self):
        return f"Household {self.household_key}"


class Campaign(models.Model):
    """Marketing campaign data from Dunnhumby dataset"""
    campaign = models.IntegerField(primary_key=True)
    description = models.CharField(max_length=10, null=True, blank=True)
    start_day = models.IntegerField(null=True, blank=True)
    end_day = models.IntegerField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'campaign'

    def __str__(self):
        return f"Campaign {self.campaign} - {self.description}"


class Coupon(models.Model):
    """Coupon data from Dunnhumby dataset"""
    coupon_upc = models.CharField(max_length=20, primary_key=True)
    product_id = models.BigIntegerField()
    campaign = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'coupon'

    def __str__(self):
        return f"Coupon {self.coupon_upc}"


class CouponRedemption(models.Model):
    """Coupon redemption tracking from Dunnhumby dataset"""
    id = models.BigAutoField(primary_key=True)
    household_key = models.BigIntegerField()
    day = models.IntegerField()
    coupon_upc = models.CharField(max_length=20)
    campaign = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'coupon_redemption'

    def __str__(self):
        return f"Redemption {self.household_key} - {self.coupon_upc}"


class CampaignMember(models.Model):
    """Campaign membership tracking from Dunnhumby dataset"""
    id = models.BigAutoField(primary_key=True)
    household_key = models.BigIntegerField()
    campaign = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'campaign_member'

    def __str__(self):
        return f"Household {self.household_key} in Campaign {self.campaign}"


class CausalData(models.Model):
    """Causal data for promotional effects from Dunnhumby dataset"""
    id = models.BigAutoField(primary_key=True)
    product_id = models.BigIntegerField()
    store_id = models.BigIntegerField()
    week_no = models.IntegerField()
    display = models.IntegerField()
    mailer = models.CharField(max_length=5)

    class Meta:
        managed = False
        db_table = 'causal_data'

    def __str__(self):
        return f"Product {self.product_id} - Store {self.store_id} - Week {self.week_no}"


# Association Rules Analysis Models
class BasketAnalysis(models.Model):
    """Model to store basket analysis results"""
    basket_id = models.BigIntegerField()
    household_key = models.BigIntegerField()
    transaction_date = models.DateField(null=True, blank=True)
    total_items = models.IntegerField()
    total_value = models.DecimalField(max_digits=10, decimal_places=2)
    department_mix = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['household_key']),
            models.Index(fields=['basket_id']),
        ]

    def __str__(self):
        return f"Basket Analysis {self.basket_id}"


class AssociationRule(models.Model):
    """Model to store association rules"""
    antecedent = models.JSONField()
    consequent = models.JSONField()
    support = models.FloatField()
    confidence = models.FloatField()
    lift = models.FloatField()
    rule_type = models.CharField(max_length=20, choices=[
        ('product', 'Product Level'),
        ('category', 'Category Level'),
        ('commodity', 'Commodity Level'),
        ('department', 'Department Level'),
    ])
    min_support_threshold = models.FloatField()
    min_confidence_threshold = models.FloatField(null=True, blank=True)
    min_lift_threshold = models.FloatField(null=True, blank=True)
    source_view = models.CharField(max_length=64, blank=True, default='')
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['support']),
            models.Index(fields=['confidence']),
            models.Index(fields=['lift']),
            models.Index(fields=['rule_type']),
        ]

    def __str__(self):
        ant_str = ', '.join(map(str, self.antecedent))
        con_str = ', '.join(map(str, self.consequent))
        return f"{ant_str} → {con_str} (conf: {self.confidence:.2f})"


class CustomerSegment(models.Model):
    """Model for RFM customer segmentation"""
    id = models.BigAutoField(primary_key=True)
    household_key = models.BigIntegerField(unique=True)
    recency_score = models.IntegerField()
    frequency_score = models.IntegerField()
    monetary_score = models.IntegerField()
    rfm_segment = models.CharField(max_length=20)
    last_transaction_day = models.IntegerField()
    total_transactions = models.IntegerField()
    total_spend = models.DecimalField(max_digits=12, decimal_places=2)
    avg_basket_value = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['rfm_segment']),
            models.Index(fields=['recency_score', 'frequency_score', 'monetary_score']),
        ]

    def __str__(self):
        return f"Customer {self.household_key} - {self.rfm_segment}"
