# customers/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    # This model stores user login information (username, password, etc.)
    pass

class CustomerProfile(models.Model):
    """
    This model maps directly to the 'household' table in your database.
    It contains the demographic data for each household.
    """
    household_key = models.BigIntegerField(primary_key=True)
    age_desc = models.CharField(max_length=50, null=True, blank=True)
    marital_status_code = models.CharField(max_length=10, null=True, blank=True)
    income_desc = models.CharField(max_length=50, null=True, blank=True)
    homeowner_desc = models.CharField(max_length=50, null=True, blank=True)
    hh_comp_desc = models.CharField(max_length=50, null=True, blank=True)
    household_size_desc = models.CharField(max_length=50, null=True, blank=True)
    kid_category_desc = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        db_table = 'household' # Tells Django to use the 'household' table
        verbose_name = 'Customer Profile'
        verbose_name_plural = 'Customer Profiles'

    def __str__(self):
        return f"Household {self.household_key}"
    
class Transaction(models.Model):
    household_key = models.BigIntegerField()
    basket_id = models.BigIntegerField()
    day = models.IntegerField()
    product_id = models.BigIntegerField()
    quantity = models.IntegerField()
    sales_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    store_id = models.BigIntegerField(null=True, blank=True)
    retail_disc = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    trans_time = models.FloatField(null=True, blank=True)
    coupon_match_disc = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    week_no = models.IntegerField(null=True, blank=True)
    coupon_disc = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    id = models.BigAutoField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'transactions'

    def __str__(self):
        return f"Transaction {self.id} - Household {self.household_key}"


class Product(models.Model):
    product_id = models.BigIntegerField(primary_key=True)
    manufacturer = models.BigIntegerField(null=True, blank=True)
    department = models.CharField(max_length=100, null=True, blank=True)
    brand = models.CharField(max_length=100, null=True, blank=True)
    commodity_desc = models.CharField(max_length=100, null=True, blank=True)
    sub_commodity_desc = models.CharField(max_length=100, null=True, blank=True)
    curr_size_of_product = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'product'

    def __str__(self):
        return f"Product {self.product_id} - {self.brand}"
