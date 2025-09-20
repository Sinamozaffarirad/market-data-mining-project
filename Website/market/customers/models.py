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