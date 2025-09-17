# customers/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    # extend later if you need phone, loyalty tier, etc.
    pass


class CustomerProfile(models.Model):
    # The model is renamed, but the fields match the 'household' table.
    household_key = models.CharField(
        max_length=50,
        primary_key=True
    )
    age_desc = models.CharField(max_length=50, blank=True, null=True)
    marital_status_code = models.CharField(max_length=10, blank=True, null=True)
    income_desc = models.CharField(max_length=50, blank=True, null=True)
    homeowner_desc = models.CharField(max_length=50, blank=True, null=True)
    hh_comp_desc = models.CharField(max_length=50, blank=True, null=True)
    household_size_desc = models.CharField(max_length=50, blank=True, null=True)
    kid_category_desc = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        db_table = 'household'
        verbose_name = 'Customer Profile'
        verbose_name_plural = 'Customer Profiles'

    def __str__(self):
        return f"Profile for Household {self.household_key}"