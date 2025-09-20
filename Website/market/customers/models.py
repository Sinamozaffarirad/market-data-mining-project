# customers/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    # This model will store user login information (username, password, etc.)
    pass

class Household(models.Model):
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
        managed = False  # Django will not manage this table's schema
        db_table = 'household'
        verbose_name = 'Household Demographic'
        verbose_name_plural = 'Household Demographics'

    def __str__(self):
        return f"Household {self.household_key}"

class CustomerProfile(models.Model):
    """
    This model extends the Django User model and links it to a Household.
    It stores application-specific data.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # This correctly links the profile to the household data.
    # It can be null if a user doesn't have an associated household yet.
    household = models.ForeignKey(Household, on_delete=models.SET_NULL, null=True, blank=True, db_column='household_key')
    joined_at = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    churn_score = models.FloatField(default=0.0)
    spend_90d = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    def __str__(self):
        return self.user.username

