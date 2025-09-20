# customers/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from .models import User, CustomerProfile, Household

@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    list_display = ("username", "email", "is_staff", "is_active")
    search_fields = ("username", "email")

@admin.register(CustomerProfile)
class CustomerProfileAdmin(admin.ModelAdmin):
    # The fields now correctly reference the related Household model
    list_display = ("user", "household", "get_age_desc", "get_income_desc")
    search_fields = ("user__username", "household__household_key")
    list_select_related = ('household',) # Improves performance

    @admin.display(description='Age Description', ordering='household__age_desc')
    def get_age_desc(self, obj):
        if obj.household:
            return obj.household.age_desc
        return None

    @admin.display(description='Income Description', ordering='household__income_desc')
    def get_income_desc(self, obj):
        if obj.household:
            return obj.household.income_desc
        return None

@admin.register(Household)
class HouseholdAdmin(admin.ModelAdmin):
    list_display = ('household_key', 'age_desc', 'income_desc', 'homeowner_desc')
    search_fields = ('household_key',)
