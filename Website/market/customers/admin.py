# customers/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from .models import User, CustomerProfile

@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    list_display = ("username", "email", "is_staff", "is_active")
    search_fields = ("username", "email")

@admin.register(CustomerProfile)
class CustomerProfileAdmin(admin.ModelAdmin):
    list_display = ("household_key", "age_desc", "income_desc", "homeowner_desc")
    search_fields = ("household_key",)