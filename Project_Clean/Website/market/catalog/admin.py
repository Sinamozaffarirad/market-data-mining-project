from django.contrib import admin
from .models import Product, Order, OrderItem

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("sku", "name", "price")
    search_fields = ("sku", "name")

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("id", "customer", "created_at", "total")
    # The problematic filter has been removed.
    list_filter = ("created_at",)
    # A search field is added to allow searching by the customer's household key.
    search_fields = ("customer__household_key",)
    date_hierarchy = "created_at"

@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ("order", "product", "qty", "price", "subtotal")
    search_fields = ("order__id", "product__name")