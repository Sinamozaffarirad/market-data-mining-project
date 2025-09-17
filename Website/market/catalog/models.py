# catalog/models.py
from decimal import Decimal

from django.db import models
from django.db.models import F, Sum

class Product(models.Model):
    sku   = models.CharField(max_length=32, unique=True)
    name  = models.CharField(max_length=120)
    price = models.DecimalField(max_digits=9, decimal_places=2)

    def __str__(self):
        return f"{self.name} ({self.sku})"

class Order(models.Model):
    customer = models.ForeignKey(
        "customers.CustomerProfile",
        on_delete=models.CASCADE,
        related_name='orders'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def total(self):
        return (
            self.items.aggregate(total=Sum(F("qty") * F("price")))
            ["total"]
            or Decimal("0")
        )

class OrderItem(models.Model):
    order    = models.ForeignKey(Order, related_name="items",
                                   on_delete=models.CASCADE)
    product  = models.ForeignKey(Product, on_delete=models.PROTECT)
    qty      = models.PositiveSmallIntegerField()
    price    = models.DecimalField(max_digits=9, decimal_places=2)

    @property
    def subtotal(self):
        return self.qty * self.price
