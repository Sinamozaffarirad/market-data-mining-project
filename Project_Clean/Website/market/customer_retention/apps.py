from django.apps import AppConfig


class CustomerRetentionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'customer_retention'
    verbose_name = "Customer Retention & Churn"
