from django.urls import path
from . import views

app_name = 'customer_retention'

urlpatterns = [
    path('', views.retention_dashboard, name='dashboard'),
]
