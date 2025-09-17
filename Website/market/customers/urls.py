# customers/urls.py
from django.urls import path
from . import views

app_name = "customers"

urlpatterns = [
    # The 'name' has been changed from 'customer_search' to 'search'
    path('search/', views.customer_search, name='search'),
    path('detail/<int:pk>/', views.customer_detail, name='detail'),
]