# customers/urls.py
from django.urls import path
from . import views

app_name = "customers"

urlpatterns = [
    # The 'name' has been changed from 'customer_search' to 'search'
    path('search/', views.customer_search, name='search'),
    path('detail/<int:pk>/', views.customer_detail, name='detail'),
    path('detail/<int:pk>/recommendations/', views.customer_recommendations, name='recommendations'),
    path('detail/<int:pk>/churn/', views.customer_churn, name='churn'),
    path('detail/<int:pk>/purchases/', views.customer_purchases, name='purchases'),
]