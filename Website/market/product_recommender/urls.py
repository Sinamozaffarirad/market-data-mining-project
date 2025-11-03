from django.urls import path
from . import views

app_name = 'product_recommender'

urlpatterns = [
    path('', views.recommend_customers_view, name='recommend_customers'),
]
