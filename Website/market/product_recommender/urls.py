# product_recommender/urls.py
from django.urls import path
from .views import recommend_home, product_detail, recommend_customers

app_name = "product_recommender"

urlpatterns = [
    path("", recommend_home, name="home"),
    path("recommend/", recommend_customers, name="recommend_customers"),
    path("product/<int:product_id>/", product_detail, name="product_detail"),
]