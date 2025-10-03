from django.urls import path
from . import views

app_name = "dunnhumby_site"

urlpatterns = [
    # Authentication
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),
    # Main pages
    path("", views.site_index, name="index"),
    path("basket-analysis/", views.basket_analysis, name="basket_analysis"),
    path("association-rules/", views.association_rules, name="association_rules"),
    path("customer-segments/", views.customer_segments, name="customer_segments"),
    path("data-management/", views.data_management, name="data_management"),
    # JSON/API endpoints used by front-end JS
    path("api/table/", views.api_get_table_data, name="api_get_table_data"),
    path("api/association-rules/insert/", views.api_insert_association_rule, name="api_insert_association_rule"),
    path("api/association-rules/department/", views.api_generate_department_rules, name="api_generate_department_rules"),
    path("api/association-rules/commodity/", views.api_generate_commodity_rules, name="api_generate_commodity_rules"),
    path("api/period-metrics/", views.api_get_period_metrics, name="api_get_period_metrics"),
    path("api/create/", views.api_create_record, name="api_create_record"),
    path("api/update/", views.api_update_record, name="api_update_record"),
    path("api/delete/", views.api_delete_record, name="api_delete_record"),
    path("api/export/", views.api_export_data, name="api_export_data"),
    path("api/schema/", views.api_table_schema, name="api_table_schema"),
    path("api/basket/", views.api_basket_details, name="api_basket_details"),
    path("api/product/", views.api_product_details, name="api_product_details"),
    path("api/household/", views.api_household_details, name="api_household_details"),
    path("api/segment/", views.api_rfm_details, name="api_rfm_details"),
    path("api/churn/", views.churn_api, name="churn_api"),
    path("api/differential/", views.api_differential_analysis, name="api_differential_analysis"),
    path("api/market-trends/", views.api_market_trends, name="api_market_trends"),
    # ML API endpoints
    path("api/ml/predictive/", views.predictive_analysis_api, name="predictive_analysis_api"),
    path("api/ml/train/", views.train_ml_models, name="train_ml_models"),
    path("api/ml/predictions/", views.get_predictions, name="get_predictions"),
    path("api/ml/recommendations/", views.get_recommendations, name="get_recommendations"),
    path("api/ml/performance/", views.get_model_performance, name="get_model_performance"),
    path("api/ml/training-status/", views.training_status_api, name="training_status_api"),
]
