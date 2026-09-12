
from django.contrib import admin
from django.urls import path, include
from dunnhumby.admin import dunnhumby_admin_site

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dunnhumby-admin/', dunnhumby_admin_site.urls),
    path('', include('core.urls')),
    path('analysis/', include('dunnhumby.urls', namespace='dunnhumby_site')),
    path('customers/', include('customers.urls', namespace='customers')),
    path('analysis/product-recommender/', include('product_recommender.urls')),
    path('analysis/customer-retention/', include('customer_retention.urls')),

]
