
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


POWER_BI_EMBED_URL = os.getenv(
    "POWER_BI_EMBED_URL",
    "https://app.powerbi.com/reportEmbed?reportId=f3cf246b-d31e-4b89-b5d2-745e1faf6304&autoAuth=true&ctid=58b8e67e-047a-4104-a2f7-a17767b20f01",
).strip()


SECRET_KEY = "django-insecure--7&=%p1gosrs=_zv(+immez89e$w*@x1lb&9rad&s_ym*qur0&"


DEBUG = True

ALLOWED_HOSTS = []



INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "core",
    "customers",
    "catalog",
    "dunnhumby",
    "customer_retention",
    "product_recommender",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "market.middleware.StripCommentsMiddleware",
]

ROOT_URLCONF = "market.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "market.wsgi.application"


DATABASES = {
    "default": {
        "ENGINE": "mssql",
        "NAME": "marketdb",
        "HOST": "localhost",
        # "HOST": "localhost\SQLEXPRESS",
        "OPTIONS": {
            "driver": "ODBC Driver 17 for SQL Server",
            "Encrypt": "no",
            "TrustServerCertificate": "yes",
            "Trusted_Connection": "yes",
        },
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

AUTH_USER_MODEL = "customers.User"


LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Tehran"

USE_I18N = True

USE_TZ = True


STATIC_URL = "static/"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


SESSION_COOKIE_AGE = 3600  
SESSION_SAVE_EVERY_REQUEST = True  
SESSION_EXPIRE_AT_BROWSER_CLOSE = (
    False  
)
SESSION_COOKIE_SECURE = False 
SESSION_COOKIE_HTTPONLY = True  
SESSION_COOKIE_SAMESITE = "Lax"  


LOGIN_URL = "/analysis/login/"
LOGIN_REDIRECT_URL = "/analysis/"

