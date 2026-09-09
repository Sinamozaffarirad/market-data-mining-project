from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'Project_Clean/Website/market'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')
from django.conf import settings
settings.DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
settings.PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']
import django
django.setup()
from django.test.runner import DiscoverRunner
runner = DiscoverRunner(verbosity=2, interactive=False)
sys.exit(bool(runner.run_tests(['dunnhumby', 'customers', 'catalog', 'core'])))
