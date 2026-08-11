# customers/management/commands/build_cf_cache.py
from django.core.management.base import BaseCommand
from customers.ml.cf_cache import build_and_save


class Command(BaseCommand):
    help = (
        "Precomputes and caches the collaborative-filtering similarity matrix "
        "used by the Hybrid Recommender, so recommendation pages load instantly "
        "instead of rebuilding the matrix from all transactions on every request."
    )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 Building CF similarity cache..."))
        shape, n_products = build_and_save()
        self.stdout.write(self.style.SUCCESS(
            f"✅ Cached. {shape[0]} households x {shape[1]} products, "
            f"{n_products} product metadata entries."
        ))