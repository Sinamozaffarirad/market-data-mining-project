from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Deprecated. Churn models are now managed by time-window experiments."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING(
            "This command is deprecated and no churn model was trained.\n"
            "Use Customer Segments to train, compare, and activate a time-window experiment."
        ))
