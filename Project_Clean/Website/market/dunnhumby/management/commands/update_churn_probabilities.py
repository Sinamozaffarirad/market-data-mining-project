from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Deprecated. Churn scores are now managed by time-window experiments."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING(
            "This command no longer trains or overwrites churn probabilities.\n"
            "Use Customer Segments to train a time-window experiment, review its "
            "metrics, and activate the experiment you choose."
        ))
