from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Deprecated. Compare churn experiments from Customer Segments instead."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING(
            "This command is deprecated and no optimization was run.\n"
            "Use Customer Segments to compare time-window experiments by Recall, F1, and ROC-AUC."
        ))
