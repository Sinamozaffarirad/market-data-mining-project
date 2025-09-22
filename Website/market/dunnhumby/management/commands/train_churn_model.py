from django.core.management.base import BaseCommand
from dunnhumby.ml_models import ChurnPredictor
import pandas as pd

class Command(BaseCommand):
    help = 'Trains and evaluates the customer churn prediction model.'

    def add_arguments(self, parser):
        # اضافه کردن یک آرگومان اختیاری برای تغییر آستانه Churn
        parser.add_argument(
            '--threshold',
            type=int,
            default=14,
            help='The number of days of inactivity to consider a customer as churned. Default is 14.'
        )

    def handle(self, *args, **options):
        """
        نقطه ورود اصلی برای اجرای دستور.
        """
        self.stdout.write(self.style.SUCCESS("🚀 Starting the Churn Prediction Pipeline..."))
        
        churn_threshold = options['threshold']
        self.stdout.write(self.style.HTTP_INFO(f"Using a churn threshold of {churn_threshold} days."))

        # ۱. ساخت یک نمونه از کلاس پیش‌بینی‌کننده
        predictor = ChurnPredictor()

        # ۲. آماده‌سازی داده‌ها
        if not predictor.prepare_data(churn_threshold_days=churn_threshold):
            self.stdout.write(self.style.ERROR("Data preparation failed. Aborting pipeline."))
            return

        # ۳. آموزش مدل
        predictor.train_model()

        # ۴. ارزیابی مدل
        self.stdout.write(self.style.HTTP_SUCCESS("\n--- Model Evaluation Results ---"))
        evaluation_results = predictor.evaluate_model()

        # ۵. نمایش اهمیت ویژگی‌ها
        self.stdout.write(self.style.HTTP_SUCCESS("\n--- Top 10 Most Important Features ---"))
        feature_importance = predictor.get_feature_importance()
        
        if feature_importance is not None:
            # برای نمایش بهتر، دیتافریم را به استرینگ تبدیل می‌کنیم
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 100):
                self.stdout.write(str(feature_importance.head(10)))
        
        self.stdout.write(self.style.SUCCESS("\n✅ Churn Prediction Pipeline finished successfully!"))