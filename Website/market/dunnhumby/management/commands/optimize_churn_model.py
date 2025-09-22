import pandas as pd
from django.core.management.base import BaseCommand
from dunnhumby.ml_models import ChurnPredictor
from sklearn.metrics import classification_report

class Command(BaseCommand):
    help = 'Optimizes the churn prediction model by testing a range of thresholds and finding the best one.'

    def handle(self, *args, **options):
        """
        نقطه ورود اصلی برای اجرای اسکریپت بهینه‌سازی.
        """
        self.stdout.write(self.style.SUCCESS("🚀 Starting Churn Model Optimization Pipeline..."))

        # تعریف بازه‌ای از مقادیر آستانه برای آزمایش
        threshold_range = range(10, 31, 3) 
        
        results = []

        for threshold in threshold_range:
            self.stdout.write(self.style.HTTP_INFO(f"\n--- Testing Threshold: {threshold} days ---"))
            
            try:
                predictor = ChurnPredictor()
                
                if not predictor.prepare_data(churn_threshold_days=threshold):
                    self.stdout.write(self.style.WARNING(f"Skipping threshold {threshold} due to data preparation error."))
                    continue
                
                predictor.train_model()
                
                y_pred = predictor.model.predict(predictor.X_test)
                report = classification_report(predictor.y_test, y_pred, output_dict=True)
                
                churn_recall = report.get('1', {}).get('recall', 0) # Handling cases where a class might not be predicted
                accuracy = report.get('accuracy', 0)
                
                results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'churn_recall': churn_recall
                })
                
                self.stdout.write(self.style.SUCCESS(f"  - Accuracy: {accuracy:.2f}"))
                self.stdout.write(self.style.SUCCESS(f"  - Churn Recall: {churn_recall:.2f}  <-- Key Metric"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"An error occurred at threshold {threshold}: {e}"))

        if not results:
            self.stdout.write(self.style.ERROR("Optimization did not produce any results."))
            return

        self.stdout.write(self.style.HTTP_SUCCESS("\n\n--- Optimization Summary ---"))
        results_df = pd.DataFrame(results)
        
        best_result = results_df.loc[results_df['churn_recall'].idxmax()]
        
        self.stdout.write("Performance across different thresholds:")
        self.stdout.write(results_df.to_string(index=False))
        
        self.stdout.write(self.style.SUCCESS("\n\n--- Best Hyperparameter Found ---"))
        self.stdout.write(f"  - Optimal Threshold: {int(best_result['threshold'])} days")
        self.stdout.write(f"  - Highest Churn Recall: {best_result['churn_recall']:.2f}")
        self.stdout.write(f"  - Accuracy at this threshold: {best_result['accuracy']:.2f}")
        
        self.stdout.write(self.style.SUCCESS("\n✅ Optimization Pipeline finished successfully!"))