import pandas as pd
from django.core.management.base import BaseCommand
from dunnhumby.ml_models import ChurnPredictor
from dunnhumby.models import CustomerSegment
from dunnhumby.analytics import build_churn_feature_set

class Command(BaseCommand):
    help = 'Trains the churn model and updates the churn probability for all customer segments.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 Starting Churn Probability Update Pipeline..."))

        # --- 1. Train the optimal model ---
        self.stdout.write(self.style.HTTP_INFO("Step 1: Training the optimal churn model (threshold=13 days)..."))
        predictor = ChurnPredictor()
        
        if not predictor.prepare_data(churn_threshold_days=13):
            self.stdout.write(self.style.ERROR("Model training failed during data preparation. Aborting."))
            return
        predictor.train_model()
        self.stdout.write(self.style.SUCCESS("Optimal model trained successfully."))

        # --- 2. Prepare feature set for ALL customers ---
        self.stdout.write(self.style.HTTP_INFO("Step 2: Preparing feature set for ALL customers..."))
        all_customer_features_df = build_churn_feature_set(prediction_point_offset=13)
        
        if all_customer_features_df.empty:
            self.stdout.write(self.style.ERROR("Failed to build feature set for all customers. Aborting."))
            return
        
        customer_ids = all_customer_features_df['household_key']
        features_for_prediction = all_customer_features_df.drop(columns=['is_churn', 'household_key'])
        
        features_for_prediction = pd.get_dummies(features_for_prediction)

        # --- 3. Predict churn probabilities ---
        self.stdout.write(self.style.HTTP_INFO("Step 3: Predicting churn probabilities for all customers..."))
        probabilities = predictor.predict_probabilities(features_for_prediction)

        if probabilities is None:
            self.stdout.write(self.style.ERROR("Prediction failed. Aborting."))
            return
            
        results_df = pd.DataFrame({
            'household_key': customer_ids,
            'churn_probability': probabilities
        })
        self.stdout.write(self.style.SUCCESS(f"Successfully calculated probabilities for {len(results_df)} customers."))

        # --- 4. Update the database ---
        self.stdout.write(self.style.HTTP_INFO("Step 4: Updating database with churn scores..."))
        
        segments_to_update = []
        customer_segments = CustomerSegment.objects.filter(household_key__in=results_df['household_key'].tolist())
        
        prob_map = {row.household_key: row.churn_probability for row in results_df.itertuples()}
        
        for segment in customer_segments:
            # CORRECTED LINE: Changed segment.household_key.pk to just segment.household_key
            segment.churn_probability = prob_map.get(segment.household_key, 0.0)
            segments_to_update.append(segment)
            
        CustomerSegment.objects.bulk_update(segments_to_update, ['churn_probability'])
        
        self.stdout.write(self.style.SUCCESS(f"\n✅ Finished! Successfully updated churn probabilities for {len(segments_to_update)} customer segments."))