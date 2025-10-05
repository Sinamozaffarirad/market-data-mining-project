"""
Test script to demonstrate the new future prediction capability

This shows the difference between:
1. Historical validation (backtesting on known data)
2. Future prediction (actual ML-based forecasting)
"""

import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')
django.setup()

from dunnhumby.ml_models import PredictiveMarketBasketAnalyzer

def test_future_prediction():
    print("="*80)
    print("TESTING FUTURE PREDICTION CAPABILITY")
    print("="*80)

    # Create analyzer instance
    analyzer = PredictiveMarketBasketAnalyzer()

    print("\n1. Training models on historical data...")
    print("-" * 80)
    print("This trains 16 models (4 horizons × 4 model types)")
    print("Each model learns from past data and validates accuracy\n")

    # Train models
    success = analyzer.train_models(training_size=0.8)

    if not success:
        print("❌ Training failed!")
        return

    print("\n✅ Training complete!")
    print("\n2. Model Performance (Historical Accuracy):")
    print("-" * 80)

    # Show model accuracy
    for key, metrics in analyzer.model_metrics.items():
        print(f"{key:25s} - Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")

    print("\n3. Making FUTURE predictions (beyond day 711)...")
    print("-" * 80)

    # Test future predictions for different horizons
    for horizon in [1, 3, 6, 12]:
        print(f"\n🔮 Predicting {horizon} month(s) ahead using Neural Network:")
        print("-" * 60)

        predictions = analyzer.predict_future_purchases(
            model_name='neural_network',
            time_horizon=horizon,
            top_n=5
        )

        if predictions:
            print(f"Top 5 predicted departments for next {horizon} month(s):\n")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['department']:20s} - "
                      f"Confidence: {pred['confidence']:.3f} "
                      f"(ML Score: {pred['ml_prediction_score']:.3f}, "
                      f"Historical Accuracy: {pred['historical_accuracy']:.3f}) - "
                      f"Projected Revenue: ${pred['projected_revenue']:,.2f}")
        else:
            print(f"⚠️ No predictions available for {horizon} months ahead")

    print("\n" + "="*80)
    print("COMPARISON: Historical vs Future Predictions")
    print("="*80)

    print("\nHistorical Validation (what the old method does):")
    print("- Uses data from weeks 1-98")
    print("- Tests prediction on weeks 99-102 (data we HAVE)")
    print("- Measures accuracy by comparing prediction vs actual data")
    print("- Example: 74% accuracy means 74% of predictions were correct")

    print("\nFuture Prediction (what the NEW method does):")
    print("- Uses all data up to week 102 (day 711)")
    print("- Predicts what will happen in weeks 103+ (data we DON'T have)")
    print("- Cannot measure accuracy yet (future hasn't happened)")
    print("- Uses the 74% historical accuracy as confidence score")

    print("\n" + "="*80)
    print("✅ Test Complete!")
    print("="*80)

if __name__ == '__main__':
    test_future_prediction()
