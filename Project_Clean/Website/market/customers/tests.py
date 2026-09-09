from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from customers.models import CustomerProfile
from dunnhumby.models import ChurnExperiment, CustomerWindowHistory


class CustomerChurnHistoryTests(TestCase):
    def test_forecast_check_uses_the_experiment_threshold(self):
        user = get_user_model().objects.create_user(
            username="admin",
            password="test-password",
        )
        household = CustomerProfile.objects.create(household_key=1830)
        experiment = ChurnExperiment.objects.create(
            method="sliding",
            observation_window_days=90,
            prediction_horizon_days=30,
            step_size_days=30,
            classification_threshold=0.30,
        )
        CustomerWindowHistory.objects.create(
            experiment=experiment,
            household_key=household.household_key,
            observation_start=1,
            observation_end=90,
            cutoff_day=90,
            label_start=91,
            label_end=120,
            recency_days=10,
            frequency=5,
            monetary=100,
            r_score=4,
            f_score=4,
            m_score=4,
            rfm_segment="Champions",
            is_churn=True,
            churn_probability=0.42,
        )

        self.client.force_login(user)
        response = self.client.get(
            reverse("customers:churn", args=[household.household_key])
        )

        self.assertEqual(response.status_code, 200)
        history_row = response.context["history"]["page"][0]
        self.assertTrue(history_row.prediction_is_correct)
        self.assertContains(response, "0.30")
