"""
Sliding Window Generator for Time Series Data
Generates overlapping training windows with configurable size and slide interval
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.db import connection


class SlidingWindowGenerator:
    """
    Generates sliding windows for time series training data.
    Dramatically increases training samples compared to fixed monthly windows.

    Example:
        window_size = 60 days, slide_interval = 7 days
        Window 1: Days 1-60
        Window 2: Days 8-67
        Window 3: Days 15-74
        ...
    """

    def __init__(self, window_size_days, slide_interval_days):
        """
        Initialize sliding window generator

        Args:
            window_size_days (int): Size of each window in days (e.g., 30, 60, 90)
            slide_interval_days (int): Days to slide forward for next window (e.g., 3, 7, 10)
        """
        self.window_size_days = window_size_days
        self.slide_interval_days = slide_interval_days

    def generate_windows(self, start_day, end_day):
        """
        Generate all sliding windows between start and end days

        Args:
            start_day (int): Start day number (1-711)
            end_day (int): End day number (1-711)

        Returns:
            list: List of tuples (window_start_day, window_end_day) for each window
        """
        windows = []
        current_start = start_day

        while current_start + self.window_size_days <= end_day:
            window_end = current_start + self.window_size_days
            windows.append((current_start, window_end))
            current_start += self.slide_interval_days

        return windows

    def get_window_data(self, window_start_day, window_end_day, target_horizon_days):
        """
        Fetch transaction data for a specific window and calculate features

        Args:
            window_start_day (int): Window start day (1-711)
            window_end_day (int): Window end day (1-711)
            target_horizon_days (int): Days ahead to predict (e.g., 30 for 1 month)

        Returns:
            tuple: (X_features, y_target) where:
                - X_features: DataFrame with aggregated features per product
                - y_target: Series with future purchase amounts per product
        """
        # Query to get features from window period
        feature_query = """
            SELECT
                t.product_id,
                COUNT(DISTINCT t.household_key) as unique_customers,
                COUNT(*) as transaction_count,
                SUM(t.sales_value) as total_sales,
                AVG(t.sales_value) as avg_sales,
                SUM(t.quantity) as total_quantity,
                AVG(t.quantity) as avg_quantity,
                COUNT(DISTINCT t.basket_id) as basket_count,
                STDEV(t.sales_value) as sales_std,
                MIN(t.sales_value) as min_sales,
                MAX(t.sales_value) as max_sales,
                -- Temporal features
                AVG(t.week_no) as avg_week,
                AVG(t.day) as avg_day
            FROM transactions t
            WHERE t.day >= %s AND t.day < %s
            GROUP BY t.product_id
        """

        # Query to get target values from future period
        target_start_day = window_end_day
        target_end_day = window_end_day + target_horizon_days

        target_query = """
            SELECT
                t.product_id,
                SUM(t.sales_value) as future_sales,
                SUM(t.quantity) as future_quantity,
                COUNT(*) as future_transaction_count
            FROM transactions t
            WHERE t.day >= %s AND t.day < %s
            GROUP BY t.product_id
        """

        with connection.cursor() as cursor:
            # Get features
            cursor.execute(feature_query, [window_start_day, window_end_day])
            columns = [col[0] for col in cursor.description]
            features_data = cursor.fetchall()
            X_features = pd.DataFrame(features_data, columns=columns)

            # Get targets
            cursor.execute(target_query, [target_start_day, target_end_day])
            target_columns = [col[0] for col in cursor.description]
            target_data = cursor.fetchall()
            y_target = pd.DataFrame(target_data, columns=target_columns)

        # Merge features with targets
        merged = X_features.merge(
            y_target,
            on='product_id',
            how='left'
        )

        # Fill missing targets with 0 (products not purchased in future)
        merged[['future_sales', 'future_quantity', 'future_transaction_count']] = \
            merged[['future_sales', 'future_quantity', 'future_transaction_count']].fillna(0)

        # Separate features and targets
        feature_cols = [col for col in merged.columns if not col.startswith('future_')]
        target_col = 'future_sales'  # Primary target

        X = merged[feature_cols]
        y = merged[target_col]

        return X, y

    def generate_all_window_data(self, start_day, end_day, target_horizon_days):
        """
        Generate features and targets for all windows

        Args:
            start_day (int): Start day (1-711)
            end_day (int): End day (1-711)
            target_horizon_days (int): Days ahead to predict

        Returns:
            tuple: (X_all, y_all, window_indices) where:
                - X_all: Combined features from all windows
                - y_all: Combined targets from all windows
                - window_indices: Array indicating which window each sample belongs to
        """
        windows = self.generate_windows(start_day, end_day)

        X_all = []
        y_all = []
        window_indices = []

        for idx, (window_start_day, window_end_day) in enumerate(windows):
            X, y = self.get_window_data(window_start_day, window_end_day, target_horizon_days)

            X_all.append(X)
            y_all.append(y)
            window_indices.extend([idx] * len(X))

        # Combine all windows
        X_combined = pd.concat(X_all, ignore_index=True)
        y_combined = pd.concat(y_all, ignore_index=True)
        window_indices = np.array(window_indices)

        return X_combined, y_combined, window_indices

    def get_window_count(self, start_day, end_day):
        """
        Calculate number of windows that will be generated

        Args:
            start_day (int): Start day (1-711)
            end_day (int): End day (1-711)

        Returns:
            int: Number of windows
        """
        return len(self.generate_windows(start_day, end_day))

    def get_sample_multiplier(self):
        """
        Calculate how many times more samples sliding windows provide
        compared to non-overlapping monthly windows

        Returns:
            float: Sample multiplier
        """
        # Approximate: 30 days / slide_interval
        return 30.0 / self.slide_interval_days
