"""
Base Time Series Models for Market Basket Prediction
Supports LSTM, GRU, TFT, and N-BEATS architectures
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import pickle
import os


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models
    All models must implement fit, predict, and save/load methods
    """

    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False

    @abstractmethod
    def fit(self, X, y, prev_predictions=None, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the model

        Args:
            X (ndarray or DataFrame): Input features
            y (ndarray or Series): Target values
            prev_predictions (ndarray, optional): Predictions from previous stage models
            validation_split (float): Fraction of data for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training

        Returns:
            dict: Training history with losses
        """
        pass

    @abstractmethod
    def predict(self, X, prev_predictions=None):
        """
        Make predictions

        Args:
            X (ndarray or DataFrame): Input features
            prev_predictions (ndarray, optional): Predictions from previous stage models

        Returns:
            ndarray: Predictions
        """
        pass

    @abstractmethod
    def get_trainable_params(self):
        """
        Get trainable parameters for gradient updates

        Returns:
            list: Trainable parameters
        """
        pass

    @abstractmethod
    def update_weights(self, gradients):
        """
        Update model weights with gradients from loss propagation

        Args:
            gradients (list): Gradients to apply
        """
        pass

    def save(self, filepath):
        """Save model to disk"""
        pass

    def load(self, filepath):
        """Load model from disk"""
        pass

    def _prepare_features(self, X, prev_predictions):
        """
        Prepare features by combining current features with previous predictions

        Args:
            X (ndarray or DataFrame): Current features
            prev_predictions (ndarray): Previous stage predictions

        Returns:
            ndarray: Combined features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if prev_predictions is not None and len(prev_predictions) > 0:
            # Stack previous predictions as additional features
            prev_pred_features = np.column_stack(prev_predictions)
            X_combined = np.hstack([X, prev_pred_features])
        else:
            X_combined = X

        return X_combined


class LSTMModel(BaseTimeSeriesModel):
    """
    Long Short-Term Memory (LSTM) model
    Good for learning long-term dependencies in sequences
    """

    def __init__(self, name="LSTM"):
        super().__init__(name)
        self.input_dim = None
        self.optimizer = None

    def fit(self, X, y, prev_predictions=None, validation_split=0.2, epochs=50, batch_size=32):
        """Train LSTM model"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X_prepared = self._prepare_features(X, prev_predictions)

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_prepared)
        else:
            X_scaled = self.scaler.transform(X_prepared)

        # Reshape for LSTM: (samples, timesteps, features)
        # For product-level prediction, we treat each feature as a timestep
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Build model if not exists
        if self.model is None:
            self.input_dim = X_reshaped.shape[1]
            self.model = Sequential([
                LSTM(128, activation='relu', return_sequences=True, input_shape=(self.input_dim, 1)),
                Dropout(0.2),
                LSTM(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            self.optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        # Train model
        history = self.model.fit(
            X_reshaped,
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        self.is_trained = True

        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', [0] * len(history.history['loss']))
        }

    def predict(self, X, prev_predictions=None):
        """Make predictions with LSTM"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Prepare and scale features
        X_prepared = self._prepare_features(X, prev_predictions)
        X_scaled = self.scaler.transform(X_prepared)

        # Reshape for LSTM
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Predict
        predictions = self.model.predict(X_reshaped, verbose=0)

        return predictions.flatten()

    def get_trainable_params(self):
        """Get LSTM trainable parameters"""
        if self.model is not None:
            return self.model.trainable_weights
        return []

    def update_weights(self, gradients):
        """Update LSTM weights with gradients"""
        import tensorflow as tf

        if self.model is not None and len(gradients) > 0:
            trainable_vars = self.model.trainable_weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def save(self, filepath):
        """Save LSTM model"""
        if self.model is not None:
            self.model.save(filepath + '_model.h5')
            with open(filepath + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

    def load(self, filepath):
        """Load LSTM model"""
        from tensorflow.keras.models import load_model

        self.model = load_model(filepath + '_model.h5')
        with open(filepath + '_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True


class GRUModel(BaseTimeSeriesModel):
    """
    Gated Recurrent Unit (GRU) model
    Faster alternative to LSTM with similar performance
    """

    def __init__(self, name="GRU"):
        super().__init__(name)
        self.input_dim = None
        self.optimizer = None

    def fit(self, X, y, prev_predictions=None, validation_split=0.2, epochs=50, batch_size=32):
        """Train GRU model"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X_prepared = self._prepare_features(X, prev_predictions)

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_prepared)
        else:
            X_scaled = self.scaler.transform(X_prepared)

        # Reshape for GRU
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Build model if not exists
        if self.model is None:
            self.input_dim = X_reshaped.shape[1]
            self.model = Sequential([
                GRU(128, activation='relu', return_sequences=True, input_shape=(self.input_dim, 1)),
                Dropout(0.2),
                GRU(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            self.optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        # Train model
        history = self.model.fit(
            X_reshaped,
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        self.is_trained = True

        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', [0] * len(history.history['loss']))
        }

    def predict(self, X, prev_predictions=None):
        """Make predictions with GRU"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Prepare and scale features
        X_prepared = self._prepare_features(X, prev_predictions)
        X_scaled = self.scaler.transform(X_prepared)

        # Reshape for GRU
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Predict
        predictions = self.model.predict(X_reshaped, verbose=0)

        return predictions.flatten()

    def get_trainable_params(self):
        """Get GRU trainable parameters"""
        if self.model is not None:
            return self.model.trainable_weights
        return []

    def update_weights(self, gradients):
        """Update GRU weights with gradients"""
        import tensorflow as tf

        if self.model is not None and len(gradients) > 0:
            trainable_vars = self.model.trainable_weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def save(self, filepath):
        """Save GRU model"""
        if self.model is not None:
            self.model.save(filepath + '_model.h5')
            with open(filepath + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

    def load(self, filepath):
        """Load GRU model"""
        from tensorflow.keras.models import load_model

        self.model = load_model(filepath + '_model.h5')
        with open(filepath + '_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True


class TFTModel(BaseTimeSeriesModel):
    """
    Temporal Fusion Transformer (TFT) model
    State-of-the-art for multi-horizon forecasting with attention mechanism
    """

    def __init__(self, name="TFT"):
        super().__init__(name)
        # TFT requires pytorch-forecasting which has complex setup
        # Placeholder implementation - can be expanded

    def fit(self, X, y, prev_predictions=None, validation_split=0.2, epochs=50, batch_size=32):
        """Train TFT model (simplified implementation)"""
        # For now, use a simple feedforward network as placeholder
        # Full TFT implementation requires complex temporal data formatting
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import StandardScaler

        X_prepared = self._prepare_features(X, prev_predictions)

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_prepared)
        else:
            X_scaled = self.scaler.transform(X_prepared)

        if self.model is None:
            self.model = Sequential([
                Dense(256, activation='relu', input_shape=(X_scaled.shape[1],)),
                LayerNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                LayerNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dense(1)
            ])

            self.optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        history = self.model.fit(
            X_scaled,
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        self.is_trained = True

        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', [0] * len(history.history['loss']))
        }

    def predict(self, X, prev_predictions=None):
        """Make predictions with TFT"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_prepared = self._prepare_features(X, prev_predictions)
        X_scaled = self.scaler.transform(X_prepared)

        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()

    def get_trainable_params(self):
        """Get TFT trainable parameters"""
        if self.model is not None:
            return self.model.trainable_weights
        return []

    def update_weights(self, gradients):
        """Update TFT weights with gradients"""
        if self.model is not None and len(gradients) > 0:
            trainable_vars = self.model.trainable_weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def save(self, filepath):
        """Save TFT model"""
        if self.model is not None:
            self.model.save(filepath + '_model.h5')
            with open(filepath + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

    def load(self, filepath):
        """Load TFT model"""
        from tensorflow.keras.models import load_model

        self.model = load_model(filepath + '_model.h5')
        with open(filepath + '_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True


class NBEATSModel(BaseTimeSeriesModel):
    """
    N-BEATS (Neural Basis Expansion Analysis for Time Series)
    Specialized architecture for univariate time series forecasting
    """

    def __init__(self, name="NBEATS"):
        super().__init__(name)

    def fit(self, X, y, prev_predictions=None, validation_split=0.2, epochs=50, batch_size=32):
        """Train N-BEATS model (simplified implementation)"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import StandardScaler

        X_prepared = self._prepare_features(X, prev_predictions)

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_prepared)
        else:
            X_scaled = self.scaler.transform(X_prepared)

        if self.model is None:
            # Simplified N-BEATS architecture (stack of blocks)
            self.model = Sequential([
                Dense(512, activation='relu', input_shape=(X_scaled.shape[1],)),
                Dropout(0.2),
                Dense(512, activation='relu'),
                Dropout(0.2),
                Dense(256, activation='relu'),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dense(1)
            ])

            self.optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        history = self.model.fit(
            X_scaled,
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        self.is_trained = True

        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', [0] * len(history.history['loss']))
        }

    def predict(self, X, prev_predictions=None):
        """Make predictions with N-BEATS"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_prepared = self._prepare_features(X, prev_predictions)
        X_scaled = self.scaler.transform(X_prepared)

        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()

    def get_trainable_params(self):
        """Get N-BEATS trainable parameters"""
        if self.model is not None:
            return self.model.trainable_weights
        return []

    def update_weights(self, gradients):
        """Update N-BEATS weights with gradients"""
        if self.model is not None and len(gradients) > 0:
            trainable_vars = self.model.trainable_weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def save(self, filepath):
        """Save N-BEATS model"""
        if self.model is not None:
            self.model.save(filepath + '_model.h5')
            with open(filepath + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

    def load(self, filepath):
        """Load N-BEATS model"""
        from tensorflow.keras.models import load_model

        self.model = load_model(filepath + '_model.h5')
        with open(filepath + '_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True


def create_model(model_type, name=None):
    """
    Factory function to create time series models

    Args:
        model_type (str): Type of model ('lstm', 'gru', 'tft', 'nbeats')
        name (str, optional): Custom name for the model

    Returns:
        BaseTimeSeriesModel: Instantiated model

    Raises:
        ValueError: If model_type is not recognized
    """
    model_map = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'tft': TFTModel,
        'nbeats': NBEATSModel
    }

    model_type_lower = model_type.lower()
    if model_type_lower not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_map.keys())}")

    model_class = model_map[model_type_lower]
    return model_class(name if name else model_type.upper())
