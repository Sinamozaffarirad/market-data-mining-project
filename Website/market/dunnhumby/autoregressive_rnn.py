"""Small NumPy encoder-decoder RNN for recursive product-revenue forecasting.

The decoder feeds each predicted period into the next one.  Training uses a
joint masked loss across the complete available rollout and backpropagation
through time, so a later-period error updates the recurrent states and outputs
that produced earlier predictions.  Keeping the implementation in NumPy avoids
adding a large deep-learning runtime to the Django project.
"""
from __future__ import annotations

import numpy as np


class AutoregressiveRevenueRNN:
    """One-layer tanh encoder-decoder trained with Adam and full BPTT."""

    def __init__(
        self,
        hidden_size=16,
        epochs=10,
        batch_size=4096,
        learning_rate=0.008,
        huber_delta=0.75,
        gradient_clip=5.0,
        l2=1e-5,
        feedback_rate=0.5,
        random_state=42,
    ):
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.huber_delta = float(huber_delta)
        self.gradient_clip = float(gradient_clip)
        self.l2 = float(l2)
        self.feedback_rate = float(feedback_rate)
        if not 0.0 < self.feedback_rate <= 1.0:
            raise ValueError("feedback_rate must be in (0, 1].")
        self.random_state = int(random_state)
        self.input_size = 4
        self.parameters_ = None
        self.training_history_ = []
        # Size-conditional retransformation correction for the log-space
        # inverse transform; estimated on the training fold in ``fit`` and
        # applied in ``_forward``.  Defaults to the identity correction.
        self.smearing_edges_ = np.array([-np.inf, np.inf])
        self.smearing_factors_ = np.array([1.0])

    @staticmethod
    def _softplus(value):
        value = np.asarray(value, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)

    @staticmethod
    def _sigmoid(value):
        value = np.asarray(value, dtype=np.float64)
        positive = value >= 0
        result = np.empty_like(value)
        result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exp_value = np.exp(value[~positive])
        result[~positive] = exp_value / (1.0 + exp_value)
        return result

    @staticmethod
    def _scales(lags):
        lags = np.maximum(np.asarray(lags, dtype=np.float64), 0.0)
        recent = lags[:, -min(3, lags.shape[1]):].mean(axis=1)
        history = lags.mean(axis=1)
        return np.maximum(np.where(recent > 0.0, recent, history), 1e-3)

    @staticmethod
    def _period_features(period_indices):
        indices = np.asarray(period_indices, dtype=np.float64)
        angle = 2.0 * np.pi * (indices + 1.0) / 12.0
        return np.sin(angle), np.cos(angle)

    def _initialize_parameters(self, rng):
        hidden = self.hidden_size
        input_scale = np.sqrt(2.0 / (self.input_size + hidden))
        output_scale = np.sqrt(2.0 / (hidden + 1))
        recurrent = rng.normal(size=(hidden, hidden))
        q, _ = np.linalg.qr(recurrent)
        return {
            "wx": rng.normal(scale=input_scale, size=(self.input_size, hidden)),
            "wh": q * 0.75,
            "bh": np.zeros(hidden, dtype=np.float64),
            "wy": rng.normal(scale=output_scale, size=(hidden, 1)),
            "by": np.array([-0.35], dtype=np.float64),
        }

    def _normalized_inputs(self, lags, scales, start_periods):
        normalized = np.log1p(np.maximum(lags, 0.0) / scales[:, None])
        inputs = []
        for offset in range(lags.shape[1]):
            period = start_periods - lags.shape[1] + offset
            sin_period, cos_period = self._period_features(period)
            inputs.append(np.column_stack((
                normalized[:, offset],
                (lags[:, offset] > 0.0).astype(np.float64),
                sin_period,
                cos_period,
            )))
        return inputs, normalized[:, -1]

    def _forward(self, lags, start_periods, horizon, retain_cache=False):
        params = self.parameters_
        batch = len(lags)
        scales = self._scales(lags)
        encoder_inputs, feedback_value = self._normalized_inputs(
            lags, scales, start_periods
        )
        hidden = np.zeros((batch, self.hidden_size), dtype=np.float64)
        encoder_hidden = [hidden]
        for input_values in encoder_inputs:
            hidden = np.tanh(
                input_values @ params["wx"]
                + hidden @ params["wh"]
                + params["bh"]
            )
            encoder_hidden.append(hidden)

        decoder_inputs, decoder_hidden, raw_outputs, outputs = [], [hidden], [], []
        for step in range(int(horizon)):
            period = start_periods + step
            sin_period, cos_period = self._period_features(period)
            decoder_input = np.column_stack((
                feedback_value,
                (feedback_value > 1e-8).astype(np.float64),
                sin_period,
                cos_period,
            ))
            hidden = np.tanh(
                decoder_input @ params["wx"]
                + hidden @ params["wh"]
                + params["bh"]
            )
            raw_output = (hidden @ params["wy"] + params["by"]).ravel()
            output = self._softplus(raw_output)
            decoder_inputs.append(decoder_input)
            decoder_hidden.append(hidden)
            raw_outputs.append(raw_output)
            outputs.append(output)
            feedback_value = (
                self.feedback_rate * output
                + (1.0 - self.feedback_rate) * feedback_value
            )

        normalized_predictions = np.column_stack(outputs)
        if not retain_cache:
            # exp(mean of logs) is a geometric mean; the size-conditional
            # factor restores the arithmetic mean of skewed revenue.
            level = np.log(scales)[:, None] + normalized_predictions
            factor = self._apply_smearing(level)
            return np.maximum(np.exp(level) * factor - scales[:, None], 0.0)
        return normalized_predictions, {
            "scales": scales,
            "encoder_inputs": encoder_inputs,
            "encoder_hidden": encoder_hidden,
            "decoder_inputs": decoder_inputs,
            "decoder_hidden": decoder_hidden,
            "raw_outputs": raw_outputs,
        }

    def _loss_and_gradients(
        self, lags, targets, target_mask, start_periods, sample_weight
    ):
        horizon = targets.shape[1]
        predictions, cache = self._forward(
            lags, start_periods, horizon, retain_cache=True
        )
        normalized_targets = np.log1p(
            np.maximum(targets, 0.0) / cache["scales"][:, None]
        )
        mask = np.asarray(target_mask, dtype=np.float64)
        weights = np.asarray(sample_weight, dtype=np.float64)[:, None] * mask
        denominator = max(float(weights.sum()), 1.0)
        error = predictions - normalized_targets
        absolute_error = np.abs(error)
        delta = self.huber_delta
        point_loss = np.where(
            absolute_error <= delta,
            0.5 * error ** 2,
            delta * (absolute_error - 0.5 * delta),
        )
        loss = float((point_loss * weights).sum() / denominator)
        output_gradient = np.where(
            absolute_error <= delta, error, delta * np.sign(error)
        ) * weights / denominator

        params = self.parameters_
        gradients = {name: np.zeros_like(value) for name, value in params.items()}
        hidden_gradient = np.zeros_like(cache["decoder_hidden"][-1])
        feedback_gradient = np.zeros(len(lags), dtype=np.float64)

        for step in range(horizon - 1, -1, -1):
            total_output_gradient = (
                output_gradient[:, step] + self.feedback_rate * feedback_gradient
            )
            raw_gradient = total_output_gradient * self._sigmoid(
                cache["raw_outputs"][step]
            )
            current_hidden = cache["decoder_hidden"][step + 1]
            previous_hidden = cache["decoder_hidden"][step]
            gradients["wy"] += current_hidden.T @ raw_gradient[:, None]
            gradients["by"] += raw_gradient.sum(keepdims=True)
            hidden_gradient += raw_gradient[:, None] @ params["wy"].T
            activation_gradient = hidden_gradient * (1.0 - current_hidden ** 2)
            gradients["wx"] += cache["decoder_inputs"][step].T @ activation_gradient
            gradients["wh"] += previous_hidden.T @ activation_gradient
            gradients["bh"] += activation_gradient.sum(axis=0)
            input_gradient = activation_gradient @ params["wx"].T
            hidden_gradient = activation_gradient @ params["wh"].T
            feedback_gradient = (
                input_gradient[:, 0]
                + (1.0 - self.feedback_rate) * feedback_gradient
            ) if step > 0 else np.zeros(len(lags))

        for step in range(len(cache["encoder_inputs"]) - 1, -1, -1):
            current_hidden = cache["encoder_hidden"][step + 1]
            previous_hidden = cache["encoder_hidden"][step]
            activation_gradient = hidden_gradient * (1.0 - current_hidden ** 2)
            gradients["wx"] += cache["encoder_inputs"][step].T @ activation_gradient
            gradients["wh"] += previous_hidden.T @ activation_gradient
            gradients["bh"] += activation_gradient.sum(axis=0)
            hidden_gradient = activation_gradient @ params["wh"].T

        for name in ("wx", "wh", "wy"):
            loss += 0.5 * self.l2 * float((params[name] ** 2).sum())
            gradients[name] += self.l2 * params[name]
        gradient_norm = np.sqrt(sum(float((gradient ** 2).sum()) for gradient in gradients.values()))
        if gradient_norm > self.gradient_clip:
            multiplier = self.gradient_clip / gradient_norm
            gradients = {name: gradient * multiplier for name, gradient in gradients.items()}
        return loss, gradients

    def fit(self, lags, targets, target_mask, start_periods, sample_weight=None):
        lags = np.maximum(np.asarray(lags, dtype=np.float64), 0.0)
        targets = np.maximum(np.asarray(targets, dtype=np.float64), 0.0)
        target_mask = np.asarray(target_mask, dtype=bool)
        start_periods = np.asarray(start_periods, dtype=np.int64)
        if lags.ndim != 2 or targets.ndim != 2 or target_mask.shape != targets.shape:
            raise ValueError("lags, targets, and target_mask must be aligned 2-D arrays.")
        if len(lags) != len(targets) or len(start_periods) != len(lags):
            raise ValueError("Training arrays must contain the same number of samples.")
        if not target_mask.any(axis=1).all():
            raise ValueError("Every sequence sample needs at least one supervised forecast step.")
        if sample_weight is None:
            scale_weight = 1.0 + np.log1p(self._scales(lags))
            cap = np.quantile(scale_weight, 0.99)
            sample_weight = np.minimum(scale_weight, cap)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        sample_weight = sample_weight / max(float(sample_weight.mean()), 1e-12)

        rng = np.random.default_rng(self.random_state)
        self.parameters_ = self._initialize_parameters(rng)
        first_moment = {name: np.zeros_like(value) for name, value in self.parameters_.items()}
        second_moment = {name: np.zeros_like(value) for name, value in self.parameters_.items()}
        update = 0
        self.training_history_ = []
        indices = np.arange(len(lags))
        for epoch in range(self.epochs):
            rng.shuffle(indices)
            epoch_losses = []
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                loss, gradients = self._loss_and_gradients(
                    lags[batch_indices],
                    targets[batch_indices],
                    target_mask[batch_indices],
                    start_periods[batch_indices],
                    sample_weight[batch_indices],
                )
                update += 1
                epoch_losses.append(loss)
                for name, gradient in gradients.items():
                    first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                    second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient ** 2
                    corrected_first = first_moment[name] / (1.0 - 0.9 ** update)
                    corrected_second = second_moment[name] / (1.0 - 0.999 ** update)
                    self.parameters_[name] -= self.learning_rate * corrected_first / (
                        np.sqrt(corrected_second) + 1e-8
                    )
            self.training_history_.append(round(float(np.mean(epoch_losses)), 8))
        self.supervised_steps_ = int(target_mask.sum())
        self.max_supervised_horizon_ = int(target_mask.sum(axis=1).max())
        self.smearing_edges_, self.smearing_factors_ = self._estimate_smearing(
            lags, targets, target_mask, start_periods
        )
        return self

    def _apply_smearing(self, level):
        factors = np.asarray(getattr(self, "smearing_factors_", [1.0]), dtype=np.float64)
        if factors.size == 0:
            return 1.0
        edges = np.asarray(
            getattr(self, "smearing_edges_", [-np.inf, np.inf]), dtype=np.float64
        )
        index = np.clip(
            np.searchsorted(edges, level, side="right") - 1, 0, factors.size - 1
        )
        return factors[index]

    def _estimate_smearing(self, lags, targets, target_mask, start_periods, n_bins=10):
        """Solve the log-space retransformation correction on the training fold.

        The decoder is trained on ``log1p(y / scale)`` and inverted with
        ``expm1``, so exponentiating a conditional mean of logs returns a
        geometric mean and understates skewed revenue.  Duan's smearing factor
        assumes a plain ``log`` model with homoscedastic residuals; the
        ``log1p`` inverse carries a ``-1`` that breaks that decomposition, the
        Huber loss pulls toward the median, and residual spread varies with
        product size on this sparse panel.  A single global factor therefore
        removes the aggregate bias but inflates near-zero products.

        A factor is instead solved per decile of predicted level so that within
        each bin the retransformed training total matches the observed total:

            sum[exp(level) * f - scale] = sum(target),  level = log(scale) + prediction
        """
        levels, actuals, scale_values = [], [], []
        for start in range(0, len(lags), self.batch_size):
            stop = start + self.batch_size
            batch_mask = target_mask[start:stop]
            if not batch_mask.any():
                continue
            predictions, cache = self._forward(
                lags[start:stop],
                start_periods[start:stop],
                targets.shape[1],
                retain_cache=True,
            )
            scales = np.repeat(cache["scales"][:, None], targets.shape[1], axis=1)
            levels.append((np.log(scales) + predictions)[batch_mask])
            actuals.append(np.maximum(targets[start:stop], 0.0)[batch_mask])
            scale_values.append(scales[batch_mask])
        if not levels:
            return np.array([-np.inf, np.inf]), np.array([1.0])
        level = np.concatenate(levels)
        actual = np.concatenate(actuals)
        scale = np.concatenate(scale_values)
        if level.size == 0:
            return np.array([-np.inf, np.inf]), np.array([1.0])
        edges = np.unique(np.quantile(level, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([float(level.min()), float(level.max()) + 1e-9])
        factors = np.ones(edges.size - 1, dtype=np.float64)
        index = np.clip(
            np.searchsorted(edges, level, side="right") - 1, 0, edges.size - 2
        )
        for position in range(edges.size - 1):
            mask = index == position
            if not mask.any():
                continue
            denominator = float(np.exp(level[mask]).sum())
            if not np.isfinite(denominator) or denominator <= 0.0:
                continue
            factor = (float(actual[mask].sum()) + float(scale[mask].sum())) / denominator
            if np.isfinite(factor) and factor > 0.0:
                factors[position] = float(np.clip(factor, 0.25, 6.0))
        return edges, factors

    def predict(self, lags, start_period, horizon, batch_size=8192):
        if self.parameters_ is None:
            raise ValueError("The recurrent model has not been fitted.")
        lags = np.maximum(np.asarray(lags, dtype=np.float64), 0.0)
        if np.isscalar(start_period):
            start_periods = np.full(len(lags), int(start_period), dtype=np.int64)
        else:
            start_periods = np.asarray(start_period, dtype=np.int64)
        forecasts = []
        for start in range(0, len(lags), int(batch_size)):
            stop = start + int(batch_size)
            forecasts.append(self._forward(
                lags[start:stop], start_periods[start:stop], int(horizon)
            ))
        return np.vstack(forecasts) if forecasts else np.empty((0, int(horizon)))
