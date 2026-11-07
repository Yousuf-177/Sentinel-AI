"""
SENTINEL-AI — SHAP Explainer Module

Generates human-readable SHAP-based explanations for model predictions.
"""

import logging

import numpy as np
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wraps a SHAP explainer for the CNN-LSTM model.

    Tries DeepExplainer first. TensorFlow 2 often rejects LSTM layers that
    use recurrent dropout, so this falls back to GradientExplainer and then
    KernelExplainer.

    The ranked class is the class model.predict chose, not the class with
    the largest summed SHAP magnitude.

    Usage:
        explainer = SHAPExplainer(model, background_data)
        explanation = explainer.explain_alert(flow_features, feature_names)
        explanations = explainer.explain_batch(flows, feature_names)
    """

    def __init__(self, model, background_data: np.ndarray, max_background: int = 100):
        """
        Args:
            model: Trained Keras model.
            background_data: Sample of scaled training/test rows used as the
                SHAP background. Must already be on the same scale as live inputs.
            max_background: Max number of background samples (controls speed).
        """
        background_data = np.asarray(background_data, dtype=np.float32)
        if background_data.ndim == 1:
            background_data = background_data.reshape(1, -1)
        if len(background_data) > max_background:
            indices = np.random.default_rng(42).choice(
                len(background_data), max_background, replace=False
            )
            background_data = background_data[indices]

        self.model = model
        self.background = np.ascontiguousarray(background_data)
        self.background_mean = self.background.mean(axis=0)
        self.max_background = int(len(self.background))
        self.explainer, self.explainer_name = self._build_explainer()
        logger.info(
            f"SHAP {self.explainer_name} initialized with {len(self.background)} background samples"
        )

    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Run the model and return class probabilities with shape (n, n_classes)."""
        rows = np.asarray(features, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        proba = self.model.predict(rows, verbose=0)
        return np.asarray(proba)

    def _build_explainer(self):
        """Try DeepExplainer, then GradientExplainer, then KernelExplainer."""
        errors = []

        for name, factory in (
            ("DeepExplainer", lambda: shap.DeepExplainer(self.model, self.background)),
            ("GradientExplainer", lambda: shap.GradientExplainer(self.model, self.background)),
        ):
            try:
                explainer = factory()
                probe = explainer.shap_values(self.background[:1])
                if probe is None:
                    raise RuntimeError(f"{name} returned no values")
                logger.info(f"SHAP backend selected: {name}")
                return explainer, name
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.warning(f"SHAP {name} unavailable ({exc}). Trying the next backend.")

        try:
            explainer = shap.KernelExplainer(self._predict, self.background)
            logger.info("SHAP backend selected: KernelExplainer")
            return explainer, "KernelExplainer"
        except Exception as exc:
            errors.append(f"KernelExplainer: {exc}")
            detail = " | ".join(errors)
            raise RuntimeError(f"No SHAP backend could be created. {detail}") from exc

    def _raw_shap_values(self, flows: np.ndarray):
        """Compute SHAP values for a 2-D batch in one explainer call."""
        if self.explainer_name == "KernelExplainer":
            # 2 * features + 1 is the usual KernelExplainer budget. Capping it
            # keeps a 78-feature multiclass call inside a demo latency budget.
            nsamples = min(200, 2 * flows.shape[1] + 1)
            return self.explainer.shap_values(flows, nsamples=nsamples, silent=True)
        return self.explainer.shap_values(flows)

    def _class_shap(self, shap_values, pred_classes: np.ndarray) -> np.ndarray:
        """
        Pick, for each row, the SHAP vector of the predicted class.

        Returns an array of shape (n_samples, n_features).
        """
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        n_samples = len(pred_classes)

        if isinstance(shap_values, list):
            stacked = [np.asarray(block) for block in shap_values]
            # One (n, features) block per class, or a single block.
            if len(stacked) == 1 and stacked[0].ndim >= 2 and stacked[0].shape[0] == n_samples:
                return self._class_shap(stacked[0], pred_classes)
            rows = []
            for i, class_id in enumerate(pred_classes):
                block = stacked[int(class_id)]
                rows.append(np.asarray(block[i]).reshape(-1))
            return np.vstack(rows)

        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # (n, features, classes)
            if arr.shape[0] == n_samples and arr.shape[-1] <= 32 and arr.shape[-1] < arr.shape[1]:
                return np.vstack([arr[i, :, int(pred_classes[i])] for i in range(n_samples)])
            # (n, classes, features)
            if arr.shape[0] == n_samples and arr.shape[1] <= 32:
                return np.vstack([arr[i, int(pred_classes[i]), :] for i in range(n_samples)])
            # (classes, n, features)
            if arr.shape[1] == n_samples and arr.shape[0] <= 32:
                return np.vstack([arr[int(pred_classes[i]), i, :] for i in range(n_samples)])

        raise ValueError(f"Unrecognised SHAP value layout: {getattr(arr, 'shape', type(shap_values))}")

    def _format_row(
        self,
        flow_row: np.ndarray,
        class_shap: np.ndarray,
        feature_names: list,
        top_k: int,
    ) -> dict:
        """Turn one SHAP vector into the alert explanation dict."""
        importance = np.abs(np.asarray(class_shap, dtype=np.float64).reshape(-1))
        k = min(top_k, importance.size)
        top_indices = np.argsort(importance)[-k:][::-1]

        top_features = []
        for idx in top_indices:
            baseline = float(self.background_mean[idx])
            pct_change = ((float(flow_row[idx]) - baseline) / (abs(baseline) + 1e-8)) * 100
            shap_value = float(class_shap[idx])
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            top_features.append({
                "feature": name,
                "shap_value": shap_value,
                "direction": "positive" if shap_value > 0 else "negative",
                "pct_vs_baseline": f"{pct_change:+.0f}%",
            })

        human_readable = "; ".join(
            f"{feat['feature']}: {feat['pct_vs_baseline']} vs baseline" for feat in top_features
        )
        return {"top_features": top_features, "human_readable": human_readable}

    def explain_batch(self, flows: np.ndarray, feature_names: list, top_k: int = 3) -> list:
        """
        Generate explanations for a batch of flows in one SHAP call.

        Args:
            flows: Array of shape (n_samples, n_features), already scaled.
            feature_names: Feature names in the same column order as ``flows``.
            top_k: Number of top contributing features to return.

        Returns:
            List of explanation dicts, one per row.
        """
        flows = np.asarray(flows, dtype=np.float32)
        if flows.ndim == 1:
            flows = flows.reshape(1, -1)

        proba = self._predict(flows)
        pred_classes = np.argmax(proba, axis=1).astype(int)
        class_shap = self._class_shap(self._raw_shap_values(flows), pred_classes)

        return [
            self._format_row(flows[i], class_shap[i], feature_names, top_k)
            for i in range(len(flows))
        ]

    def explain_alert(self, flow_features: np.ndarray, feature_names: list, top_k: int = 3) -> dict:
        """
        Generate a SHAP explanation for a single flow.

        Args:
            flow_features: Feature vector for one flow (1D array), already scaled.
            feature_names: List of feature names.
            top_k: Number of top contributing features to return.

        Returns:
            dict with 'top_features' list and 'human_readable' string.
        """
        return self.explain_batch(flow_features, feature_names, top_k=top_k)[0]
