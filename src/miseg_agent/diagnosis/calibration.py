"""Apply stored temperatures to factorized error and polarity logits."""
from __future__ import annotations
from typing import Mapping
import numpy as np

def apply_calibration(
    error_logit: np.ndarray,
    polarity_logit: np.ndarray,
    calibration: Mapping[str, float] | None,
) -> dict[str, np.ndarray]:
    calibration = dict(calibration or {})
    error_temperature = max(float(calibration.get("error_temperature", 1.0)), 0.05)
    polarity_temperature = max(float(calibration.get("polarity_temperature", 1.0)), 0.05)
    error = 1.0 / (1.0 + np.exp(-np.clip(error_logit / error_temperature, -40.0, 40.0)))
    polarity = 1.0 / (1.0 + np.exp(-np.clip(polarity_logit / polarity_temperature, -40.0, 40.0)))
    return {
        "correct": (1.0 - error).astype(np.float32),
        "false_negative": (error * polarity).astype(np.float32),
        "false_positive": (error * (1.0 - polarity)).astype(np.float32),
        "error": error.astype(np.float32),
    }
