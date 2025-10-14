"""
Orientation estimation for robotic grasping using PCA.

Provides a single helper to calculate the grasp yaw angle from a segmentation
mask. This is the implementation used on the laptop during normal operation.
"""

from typing import Any, Dict

import cv2
import numpy as np


def calculate_grasp_angle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculate grasp orientation using Principal Component Analysis.

    Args:
        mask: Binary segmentation mask (HxW numpy array, values 0-255)

    Returns:
        Dictionary with:
            - success: bool, whether calculation succeeded
            - yaw_angle: float, rotation angle in degrees [-90, 90]
            - confidence: float, confidence score based on elongation (0-1)
            - grasp_width_px: always None (not available from PCA)
            - center: (x, y) centroid of the mask
            - elongation: float, ratio between major/minor axes
            - error: str (only if success=False)
    """
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 5:
        return {"success": False, "error": "Insufficient mask points"}

    pts = points.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = int(np.argmax(eigenvalues))
    principal_axis = eigenvectors[:, idx]

    angle = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180

    elongation = float(np.sqrt(eigenvalues[idx] / max(eigenvalues[1 - idx], 1e-6)))
    if elongation < 1.3:
        confidence = 0.2
    elif elongation < 2.0:
        confidence = 0.3 + (elongation - 1.3) * 0.4 / 0.7
    else:
        confidence = min(0.7 + (elongation - 2.0) * 0.15, 1.0)

    return {
        "success": True,
        "yaw_angle": angle,
        "confidence": float(confidence),
        "grasp_width_px": None,
        "center": (float(mean[0]), float(mean[1])),
        "elongation": elongation,
    }
