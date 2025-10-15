"""
Orientation estimation for robotic grasping.

Provides methods to calculate optimal grasp angle from segmentation masks,
allowing the robotic claw to rotate (yaw) to align with elongated objects.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


def calculate_grasp_angle(mask: np.ndarray, method: str = "minrect") -> Dict[str, Any]:
    """
    Calculate optimal grasp angle and parameters from a segmentation mask.
    
    Args:
        mask: Binary segmentation mask (HxW numpy array, values 0-255)
        method: Algorithm to use:
            - "minrect": Minimum area rotated rectangle (recommended)
            - "pca": Principal Component Analysis
            - "moments": Image moments-based orientation
    
    Returns:
        Dictionary with:
            - success: bool, whether calculation succeeded
            - yaw_angle: float, rotation angle in degrees [-90, 90]
                        0° = horizontal, 90° = vertical
            - confidence: float, how confident we are about orientation (0-1)
            - grasp_width_px: estimated object width at grasp point (pixels)
            - center: (x, y) centroid
            - error: str (only if success=False)
    """
    if method == "minrect":
        return _method_minrect(mask)
    elif method == "pca":
        return _method_pca(mask)
    elif method == "moments":
        return _method_moments(mask)
    else:
        return {"success": False, "error": f"Unknown method: {method}"}


def _method_minrect(mask: np.ndarray) -> Dict[str, Any]:
    """
    Use minimum area rotated rectangle to find orientation.
    
    Best for: Most objects, especially elongated ones (spoons, bottles, phones).
    Pros: Fast, robust, gives grasp width estimate.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"success": False, "error": "No contours found in mask"}
    
    # Get largest contour (should be the centered object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 5:
        return {"success": False, "error": "Insufficient contour points"}
    
    # Get minimum area rectangle
    # Returns: ((center_x, center_y), (width, height), angle)
    # angle: rotation angle in degrees, [-90, 0) range
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (width, height), angle = rect
    
    # OpenCV's minAreaRect returns angle in range [-90, 0)
    # and width/height may be swapped depending on orientation
    
    # Determine grasp orientation:
    # Strategy: Grasp along the shorter dimension for elongated objects
    if width < height:
        # Object is taller than wide (vertical)
        # Grasp horizontally across the narrow width
        grasp_angle = angle  # Already perpendicular to long axis
        grasp_width = width
    else:
        # Object is wider than tall (horizontal)
        # Grasp vertically across the narrow height
        grasp_angle = angle + 90
        grasp_width = height
    
    # Normalize angle to [-90, 90] range
    # 0° = horizontal grasp, 90° = vertical grasp
    while grasp_angle > 90:
        grasp_angle -= 180
    while grasp_angle < -90:
        grasp_angle += 180
    
    # Calculate confidence based on aspect ratio
    # Higher aspect ratio = more elongated = clearer orientation
    aspect_ratio = max(width, height) / max(min(width, height), 1e-6)
    
    # Confidence mapping:
    # aspect_ratio < 1.3: circular/square -> low confidence (0.1-0.3)
    # aspect_ratio 1.3-2.0: somewhat elongated -> medium confidence (0.3-0.7)
    # aspect_ratio > 2.0: very elongated -> high confidence (0.7-1.0)
    if aspect_ratio < 1.3:
        confidence = 0.1 + (aspect_ratio - 1.0) * 0.2 / 0.3  # 0.1-0.3
    elif aspect_ratio < 2.0:
        confidence = 0.3 + (aspect_ratio - 1.3) * 0.4 / 0.7  # 0.3-0.7
    else:
        confidence = min(0.7 + (aspect_ratio - 2.0) * 0.15, 1.0)  # 0.7-1.0
    
    return {
        "success": True,
        "yaw_angle": float(grasp_angle),
        "confidence": float(confidence),
        "grasp_width_px": float(grasp_width),
        "center": (float(cx), float(cy)),
        "aspect_ratio": float(aspect_ratio),
        "bbox_dims": (float(width), float(height))
    }


def _method_pca(mask: np.ndarray) -> Dict[str, Any]:
    """
    Use Principal Component Analysis to find object's major axis.
    
    Best for: Irregular shapes, complex objects.
    Pros: Works well with non-rectangular objects.
    """
    # Get all non-zero points from mask
    points = cv2.findNonZero(mask)
    
    if points is None or len(points) < 5:
        return {"success": False, "error": "Insufficient mask points"}
    
    # Reshape to (N, 2) array
    points = points.reshape(-1, 2).astype(np.float32)
    
    # Calculate mean (centroid)
    mean = np.mean(points, axis=0)
    
    # Center the data
    centered = points - mean
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Principal component (direction of maximum variance)
    idx = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, idx]
    
    # Calculate angle from horizontal axis
    angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
    
    # Normalize to [-90, 90]
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    
    # Confidence from eigenvalue ratio (elongation measure)
    elongation = np.sqrt(eigenvalues[idx] / max(eigenvalues[1-idx], 1e-6))
    
    if elongation < 1.3:
        confidence = 0.2
    elif elongation < 2.0:
        confidence = 0.3 + (elongation - 1.3) * 0.4 / 0.7
    else:
        confidence = min(0.7 + (elongation - 2.0) * 0.15, 1.0)
    
    return {
        "success": True,
        "yaw_angle": float(angle),
        "confidence": float(confidence),
        "grasp_width_px": None,  # Not available from PCA
        "center": (float(mean[0]), float(mean[1])),
        "elongation": float(elongation)
    }


def _method_moments(mask: np.ndarray) -> Dict[str, Any]:
    """
    Use image moments to calculate orientation.
    
    Best for: Fast approximation, simple objects.
    Pros: Very fast, built-in OpenCV function.
    """
    # Calculate moments
    M = cv2.moments(mask)
    
    if M["m00"] == 0:
        return {"success": False, "error": "Empty mask (zero area)"}
    
    # Centroid
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Central moments (normalized)
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    
    # Orientation angle
    # Formula: θ = 0.5 * arctan2(2*μ11, μ20 - μ02)
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle_rad)
    
    # Eccentricity as confidence measure
    denominator = mu20 + mu02
    if denominator > 1e-6:
        eccentricity = np.sqrt(1 - min(mu20, mu02) / max(mu20, mu02, 1e-6))
        confidence = min(eccentricity * 1.5, 1.0)  # Scale to 0-1
    else:
        confidence = 0.1
    
    return {
        "success": True,
        "yaw_angle": float(angle_deg),
        "confidence": float(confidence),
        "grasp_width_px": None,
        "center": (float(cx), float(cy))
    }


def visualize_orientation(frame: np.ndarray, mask: np.ndarray, 
                          orientation: Dict[str, Any]) -> np.ndarray:
    """
    Draw orientation visualization on frame for debugging.
    
    Args:
        frame: RGB/BGR image
        mask: Binary mask
        orientation: Result from calculate_grasp_angle()
    
    Returns:
        Frame with orientation overlay
    """
    if not orientation.get("success"):
        return frame
    
    frame = frame.copy()
    cx, cy = orientation["center"]
    cx, cy = int(cx), int(cy)
    angle = orientation["yaw_angle"]
    confidence = orientation.get("confidence", 0)
    
    # Draw center point
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    
    # Draw orientation line (100px long)
    length = 100
    angle_rad = np.radians(angle)
    x2 = int(cx + length * np.cos(angle_rad))
    y2 = int(cy + length * np.sin(angle_rad))
    
    # Color based on confidence: red (low) -> yellow -> green (high)
    if confidence < 0.5:
        color = (0, int(confidence * 510), 255)  # Red -> Yellow
    else:
        color = (0, 255, int((1 - confidence) * 510))  # Yellow -> Green
    
    cv2.arrowedLine(frame, (cx, cy), (x2, y2), color, 3, tipLength=0.3)
    
    # Draw text info
    text = f"Yaw: {angle:.1f}deg (conf: {confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 255), 2)
    
    return frame


# Example usage function for testing
def test_orientation_on_detection(box_xyxy, mask_tensor, frame_shape: Tuple[int, int]):
    """
    Example function showing how to integrate with YOLO detections.
    
    Args:
        box_xyxy: Bounding box from YOLO
        mask_tensor: Segmentation mask tensor from YOLO
        frame_shape: (height, width) of the original frame
    
    Returns:
        Orientation dict
    """
    if mask_tensor is None:
        return {"success": False, "error": "No segmentation mask available"}
    
    # Convert mask tensor to binary numpy array
    h, w = frame_shape
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask_resized * 255).astype(np.uint8)
    
    # Calculate orientation
    result = calculate_grasp_angle(mask_binary, method="minrect")
    
    return result

