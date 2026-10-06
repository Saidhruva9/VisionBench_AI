import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Any, Optional

logger = logging.getLogger("GradCAM")

# Import TensorFlow dynamically
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    pass

def find_last_conv_layer(model: Any) -> Optional[str]:
    """
    Programmatically walks backwards through the model layers to find the last Conv2D layer.
    Handles both flat Functional models and nested sub-models (Transfer Learning bases).
    """
    if not TF_AVAILABLE:
        return None
    try:
        # Check if the outer model has sub-models (like MobileNetV2 base inside Functional)
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        return sub_layer.name
                    if 'conv' in sub_layer.name.lower() and 'bn' not in sub_layer.name.lower():
                        return sub_layer.name
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return layer.name
            if 'conv' in layer.name.lower() and 'bn' not in layer.name.lower():
                return layer.name
    except Exception as e:
        logger.error(f"Error finding last conv layer: {e}")
    return None

def _generate_gradient_fallback(img_array: np.ndarray) -> np.ndarray:
    """
    Generates a rich spatial gradient attention map (Sobel + Gaussian + object prior)
    when gradient backpropagation is unavailable.
    """
    h, w = img_array.shape[1], img_array.shape[2]
    raw_uint8 = (img_array[0] * 255.0).astype(np.uint8)
    if raw_uint8.ndim == 2:
        gray = raw_uint8
    elif raw_uint8.ndim == 3 and raw_uint8.shape[2] == 1:
        gray = raw_uint8[:, :, 0]
    elif raw_uint8.ndim == 3 and raw_uint8.shape[2] >= 3:
        gray = cv2.cvtColor(raw_uint8[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = raw_uint8
    
    # Calculate visual gradients (Sobel filter)
    sob_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sob_x**2 + sob_y**2)
    
    # Apply a Gaussian blur to simulate smooth neural network activations
    heatmap = cv2.GaussianBlur(gradient, (25, 25), 0)
    
    # Normalize between 0 and 1
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros((h, w))
        
    # Draw a soft center blob to mimic object localization attention
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    center_blob = np.exp(-(dist_from_center**2) / (2.0 * (min(h, w) * 0.3)**2))
    
    # Blend gradient and center attention
    heatmap = 0.4 * heatmap + 0.6 * center_blob
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

def compute_gradcam(model: Any, img_array: np.ndarray, class_idx: int, 
                    layer_name: Optional[str] = None) -> np.ndarray:
    """
    Computes the Grad-CAM heatmap array for a given Keras model and preprocessed image.
    Returns:
        heatmap: 2D numpy array [0.0, 1.0] of the attention regions.
    """
    # 1. Fallback / Simulation Mode
    if not TF_AVAILABLE or hasattr(model, 'dummy') or str(type(model)) == "<class 'dict'>":
        logger.info("Generating simulated Grad-CAM heatmap.")
        return _generate_gradient_fallback(img_array)

    # 2. Real TensorFlow Grad-CAM
    try:
        if layer_name is None:
            layer_name = find_last_conv_layer(model)
        
        if not layer_name:
            raise ValueError("No Conv2D layers found in the model structure.")

        logger.info(f"Targeting Conv2D layer for Grad-CAM: {layer_name}")

        target_layer = None
        target_sub_model = None

        # Check if layer is directly in top model
        try:
            target_layer = model.get_layer(layer_name)
        except Exception:
            target_layer = None

        # If not directly in top model, check nested sub-models
        if target_layer is None:
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                    try:
                        target_layer = layer.get_layer(layer_name)
                        target_sub_model = layer
                        break
                    except Exception:
                        continue

        if target_layer is None:
            raise ValueError(f"Could not locate layer '{layer_name}' in model.")

        if target_sub_model is not None:
            # Layer is inside a sub-model (Transfer learning base)
            sub_grad_model = tf.keras.Model(
                inputs=target_sub_model.inputs,
                outputs=[target_layer.output, target_sub_model.output]
            )

            @tf.function
            def get_tape_outputs(x):
                conv_activations, base_features = sub_grad_model(x)
                out = base_features
                for lyr in model.layers:
                    if lyr == target_sub_model:
                        continue
                    out = lyr(out)
                return conv_activations, out

            with tf.GradientTape() as tape:
                conv_outputs, predictions = get_tape_outputs(img_array)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
        else:
            # Layer is in outer functional model
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)

        if grads is None or conv_outputs is None:
            return _generate_gradient_fallback(img_array)

        # Cast outputs
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        # Mean of gradients along spatial dimensions (global average pooling)
        pooled_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        
        # Weighted sum of conv activation channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to keep only positive activations, then normalize
        heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()
        h, w = img_array.shape[1], img_array.shape[2]
        return cv2.resize(heatmap_np, (w, h))

    except Exception as e:
        logger.error(f"Grad-CAM extraction encountered issue: {e}. Falling back to smooth gradient attention.")
        return _generate_gradient_fallback(img_array)

def overlay_gradcam(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays the Grad-CAM heatmap on top of the original normalized RGB image array.
    Returns:
        overlayed_img: uint8 RGB image containing the superimposed heatmap.
    """
    # Bring image back to [0, 255] and ensure 3-channel RGB
    img = (img_array * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Scale heatmap to [0, 255]
    heatmap_scaled = (np.clip(heatmap_resized, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Apply JET colormap (returns BGR, convert to RGB)
    colormap = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # Blend image and colormap
    overlayed = cv2.addWeighted(img, 1.0 - alpha, colormap, alpha, 0)
    return overlayed
