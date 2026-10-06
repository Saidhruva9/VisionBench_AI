import os
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("MetricsTracker")

# Try importing psutil for RSS memory usage
try:
    import psutil
    def get_memory_usage_mb() -> float:
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
except ImportError:
    def get_memory_usage_mb() -> float:
        # Mock / Fallback if psutil is not available
        return 0.0

def get_file_size_mb(file_path: Path) -> float:
    """
    Returns the size of a file in Megabytes.
    Guarantees a minimum non-zero display representation (>= 0.01 MB) for existing non-empty models.
    """
    if not file_path.exists():
        return 0.0
    if file_path.is_file():
        size_bytes = file_path.stat().st_size
        if size_bytes == 0:
            return 0.0
        return max(0.01, float(size_bytes / (1024 * 1024)))
    elif file_path.is_dir():
        total_size = 0
        for dirpath, _, filenames in os.walk(file_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        if total_size == 0:
            return 0.0
        return max(0.01, float(total_size / (1024 * 1024)))
    return 0.0

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates Accuracy, Precision, Recall, and F1 Score (weighted).
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

def measure_inference_latency(model: Any, X_samples: np.ndarray, is_dl: bool = False, num_iterations: int = 50) -> float:
    """
    Measures the average inference latency (in milliseconds) for a single image.
    """
    logger.info(f"Measuring inference latency (is_dl={is_dl}) over {num_iterations} samples.")
    if len(X_samples) == 0:
        return 0.0
        
    latencies = []
    
    # Warm-up run
    try:
        if is_dl:
            model.predict(X_samples[:1], verbose=0)
        else:
            model.predict(X_samples[:1])
    except Exception:
        pass

    for _ in range(num_iterations):
        # Pick a random sample index
        idx = np.random.randint(0, len(X_samples))
        sample = X_samples[idx:idx+1]
        
        t0 = time.perf_counter()
        if is_dl:
            model.predict(sample, verbose=0)
        else:
            model.predict(sample)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0) # in ms

    avg_latency = float(np.mean(latencies))
    logger.info(f"Average latency calculated: {avg_latency:.3f} ms")
    return avg_latency
