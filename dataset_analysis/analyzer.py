import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
from collections import defaultdict
from config import ALLOWED_EXTENSIONS, LOGS_DIR

logger = logging.getLogger("DatasetAnalyzer")

class DatasetAnalyzer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.classes = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))]
        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self) -> List[Path]:
        paths = []
        for class_name in self.classes:
            class_dir = self.dataset_path / class_name
            for file in class_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
                    paths.append(file)
        return paths

    @staticmethod
    def calculate_image_hash(img_path: Path) -> int:
        """
        Computes Average Hash (aHash) for duplicate image detection.
        """
        try:
            with Image.open(img_path) as img:
                img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
                pixels = np.array(img)
                avg = pixels.mean()
                diff = pixels > avg
                # Convert 8x8 boolean matrix to a 64-bit integer hash
                hash_val = 0
                for i, b in enumerate(diff.flatten()):
                    if b:
                        hash_val |= (1 << i)
                return hash_val
        except Exception:
            return 0

    @staticmethod
    def analyze_single_image(img_path: Path) -> Dict[str, Any]:
        """
        Analyzes a single image for blur, noise, brightness, contrast, and basic attributes.
        """
        try:
            # OpenCV loading (headless mode compatible)
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                return {"error": "Could not read image"}

            h, w, c = img_cv.shape
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # 1. Blur Detection (Laplacian Variance)
            blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = bool(blur_var < 100.0)

            # 2. Noise Estimation (High-pass Gaussian difference standard deviation)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_est = np.std(gray.astype(np.float64) - blurred.astype(np.float64))
            is_noisy = bool(noise_est > 15.0)

            # 3. Brightness & Contrast
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            # 4. Color Channels
            # Check if image is grayscale (channels are identical)
            is_grayscale = True
            if c == 3:
                b, g, r = cv2.split(img_cv)
                if not (np.array_equal(b, g) and np.array_equal(g, r)):
                    is_grayscale = False

            # 5. Image Entropy (Shannon entropy)
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist_norm = hist / hist.sum()
            entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-7)))

            return {
                "width": w,
                "height": h,
                "aspect_ratio": float(w / h),
                "is_grayscale": is_grayscale,
                "blur_variance": float(blur_var),
                "is_blurry": is_blurry,
                "noise_level": float(noise_est),
                "is_noisy": is_noisy,
                "brightness": brightness,
                "contrast": contrast,
                "entropy": entropy,
                "channels": c,
                "error": None
            }
        except Exception as e:
            return {"error": str(e)}

    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Runs complete intelligence analysis over the whole dataset directory.
        """
        logger.info(f"Starting detailed analysis of {len(self.image_paths)} images")
        
        class_distribution = defaultdict(int)
        widths, heights, aspect_ratios = [], [], []
        grayscale_count = 0
        blurry_count = 0
        noisy_count = 0
        total_brightness = 0.0
        total_contrast = 0.0
        total_entropy = 0.0
        
        hashes = {}
        duplicates = []
        
        total_dataset_images = len(self.image_paths)
        for path in self.image_paths:
            class_name = path.parent.name
            class_distribution[class_name] += 1

        # Sample for detailed pixel metrics if dataset is large (speed & scalability)
        if total_dataset_images > 250:
            analysis_paths = list(np.random.choice(self.image_paths, 250, replace=False))
        else:
            analysis_paths = self.image_paths

        valid_img_count = 0
        for path in analysis_paths:
            # Analyze individual metrics
            metrics = self.analyze_single_image(path)
            if metrics.get("error") is not None:
                continue
                
            valid_img_count += 1
            widths.append(metrics["width"])
            heights.append(metrics["height"])
            aspect_ratios.append(metrics["aspect_ratio"])
            
            if metrics["is_grayscale"]:
                grayscale_count += 1
            if metrics["is_blurry"]:
                blurry_count += 1
            if metrics["is_noisy"]:
                noisy_count += 1
                
            total_brightness += metrics["brightness"]
            total_contrast += metrics["contrast"]
            total_entropy += metrics["entropy"]

            # Duplicate Detection
            img_hash = self.calculate_image_hash(path)
            if img_hash != 0:
                # Find near-duplicates
                is_duplicate = False
                for existing_path, existing_hash in hashes.items():
                    hamming_dist = bin(img_hash ^ existing_hash).count('1')
                    if hamming_dist <= 2:
                        duplicates.append((str(path.relative_to(self.dataset_path)), str(existing_path.relative_to(self.dataset_path))))
                        is_duplicate = True
                        break
                if not is_duplicate:
                    hashes[path] = img_hash

        if valid_img_count == 0:
            return {"error": "No readable images were found in the dataset."}

        # Class balance calculation (Shannon entropy of class sizes)
        class_sizes = list(class_distribution.values())
        total_imgs = sum(class_sizes)
        probs = [size / total_imgs for size in class_sizes]
        class_entropy = -sum(p * np.log2(p) for p in probs)
        max_possible_entropy = np.log2(len(self.classes))
        balance_index = float(class_entropy / max_possible_entropy) if max_possible_entropy > 0 else 1.0

        # Dataset Health Score Formula
        # Weights: 30% Balance, 25% Blur-free, 20% Noise-free, 15% Duplicate-free, 10% Resolution adequacy (>= 128px)
        blur_free_ratio = 1.0 - (blurry_count / valid_img_count)
        noise_free_ratio = 1.0 - (noisy_count / valid_img_count)
        duplicate_free_ratio = 1.0 - (len(duplicates) / valid_img_count)
        
        avg_res = (np.mean(widths) + np.mean(heights)) / 2.0
        res_score = min(1.0, avg_res / 224.0) # 224px is ideal standard

        health_score = int(
            (0.30 * balance_index +
             0.25 * blur_free_ratio +
             0.20 * noise_free_ratio +
             0.15 * duplicate_free_ratio +
             0.10 * res_score) * 100
        )

        # Dataset Complexity Score
        # Complexity is high if: images are high resolution, high image entropy, high contrast,
        # and high intra-class variance.
        avg_entropy = total_entropy / valid_img_count
        avg_contrast = total_contrast / valid_img_count
        entropy_factor = min(1.0, avg_entropy / 8.0)
        contrast_factor = min(1.0, avg_contrast / 128.0)
        complexity_score = int((0.4 * entropy_factor + 0.3 * contrast_factor + 0.3 * res_score) * 100)

        # Recommendation fallback logic
        health_insights = []
        if balance_index < 0.8:
            health_insights.append("Class distribution is moderately/highly imbalanced. Consider oversampling minority classes.")
        if blurry_count / valid_img_count > 0.15:
            health_insights.append("Over 15% of images are detected as blurry. Image sharpening or pruning recommended.")
        if noisy_count / valid_img_count > 0.15:
            health_insights.append("High noise ratio detected. Denoising filters (e.g. Bilateral filter) recommended.")
        if len(duplicates) / valid_img_count > 0.05:
            health_insights.append(f"Found {len(duplicates)} near-duplicate images. Prune duplicates to prevent training bias.")
        if avg_res < 64:
            health_insights.append("Extremely low resolution. Deep learning model classification performance might be limited.")
        
        if not health_insights:
            health_insights.append("The dataset is healthy and well-suited for machine learning benchmarking.")

        return {
            "num_images": total_dataset_images,
            "sampled_analysis_count": valid_img_count,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "class_distribution": dict(class_distribution),
            "class_balance_index": balance_index,
            "resolution_stats": {
                "min_width": int(np.min(widths)),
                "max_width": int(np.max(widths)),
                "avg_width": float(np.mean(widths)),
                "min_height": int(np.min(heights)),
                "max_height": int(np.max(heights)),
                "avg_height": float(np.mean(heights)),
            },
            "grayscale_percentage": float(grayscale_count / valid_img_count) * 100,
            "blurry_percentage": float(blurry_count / valid_img_count) * 100,
            "noisy_percentage": float(noisy_count / valid_img_count) * 100,
            "average_brightness": float(total_brightness / valid_img_count),
            "average_contrast": float(total_contrast / valid_img_count),
            "average_entropy": avg_entropy,
            "duplicates": duplicates,
            "health_score": health_score,
            "complexity_score": complexity_score,
            "health_insights": health_insights
        }
