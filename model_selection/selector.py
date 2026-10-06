import logging
from typing import Dict, List, Any, Tuple
from config import LOGS_DIR

logger = logging.getLogger("ModelSelector")

class ModelSelector:
    def __init__(self, dataset_stats: Dict[str, Any]):
        self.stats = dataset_stats
        self.num_images = dataset_stats.get("num_images", 0)
        self.num_classes = dataset_stats.get("num_classes", 2)
        self.avg_width = dataset_stats.get("resolution_stats", {}).get("avg_width", 128.0)
        self.avg_height = dataset_stats.get("resolution_stats", {}).get("avg_height", 128.0)
        self.avg_res = (self.avg_width + self.avg_height) / 2.0
        self.blurry_pct = dataset_stats.get("blurry_percentage", 0.0)
        self.noisy_pct = dataset_stats.get("noisy_percentage", 0.0)
        self.health_score = dataset_stats.get("health_score", 100)
        self.complexity_score = dataset_stats.get("complexity_score", 50)
        self.balance_index = dataset_stats.get("class_balance_index", 1.0)

    def select_models(self) -> List[Dict[str, Any]]:
        """
        Computes dynamic suitability scores and confidence levels for all supported models.
        Returns a sorted list of models with detailed analyses.
        """
        logger.info("Recommending models based on dataset statistics")
        models = [
            {"name": "Support Vector Machine (SVM)", "type": "Traditional ML", "key": "svm"},
            {"name": "Decision Tree", "type": "Traditional ML", "key": "dt"},
            {"name": "Random Forest", "type": "Traditional ML", "key": "rf"},
            {"name": "Custom CNN", "type": "Deep Learning", "key": "cnn"},
            {"name": "MobileNetV2", "type": "Deep Learning (Transfer Learning)", "key": "mobilenetv2"},
            {"name": "EfficientNetB0", "type": "Deep Learning (Transfer Learning)", "key": "efficientnet"},
        ]

        recommendations = []
        for model in models:
            score, confidence, why = self._calculate_suitability(model["key"])
            recommendations.append({
                "name": model["name"],
                "type": model["type"],
                "key": model["key"],
                "suitability_score": score,
                "confidence_score": confidence,
                "reasoning": why
            })

        # Sort by suitability score descending
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        return recommendations

    def _calculate_suitability(self, model_key: str) -> Tuple[int, int, str]:
        """
        Calculates dynamic suitability and confidence scores based on empirical dataset metrics:
        - Image volume (N)
        - Class count (K)
        - Mean resolution (R)
        - Health score (H)
        - Complexity score (C)
        - Balance index (B)
        - Noise & Blur ratios
        """
        N = float(self.num_images)
        K = float(self.num_classes)
        R = float(self.avg_res)
        H = float(self.health_score)
        C = float(self.complexity_score)
        B = float(self.balance_index)
        P_blur = float(self.blurry_pct)
        P_noise = float(self.noisy_pct)

        score = 50.0
        confidence = 75.0
        why = ""

        if model_key == "svm":
            # SVM works well on small datasets with low-to-medium feature dimension.
            # Quadratic training complexity with N; multiclass penalty with K*(K-1)/2.
            base = 88.0 - 0.035 * min(N, 1200.0)
            class_pen = max(0.0, K - 2.0) * 2.2
            noise_pen = 0.28 * P_noise
            comp_pen = 0.12 * max(0.0, C - 45.0)
            score = base - class_pen - noise_pen - comp_pen
            confidence = 65.0 + 0.18 * H + 0.08 * (B * 100.0) - 0.12 * P_noise
            why = (
                f"Evaluated on {int(N)} samples across {int(K)} classes (Health: {int(H)}/100). "
                f"SVM handles small datasets well, but multi-class boundaries ({int(K)} classes) "
                f"and noise sensitivity ({P_noise:.1f}%) modulate its suitability."
            )

        elif model_key == "dt":
            # Decision Tree: high interpretability, low training time, high variance.
            base = 66.0 - 0.02 * min(N, 800.0) - 0.20 * C - 0.25 * (1.0 - B) * 100.0
            score = base - 0.12 * P_noise
            confidence = 58.0 + 0.18 * H - 0.10 * P_blur
            why = (
                f"Decision Trees offer fast training and interpretability, but decision splits exhibit high "
                f"variance on complex feature distributions (Complexity: {int(C)}/100, Balance: {B:.2f})."
            )

        elif model_key == "rf":
            # Random Forest: ensemble bagging, resistant to noise and outliers.
            base = 75.0 + 0.018 * min(N, 1000.0) - 0.025 * max(0.0, N - 1800.0)
            noise_resilience = 0.18 * min(P_noise, 30.0)
            class_pen = max(0.0, K - 10.0) * 0.8
            score = base + noise_resilience - class_pen + 0.08 * (H - 70.0)
            confidence = 72.0 + 0.16 * H + 0.08 * (B * 100.0)
            why = (
                f"Random Forest ensemble provides robust generalization against pixel noise ({P_noise:.1f}%) "
                f"and handles multi-class partitioning ({int(K)} classes) reliably."
            )

        elif model_key == "cnn":
            # Custom shallow CNN: learns spatial convolutions from scratch.
            data_adequacy = 0.038 * min(N, 1500.0)
            res_bonus = 6.0 if R < 64.0 else (3.0 if R < 128.0 else 0.0)
            blur_pen = 0.22 * P_blur
            score = 52.0 + data_adequacy + 0.16 * C + res_bonus - blur_pen + 0.1 * (H - 60.0)
            confidence = 62.0 + 0.20 * min(N / 15.0, 25.0) + 0.12 * H
            why = (
                f"Custom shallow CNN learns domain-specific spatial convolutions without pre-training bias. "
                f"Suited to native resolution {int(R)}x{int(R)} and complexity {int(C)}/100."
            )

        elif model_key == "mobilenetv2":
            # MobileNetV2: lightweight transfer learning for edge/mobile deployment.
            data_bonus = 0.022 * min(N, 1200.0)
            res_factor = 0.12 * min(R, 128.0)
            score = 64.0 + data_bonus + 0.15 * C + res_factor - 0.16 * P_blur
            confidence = 70.0 + 0.15 * H + 0.1 * min(N / 20.0, 20.0)
            why = (
                f"MobileNetV2 depthwise separable architecture delivers high parameter efficiency and "
                f"low latency, ideal for real-time mobile and edge inference across {int(K)} classes."
            )

        elif model_key == "efficientnet":
            # EfficientNetB0: compound scaling for maximum representational capacity.
            data_bonus = 0.026 * min(N, 1500.0)
            comp_bonus = 0.22 * C
            res_bonus = 0.14 * min(R, 224.0) - (8.0 if R < 64.0 else 0.0)
            score = 60.0 + data_bonus + comp_bonus + res_bonus - 0.18 * P_blur
            confidence = 73.0 + 0.16 * H + 0.1 * min(N / 25.0, 20.0)
            why = (
                f"EfficientNetB0 compound scaling maximizes feature representation on high-complexity "
                f"images ({int(C)}/100). Recommended for high-accuracy cloud deployment."
            )

        score_int = int(round(max(15.0, min(98.0, score))))
        confidence_int = int(round(max(50.0, min(97.0, confidence))))
        return score_int, confidence_int, why
