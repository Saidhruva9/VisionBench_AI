import unittest
import shutil
import os
import numpy as np
from pathlib import Path
import cv2

# Import local modules to test
from utils.file_handler import validate_dataset_structure, generate_shapes_dataset, is_valid_image
from dataset_analysis.analyzer import DatasetAnalyzer
from model_selection.selector import ModelSelector
from benchmarking.metrics import compute_classification_metrics, get_file_size_mb
from ai_engine.groq_client import GroqEngine
from explainability.gradcam import compute_gradcam, overlay_gradcam
from explainability.shap_explainer import XAIExplainer


class TestVisionBenchAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory structure for testing inside scratch
        cls.test_dir = Path(__file__).resolve().parent.parent / "tests_temp_dir"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        generate_shapes_dataset(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test directories
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_dataset_structure_validation(self):
        """
        Tests if the dataset loader properly validates a structurally compliant dataset.
        """
        report = validate_dataset_structure(self.test_dir)
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["classes"]), 2)
        self.assertIn("circle", report["classes"])
        self.assertIn("square", report["classes"])
        self.assertGreater(report["total_images"], 10)

    def test_analyzer_metrics(self):
        """
        Tests image feature extraction like blur, noise, brightness, contrast, and average hashing.
        """
        analyzer = DatasetAnalyzer(self.test_dir)
        self.assertEqual(len(analyzer.classes), 2)
        self.assertGreater(len(analyzer.image_paths), 10)
        
        # Test hash computation
        first_img = analyzer.image_paths[0]
        im_hash = DatasetAnalyzer.calculate_image_hash(first_img)
        self.assertIsInstance(im_hash, int)
        self.assertNotEqual(im_hash, 0)
        
        # Test single image analysis metrics
        metrics = DatasetAnalyzer.analyze_single_image(first_img)
        self.assertIsNone(metrics["error"])
        self.assertIn("blur_variance", metrics)
        self.assertIn("noise_level", metrics)
        self.assertIn("brightness", metrics)
        self.assertIn("contrast", metrics)
        self.assertIn("entropy", metrics)

    def test_dataset_intelligence_statistics(self):
        """
        Tests calculation of dataset health and complexity index scores.
        """
        analyzer = DatasetAnalyzer(self.test_dir)
        stats = analyzer.analyze_dataset()
        
        self.assertIn("health_score", stats)
        self.assertIn("complexity_score", stats)
        self.assertGreaterEqual(stats["health_score"], 10)
        self.assertLessEqual(stats["health_score"], 100)
        self.assertGreaterEqual(stats["complexity_score"], 10)
        self.assertLessEqual(stats["complexity_score"], 100)
        
        # Near-duplicates should be found since we intentionally wrote a duplicate circle
        self.assertGreaterEqual(len(stats["duplicates"]), 1)

    def test_model_suitability_heuristics(self):
        """
        Tests the model recommendation rankings using mock dataset metrics.
        """
        mock_stats = {
            "num_images": 200,
            "num_classes": 2,
            "resolution_stats": {
                "avg_width": 128.0,
                "avg_height": 128.0
            },
            "blurry_percentage": 5.0,
            "noisy_percentage": 2.0,
            "health_score": 92,
            "complexity_score": 45,
            "class_balance_index": 0.95,
            "duplicates": []
        }
        
        selector = ModelSelector(mock_stats)
        recs = selector.select_models()
        
        self.assertEqual(len(recs), 6)
        # Random Forest or SVM should score highly on small datasets
        self.assertGreater(recs[0]["suitability_score"], recs[-1]["suitability_score"])

    def test_metrics_computations(self):
        """
        Tests classification confusion calculation formulas and disk size checks.
        """
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        
        # Create a temp file to verify file sizes
        temp_file = self.test_dir / "test_size.txt"
        with open(temp_file, "w") as f:
            f.write("A" * 1024 * 1024) # 1 MB of characters
        
        size_mb = get_file_size_mb(temp_file)
        self.assertAlmostEqual(size_mb, 1.0, places=2)

    def test_groq_engine_fallback(self):
        """
        Verifies that Groq Engine fails back to heuristic rule generations when key is absent.
        """
        engine = GroqEngine(api_key="")
        self.assertFalse(engine.is_api_available())
        
        mock_stats = {
            "num_images": 150,
            "num_classes": 3,
            "resolution_stats": {"avg_width": 128.0, "avg_height": 128.0},
            "blurry_percentage": 10.0,
            "noisy_percentage": 5.0,
            "health_score": 85,
            "complexity_score": 60,
            "class_balance_index": 0.85,
            "duplicates": []
        }
        
        # Ensure model recommendations fallback does not crash and returns a report
        recs_report = engine.get_model_recommendations(mock_stats)
        self.assertIn("Executive Architecture Summary", recs_report)
        self.assertIn("Local Heuristics", recs_report)

    def test_xai_explainer(self):
        """
        Verifies Grad-CAM, Pixel Occlusion, and Global Feature Importance on both real and simulated models.
        """
        img_array = np.random.uniform(0, 1, (1, 128, 128, 3)).astype(np.float32)
        mock_model = {"dummy": True, "key": "svm"}
        
        # 1. Test Occlusion Sensitivity with mock/dict model
        explainer = XAIExplainer(mock_model, is_dl=False)
        heatmap = explainer.compute_occlusion_sensitivity(img_array, target_class_idx=0, patch_size=16, step_size=16)
        self.assertEqual(heatmap.shape, (128, 128))
        self.assertGreaterEqual(float(heatmap.max()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0 + 1e-5)
        
        # 2. Test Global Feature Importance with mock/dict model
        global_imp = explainer.get_global_importance()
        self.assertTrue(global_imp["available"])
        self.assertEqual(len(global_imp["importances"]), 10)
        
        # 3. Test Grad-CAM with mock/dict model
        gradcam_map = compute_gradcam(mock_model, img_array, class_idx=0)
        self.assertEqual(gradcam_map.shape, (128, 128))
        
        # 4. Test Heatmap Overlay
        overlay = overlay_gradcam(img_array[0], heatmap, alpha=0.4)
        self.assertEqual(overlay.shape, (128, 128, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_csv_dataset_auto_conversion(self):
        """
        Tests auto-detection and conversion of pixel CSV image datasets into class folders.
        """
        import pandas as pd
        csv_dir = self.test_dir / "csv_test_folder"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock CSV with 2 classes and 64 pixels (8x8)
        data = []
        for i in range(20):
            lbl = 0 if i < 10 else 1
            row = [lbl] + list(np.random.randint(0, 255, 64))
            data.append(row)
        cols = ["label"] + [f"pixel_{j}" for j in range(64)]
        df = pd.DataFrame(data, columns=cols)
        csv_file = csv_dir / "test_mnist.csv"
        df.to_csv(csv_file, index=False)
        
        report = validate_dataset_structure(csv_dir)
        self.assertTrue(report["valid"])
        self.assertGreaterEqual(len(report["classes"]), 2)
        self.assertEqual(report["total_images"], 20)
        
        # Clean up
        shutil.rmtree(csv_dir)

    def test_fer_pixel_string_csv_conversion(self):
        """
        Tests FER-2013 style CSVs where pixels are stored as a space-separated string in one column.
        """
        import pandas as pd
        fer_dir = self.test_dir / "fer_test_folder"
        fer_dir.mkdir(parents=True, exist_ok=True)

        # 16 pixels (4x4)
        data = []
        for i in range(12):
            emotion = "Happy" if i < 6 else "Sad"
            pixels_str = " ".join(str(np.random.randint(0, 255)) for _ in range(16))
            data.append({"emotion": emotion, "pixels": pixels_str})
        df = pd.DataFrame(data)
        df.to_csv(fer_dir / "fer_sample.csv", index=False)

        report = validate_dataset_structure(fer_dir)
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["classes"]), 2)
        self.assertIn("Happy", report["classes"])
        self.assertIn("Sad", report["classes"])
        self.assertEqual(report["total_images"], 12)

        shutil.rmtree(fer_dir)

    def test_csv_indexed_flat_images(self):
        """
        Tests Kaggle-style datasets where a CSV contains [image_id, label] and images are flat files.
        """
        import pandas as pd
        indexed_dir = self.test_dir / "indexed_test_folder"
        indexed_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy image files
        img_names = []
        labels = []
        for i in range(10):
            fname = f"sample_{i:03d}.png"
            label = "Malignant" if i < 5 else "Benign"
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(indexed_dir / fname), img_arr)
            img_names.append(fname)
            labels.append(label)

        df = pd.DataFrame({"image_id": img_names, "diagnosis": labels})
        df.to_csv(indexed_dir / "metadata.csv", index=False)

        report = validate_dataset_structure(indexed_dir)
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["classes"]), 2)
        self.assertIn("Malignant", report["classes"])
        self.assertIn("Benign", report["classes"])
        self.assertEqual(report["total_images"], 10)

        shutil.rmtree(indexed_dir)

    def test_flat_images_filename_pattern_organization(self):
        """
        Tests flat images where labels are embedded in filenames (e.g. cat_001.jpg, dog_002.jpg).
        """
        flat_dir = self.test_dir / "flat_test_folder"
        flat_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            cv2.imwrite(str(flat_dir / f"COVID-19_{i:03d}.png"), np.zeros((32, 32, 3), dtype=np.uint8))
            cv2.imwrite(str(flat_dir / f"Normal_{i:03d}.png"), np.zeros((32, 32, 3), dtype=np.uint8))

        report = validate_dataset_structure(flat_dir)
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["classes"]), 2)
        self.assertIn("Covid-19", report["classes"])
        self.assertIn("Normal", report["classes"])
        self.assertEqual(report["total_images"], 8)

        shutil.rmtree(flat_dir)


if __name__ == "__main__":
    unittest.main()
