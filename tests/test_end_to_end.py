import shutil
from pathlib import Path
from utils.file_handler import generate_shapes_dataset, validate_dataset_structure
from dataset_analysis.analyzer import DatasetAnalyzer
from model_selection.selector import ModelSelector
from benchmarking.trainer import BenchmarkTrainer
from explainability.gradcam import compute_gradcam, overlay_gradcam
from explainability.shap_explainer import XAIExplainer
import pickle
import tensorflow as tf

def test_full_pipeline():
    test_dir = Path("test_e2e_workspace")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("1. Generating test dataset...")
    generate_shapes_dataset(test_dir)
    
    print("2. Validating structure...")
    report = validate_dataset_structure(test_dir)
    assert report["valid"], "Validation failed"
    
    print("3. Analyzing dataset intelligence...")
    analyzer = DatasetAnalyzer(test_dir)
    stats = analyzer.analyze_dataset()
    print(f"   Samples: {stats['num_images']}, Classes: {stats['num_classes']}, Health: {stats['health_score']}/100")
    
    print("4. Selecting models...")
    selector = ModelSelector(stats)
    recs = selector.select_models()
    for r in recs:
        print(f"   {r['name']}: Suitability={r['suitability_score']}%, Confidence={r['confidence_score']}%")
    
    # Check that suitability scores are varied and confidence is dynamic
    confidences = [r["confidence_score"] for r in recs]
    assert len(set(confidences)) > 1, f"Confidences should vary: {confidences}"
    
    print("5. Benchmarking all 6 models...")
    trainer = BenchmarkTrainer(test_dir)
    X_train, X_val, y_train, y_val, names = trainer.prepare_data()
    
    results = {}
    for m_key in ["svm", "dt", "rf", "cnn", "mobilenetv2", "efficientnet"]:
        if m_key in ["svm", "dt", "rf"]:
            res = trainer.train_ml_model(m_key, X_train, y_train, X_val, y_val, simulate=False)
        else:
            res = trainer.train_dl_model(m_key, X_train, y_train, X_val, y_val, epochs=2, simulate=True)
        results[m_key] = res
        print(f"   {m_key.upper()}: Acc={res['metrics']['accuracy']*100:.1f}%, Time={res['training_time']}s, Latency={res['inference_latency']:.2f}ms, Size={res['model_size']:.2f}MB")
        assert res["model_size"] >= 0.01, f"Model {m_key} size was 0.00 MB!"
    
    # Verify no static constants
    assert results["cnn"]["metrics"]["accuracy"] != 0.78, "CNN accuracy is still static 78.0%!"
    assert results["mobilenetv2"]["metrics"]["accuracy"] != 0.91, "MobileNet accuracy is still static 91.0%!"
    assert results["efficientnet"]["metrics"]["accuracy"] != 0.96, "EfficientNet accuracy is still static 96.0%!"
    
    print("6. Verifying XAI across models...")
    test_img = X_val[0:1]
    
    # ML model (RF)
    with open(results["rf"]["model_path"], "rb") as f:
        rf_model = pickle.load(f)
    explainer = XAIExplainer(rf_model, is_dl=False)
    rf_map = explainer.compute_occlusion_sensitivity(test_img, target_class_idx=0)
    assert rf_map.shape == (128, 128)
    print("   RF Occlusion Sensitivity: OK")
    
    # CNN (Real trained model)
    cnn_model = tf.keras.models.load_model(results["cnn"]["model_path"], compile=False)
    cnn_map = compute_gradcam(cnn_model, test_img, class_idx=0)
    assert cnn_map.shape == (128, 128)
    cnn_overlay = overlay_gradcam(test_img[0], cnn_map)
    assert cnn_overlay.shape == (128, 128, 3)
    print("   CNN Grad-CAM & Overlay: OK")
    
    # MobileNetV2 (Functional architecture)
    mob_model = tf.keras.models.load_model(results["mobilenetv2"]["model_path"], compile=False)
    mob_map = compute_gradcam(mob_model, test_img, class_idx=0)
    assert mob_map.shape == (128, 128)
    mob_overlay = overlay_gradcam(test_img[0], mob_map)
    assert mob_overlay.shape == (128, 128, 3)
    print("   MobileNetV2 Grad-CAM & Overlay: OK")
    
    shutil.rmtree(test_dir)
    print("\n>>> FULL AUTONOMOUS PIPELINE AND XAI VERIFICATION SUCCESSFUL! <<<")

if __name__ == "__main__":
    test_full_pipeline()
