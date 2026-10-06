import os
import logging
from typing import Dict, List, Any, Optional
from groq import Groq
from config import DEFAULT_GROQ_MODEL, FALLBACK_GROQ_MODELS, GROQ_API_KEY_ENV

logger = logging.getLogger("GroqEngine")

class GroqEngine:
    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key if api_key is not None else os.environ.get(GROQ_API_KEY_ENV, "")
        self.api_key = raw_key.strip() if raw_key else ""
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.info("No Groq API Key provided. Running in Local Heuristic Fallback mode.")

    def is_api_available(self) -> bool:
        return self.client is not None

    def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Sends query to the Groq API trying models in FALLBACK_GROQ_MODELS.
        Returns the content if successful, or None if all models fail.
        """
        if not self.client:
            return None
            
        models_to_try = [DEFAULT_GROQ_MODEL] + [m for m in FALLBACK_GROQ_MODELS if m != DEFAULT_GROQ_MODEL]
        for model_name in models_to_try:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model_name,
                    temperature=0.2,
                    max_tokens=1500
                )
                if chat_completion and chat_completion.choices:
                    return chat_completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq API call error for model {model_name}: {e}")
                
        logger.error("All Groq models failed. Reverting to local heuristic fallback.")
        return None

    def get_model_recommendations(self, stats: Dict[str, Any]) -> str:
        """
        Retrieves LLM recommendations for which model to use.
        """
        num_images = stats.get("num_images", 0)
        num_classes = stats.get("num_classes", 0)
        avg_res = (stats.get("resolution_stats", {}).get("avg_width", 128) + stats.get("resolution_stats", {}).get("avg_height", 128)) / 2
        noise = stats.get("noisy_percentage", 0.0)
        blur = stats.get("blurry_percentage", 0.0)
        complexity = stats.get("complexity_score", 50)
        
        system_prompt = (
            "You are VisionBench AI, an expert computer vision architect. Analyze the dataset characteristics "
            "and write an academic, production-ready model selection report. Recommend transfer learning parameters, "
            "lightweight vs deep models, and how image sizing impacts performance."
        )
        user_prompt = (
            f"Analyze a dataset with the following statistics:\n"
            f"- Total Images: {num_images}\n"
            f"- Number of Classes: {num_classes}\n"
            f"- Average Resolution: {avg_res:.1f}px\n"
            f"- Blurry Images Ratio: {blur:.1f}%\n"
            f"- Noisy Images Ratio: {noise:.1f}%\n"
            f"- Dataset Complexity Score: {complexity}/100\n\n"
            f"Provide a structured report with:\n"
            f"1. Executive Architecture Summary\n"
            f"2. Traditional ML vs Deep Learning suitability trade-offs\n"
            f"3. Recommended pre-trained backbones and layers to freeze/unfreeze\n"
            f"4. Hyperparameter tuning guidelines (learning rate, optimizer, epochs)"
        )

        if self.is_api_available():
            llm_result = self._query_llm(system_prompt, user_prompt)
            if llm_result:
                return llm_result

        # Heuristic Local Generation
        report = (
            f"### Executive Architecture Summary (Local Heuristics Mode)\n"
            f"Based on **{num_images} images** across **{num_classes} classes**, the dataset "
            f"exhibits a complexity score of **{complexity}/100** with an average image resolution of **{avg_res:.1f}px**.\n\n"
        )
        if num_images < 500:
            report += (
                f"**Recommendation: Traditional ML / Transfer Learning with heavy regularization**\n"
                f"* Small dataset size makes training convolutional networks from scratch (e.g. Custom CNN) highly prone to overfitting.\n"
                f"* Use **Random Forest** or **SVM** on extracted color/HOG feature matrices for robust, fast baselines.\n"
                f"* For deep learning, utilize **MobileNetV2** with pre-trained ImageNet weights, freezing the base completely (100% layers frozen) to act as a static feature extractor.\n"
            )
        else:
            report += (
                f"**Recommendation: Deep Learning (Transfer Learning & Custom CNN)**\n"
                f"* Sufficient samples are present to utilize advanced feature learners.\n"
                f"* **EfficientNetB0** is recommended for cloud/server scale due to superior compound-scaled feature capture.\n"
                f"* **MobileNetV2** is ideal if latency and deployment efficiency are prioritised.\n"
                f"* A **Custom CNN** can be trained from scratch; ensure data augmentation (rotations, flips, zooming) is applied to mitigate variance.\n"
            )
        
        report += (
            f"\n### Hyperparameter Tuning Guidelines\n"
            f"- **Optimizer**: Adam (learning rate = $10^{-3}$ for Custom CNN, $10^{-4}$ for fine-tuning pre-trained models).\n"
            f"- **Batch Size**: 32 (suitable for general CPU/GPU hardware boundaries).\n"
            f"- **Loss**: Sparse Categorical Crossentropy.\n"
        )
        return report

    def get_performance_analysis(self, stats: Dict[str, Any], results: Dict[str, Any]) -> str:
        """
        Analyzes the actual benchmark results (accuracy, latency, memory, etc.) and provides deployment advice.
        """
        # Format results for prompt
        results_str = ""
        for m_key, r in results.items():
            metrics = r.get("metrics", {})
            results_str += (
                f"- {m_key.upper()}:\n"
                f"  * Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                f"  * F1 Score: {metrics.get('f1_score', 0):.4f}\n"
                f"  * Training Time: {r.get('training_time', 0):.2f}s\n"
                f"  * Inference Latency: {r.get('inference_latency', 0):.3f}ms\n"
                f"  * Peak Memory: {r.get('peak_memory_used', 0):.2f}MB\n"
                f"  * Model Size: {r.get('model_size', 0):.2f}MB\n\n"
            )

        system_prompt = (
            "You are VisionBench AI Decision Support Engine. Analyze the empirical benchmark metrics "
            "for various image classification models. Write a scientific comparative trade-off summary. "
            "Highlight speed vs. accuracy trade-offs, compute density, and provide actionable advice "
            "for deployment (cloud, mobile, IoT, real-time)."
        )
        user_prompt = (
            f"Here are the empirical training and inference metrics for this dataset:\n\n"
            f"{results_str}"
            f"Please generate a comprehensive benchmarking report discussing:\n"
            f"1. Overall Winner (highest accuracy & balanced f1)\n"
            f"2. Speed-Accuracy Trade-off Curve (Latency vs. Accuracy)\n"
            f"3. Operational Efficiency (Memory & Disk Footprint analysis)\n"
            f"4. Deployment Advisor matrix (Mobile/Edge vs. Cloud Server recommendations)"
        )

        if self.is_api_available():
            llm_result = self._query_llm(system_prompt, user_prompt)
            if llm_result:
                return llm_result

        # Heuristic Local Generation
        # Find best model
        best_model_name = "N/A"
        best_acc = -1.0
        fastest_model_name = "N/A"
        fastest_latency = 99999.0
        
        for m_key, r in results.items():
            acc = r.get("metrics", {}).get("accuracy", 0.0)
            lat = r.get("inference_latency", 99999.0)
            if acc > best_acc:
                best_acc = acc
                best_model_name = m_key.upper()
            if lat < fastest_latency:
                fastest_latency = lat
                fastest_model_name = m_key.upper()

        report = (
            f"### Benchmark Performance Analysis (Local Heuristics Mode)\n"
            f"The benchmarking run completed evaluating **{len(results)} models** on the dataset.\n\n"
            f"1. **Performance Leader**: **{best_model_name}** achieved the highest accuracy of **{best_acc*100:.1f}%**.\n"
            f"2. **Latency Leader**: **{fastest_model_name}** recorded the lowest single-image prediction latency of **{fastest_latency:.2f} ms**.\n\n"
            f"### Empirical Trade-offs:\n"
            f"* **Speed vs. Accuracy**: Deep learning models (e.g. MobileNetV2, EfficientNet) provide superior accuracy but introduce higher latency (10-50ms) and require deep libraries. Traditional ML models (Random Forest, SVM) exhibit ultra-low latency (<2ms) and compact storage, though accuracy is constrained by structural features.\n"
            f"* **Compute Density**: Pre-trained Transfer learning models (EfficientNet) demand higher peak memory (typically >200MB during training) and produce large model binaries (50-100MB). Standard ML pickling creates highly portable packages (<10MB).\n\n"
            f"### Deployment Suggestions Matrix:\n"
            f"- **Edge IoT (Raspberry Pi/Smartphones)**: Recommend **MobileNetV2** (high accuracy, optimized MobileNet architecture) or **Random Forest** (zero deep learning framework dependencies, ultra-low resource usage).\n"
            f"- **High-Accuracy Cloud Hosting (AWS/GCP APIs)**: Recommend **EfficientNetB0** to capture maximum class variance.\n"
            f"- **Real-time Pipelines (Industrial Sorting, Robotics)**: Recommend **Custom CNN** or **SVM** to maximize throughput while maintaining a compact parameter footprint."
        )
        return report

    def get_dataset_improvements(self, stats: Dict[str, Any]) -> str:
        """
        Inspects dataset metrics and creates improvement suggestions.
        """
        num_images = stats.get("num_images", 0)
        num_classes = stats.get("num_classes", 0)
        noise = stats.get("noisy_percentage", 0.0)
        blur = stats.get("blurry_percentage", 0.0)
        balance = stats.get("class_balance_index", 1.0)
        duplicates = len(stats.get("duplicates", []))
        
        system_prompt = (
            "You are a computer vision data scientist. Analyze the dataset health metrics and provide "
            "concrete data engineering recommendations to improve dataset quality, balance, and clean up "
            "defects to improve subsequent model training accuracy."
        )
        user_prompt = (
            f"Here are the dataset quality statistics:\n"
            f"- Total Images: {num_images}\n"
            f"- Number of Classes: {num_classes}\n"
            f"- Class Balance Shannon Index: {balance:.2f} (1.0 is perfectly balanced)\n"
            f"- Near-duplicate pairs: {duplicates}\n"
            f"- Blurry images ratio: {blur:.1f}%\n"
            f"- Noisy images ratio: {noise:.1f}%\n\n"
            f"Provide a structured Data Improvement Guide with specific recommendations."
        )

        if self.is_api_available():
            llm_result = self._query_llm(system_prompt, user_prompt)
            if llm_result:
                return llm_result

        # Heuristic Local Generation
        guide = (
            f"### AI Dataset Improvement Advisor (Local Heuristics Mode)\n"
            f"The dataset quality assessment score is **{stats.get('health_score', 100)}/100**.\n\n"
            f"#### Core Issues Detected & Mitigations:\n"
        )
        if balance < 0.8:
            guide += (
                f"1. **Imbalanced Class Distributions** (Score: {balance:.2f}/1.0)\n"
                f"   * *Impact*: Model will bias towards majority classes, showing inflated accuracy but poor recall on minority categories.\n"
                f"   * *Mitigation*: Apply **Synthetic Data Augmentation** (geometric rotations, color jitters) to minority classes. Or utilize class weights in loss calculation (`class_weight` in Keras, `class_weight='balanced'` in SVM).\n\n"
            )
        if duplicates > 0:
            guide += (
                f"2. **Data Leakage Risk via Duplicates** ({duplicates} pairs found)\n"
                f"   * *Impact*: Overfitting and inflated validation score because identical images exist in both train and validation sets.\n"
                f"   * *Mitigation*: Delete the duplicate images. The platform's analysis logs list the file names of identified duplicate groups.\n\n"
            )
        if blur > 15.0:
            guide += (
                f"3. **High Blur Ratio** ({blur:.1f}% blurry images)\n"
                f"   * *Impact*: Loss of edge and texture features, causing high bias.\n"
                f"   * *Mitigation*: Filter out low-variance images. Apply **Unsharp Masking** filtering ($I_{sharpened} = I + \\alpha(I - I_{blurred})$) in image preprocessing pipeline.\n\n"
            )
        if noise > 15.0:
            guide += (
                f"4. **High Image Noise** ({noise:.1f}% noisy images)\n"
                f"   * *Impact*: Unwanted high-frequency variations degrade model generalization.\n"
                f"   * *Mitigation*: Apply **Bilateral Filtering** (`cv2.bilateralFilter()`) or Non-Local Means Denoising to preserve edge structures while smoothing out speckle noise.\n\n"
            )
        if not (balance < 0.8 or duplicates > 0 or blur > 15.0 or noise > 15.0):
            guide += "No major data quality defects detected! The dataset is clean, balanced, and ready for deployment-grade training.\n"

        return guide

    def chat_response(self, query: str, stats: Dict[str, Any], results: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """
        Interactive conversational chatbot powered by Groq.
        """
        # Prepare context summaries
        hist_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in chat_history[-5:]])
        
        stat_summary = (
            f"Dataset: {stats.get('num_images', 0)} images, {stats.get('num_classes', 0)} classes. "
            f"Health Score: {stats.get('health_score', 100)}/100. Complexity: {stats.get('complexity_score', 50)}/100."
        )
        
        bench_summary = ""
        for m_key, r in results.items():
            acc = r.get("metrics", {}).get("accuracy", 0.0)
            lat = r.get("inference_latency", 0.0)
            bench_summary += f"{m_key.upper()} (Acc: {acc*100:.1f}%, Latency: {lat:.2f}ms), "

        system_prompt = (
            f"You are the VisionBench AI Assistant, an interactive chatbot specializing in AI decision support, "
            f"image classification benchmarking, and computer vision. You have access to the current project context:\n"
            f"- Dataset Stats: {stat_summary}\n"
            f"- Benchmarking Results: {bench_summary}\n\n"
            f"Answer the user's questions clearly, concisely, and educationally. Draw from the empirical results."
        )

        user_prompt = (
            f"Conversation History:\n{hist_str}\n\n"
            f"User Query: {query}\n"
            f"Response:"
        )

        if self.is_api_available():
            llm_result = self._query_llm(system_prompt, user_prompt)
            if llm_result:
                return llm_result

        # Heuristic Conversational QA Fallback
        query_lower = query.lower()
        
        if "mobilenet" in query_lower:
            return (
                "MobileNetV2 is designed for mobile and edge devices. It utilizes depthwise separable convolutions, "
                "which greatly reduces parameters (approx 2.2M) and computation (FLOPs) compared to standard CNNs. "
                "In our benchmark, it represents an optimal trade-off of fast inference with highly competitive accuracy."
            )
        elif "efficientnet" in query_lower:
            return (
                "EfficientNet utilizes compound scaling, balancing network depth, width, and resolution. "
                "EfficientNetB0 is highly accurate because it starts with a clean baseline designed via Neural Architecture Search (NAS). "
                "It is typically slower and has a larger memory footprint than MobileNet, making it ideal for cloud endpoints rather than low-end edge devices."
            )
        elif "svm" in query_lower or "support vector" in query_lower:
            return (
                "SVM (Support Vector Machine) finds the hyperplane that maximizes the margin between classes. "
                "For image benchmarking, we extract color histograms and structural pixels, then fit the SVM. "
                "SVM is very fast to train on small datasets and requires zero deep learning resources (like GPUs), but "
                "lacks the capacity to learn deep hierarchical features on large, complex datasets."
            )
        elif "android" in query_lower or "mobile" in query_lower or "edge" in query_lower or "raspberry" in query_lower:
            return (
                "For mobile or edge deployment (like Android or Raspberry Pi), MobileNetV2 is highly recommended. "
                "It can be exported directly to TensorFlow Lite (TFLite) format, allowing hardware acceleration. "
                "Alternatively, a Random Forest classifier is highly lightweight if you extract features locally, since it "
                "can run on raw CPU with no heavy neural network libraries."
            )
        elif "imbalance" in query_lower or "improve" in query_lower:
            return (
                "To improve your dataset, focus on the insights in the 'Improvement Advisor' tab. "
                "If you have class imbalance, try applying random image augmentations (flips, rotations, scaling) "
                "to the minority class folders, or prune duplicates to ensure the model doesn't overfit on redundant images."
            )
        else:
            return (
                f"I am the VisionBench AI Assistant running in local fallback mode. I see your dataset has "
                f"{stats.get('num_images', 0)} images across {stats.get('num_classes', 0)} classes. "
                f"The best performing model in your benchmark is the model with the highest validation accuracy. "
                f"Please ask about 'MobileNetV2 vs EfficientNet', 'SVM features', or 'how to improve imbalanced datasets' for specific answers!"
            )
