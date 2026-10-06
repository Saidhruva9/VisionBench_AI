# VisionBench AI — Intelligent Dataset-Aware Benchmarking & AI Decision Support Platform for Image Classification

VisionBench AI is an intelligent dataset-aware decision support and benchmarking platform for image classification. It automates dataset health analysis, maps out model suitability rankings, benchmarks traditional machine learning against deep learning transfer architectures, explains predictions visually, and supplies deployment-aware hardware suggestions.

---

## 🧭 System Workflow

```
User Uploads Dataset
          │
          ▼
Dataset Validation (Folder/Corrupt filter)
          │
          ▼
AI Dataset Intelligence Engine (Blur/Noise/Hash)
          │
          ▼
Dataset Quality & Complexity Assessment
          │
          ▼
AI Model Recommendation Rankings
          │
          ▼
Comparative Benchmarking Engine (ML & DL Training Loop)
          │
          ▼
Explainable AI (Grad-CAM & Occlusion Sensitivity)
          │
          ▼
AI Performance Analysis (Latency vs. Accuracy curve)
          │
          ▼
Edge-Cloud Deployment Advisor
          │
          ▼
Interactive Groq Chat Assistant
```

---

## 📁 Project Directory Structure

```
VisionBench_AI/
├── app.py                      # Main Streamlit entrance script
├── config.py                   # Configuration limits and path constants
├── requirements.txt            # Project dependencies
├── README.md                   # User documentation
├── uploads/                    # Upload cache directory
├── datasets/                   # Extracted image dataset storage
├── reports/                    # Generated report outputs
├── trained_models/             # Pickle and Keras model save files
├── logs/                       # System activity outputs
├── dataset_analysis/           # Image diagnostics package
│   ├── __init__.py
│   └── analyzer.py             # Math metrics: Blur, Noise, Average Hash
├── model_selection/            # Recommendation heuristics package
│   ├── __init__.py
│   └── selector.py             # Computes suitability & justifications
├── benchmarking/               # Metrics and training package
│   ├── __init__.py
│   ├── trainer.py              # ML & Keras DL execution
│   └── metrics.py              # Latency, memory, and confusion math
├── explainability/             # Explainable AI (XAI) package
│   ├── __init__.py
│   ├── gradcam.py              # Grad-CAM heatmap generator
│   └── shap_explainer.py       # Pixel Occlusion & global tree features
├── ai_engine/                  # LLM integration package
│   ├── __init__.py
│   └── groq_client.py          # Groq API client with heuristic fallbacks
├── dashboard/                  # Dashboard layout styles package
│   └── ui_components.py        # Glassmorphic themes & CSS injectors
└── tests/                      # Testing package
    └── test_modules.py         # Unit tests suite
```

---

## 🧪 Key Mathematical Formulations & Research Contributions

### 1. Dataset Quality Index ($Q_{dataset}$)
Assesses general dataset hygiene on a scale of $0 - 100$:
$$Q_{dataset} = 100 \times \left(0.30 \cdot H_{norm} + 0.25 \cdot (1 - P_{blur}) + 0.20 \cdot (1 - P_{noise}) + 0.15 \cdot (1 - P_{dup}) + 0.10 \cdot S_{res}\right)$$
Where:
- $H_{norm}$: Normalized Shannon Entropy of class distributions ($1.0$ is perfectly balanced).
- $P_{blur}$: Blurry image percentage ($Var(\text{Laplacian}) < 100$).
- $P_{noise}$: Noisy image percentage (deviation of high-frequency sub-bands).
- $P_{dup}$: Near-duplicate ratio calculated using 64-bit Average Hash Hamming distance ($\le 2$).
- $S_{res}$: Resolution suitability index ($1.0$ if average resolution is $\ge 224px$).

### 2. Perturbation Occlusion Sensitivity Map (XAI)
Calculates pixel-level feature importance for non-differentiable model boundaries:
$$\Delta P(x, y) = P_{\text{baseline}}(C_{\text{target}}) - P_{\text{occluded}}(C_{\text{target}} \mid \text{patch}_{xy} = \text{grey})$$
The difference maps the exact spatial coordinates the model utilizes for prediction.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8 to 3.11 installed.
- Streamlit, OpenCV, and TensorFlow requirements.

### 2. Environment Setup
Clone or copy the directory and navigate into it:
```bash
cd VisionBench_AI
```

Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux/Mac
```

Install requirements:
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify that all packages compile and execute clean math limits:
```bash
python -m unittest tests/test_modules.py
```

### 4. Start Dashboard
Launch the local Streamlit application server:
```bash
streamlit run app.py
```
A browser tab will automatically open at `http://localhost:8501`.

---

## 🧠 Using the AI Capabilities (Groq)
- VisionBench AI incorporates **Groq Llama 3** for advanced recommendations.
- Set a `GROQ_API_KEY` system environment variable, or paste your API key directly inside the **AI Core Configuration** field on the dashboard's sidebar.
- If no key is set, the system seamlessly triggers **Heuristic Fallback Engine** to render pre-defined analysis matrices without crashes.
