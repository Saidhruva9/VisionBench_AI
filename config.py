import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_DIR = Path(__file__).resolve().parent

# Load backend environment configuration securely
load_dotenv(PROJECT_DIR / ".env")
UPLOADS_DIR = PROJECT_DIR / "uploads"
DATASETS_DIR = PROJECT_DIR / "datasets"
REPORTS_DIR = PROJECT_DIR / "reports"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"
LOGS_DIR = PROJECT_DIR / "logs"

# Ensure directories exist
for directory in [UPLOADS_DIR, DATASETS_DIR, REPORTS_DIR, TRAINED_MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Analysis Settings
DEFAULT_IMAGE_SIZE = (128, 128)  # standard size for DL classification and analysis resizing
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

# Benchmarking Parameters
FAST_TRAINING_EPOCHS = 3
STANDARD_TRAINING_EPOCHS = 10
BATCH_SIZE = 32
RANDOM_STATE = 42

# AI / Groq Engine Settings
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
GROQ_API_KEY_ENV = "GROQ_API_KEY"

# Fallback UI Messages
UI_GLASS_STYLE = """
<style>
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    .metric-card {
        text-align: center;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00C6FF, #0072FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
"""
