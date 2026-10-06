from typing import Any
import streamlit as st
from config import UI_GLASS_STYLE

def inject_custom_css():
    """
    Injects global stylesheet and glassmorphic card configurations.
    """
    # Load Outfit and Inter fonts
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )
    
    # Inject Custom CSS overrides
    st.markdown(UI_GLASS_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            /* Base fonts */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Outfit', sans-serif;
                font-weight: 600;
                color: #F8FAFC;
                letter-spacing: -0.5px;
            }
            
            /* Sidebar Custom Styling */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.05);
            }
            section[data-testid="stSidebar"] .stMarkdown {
                font-family: 'Outfit', sans-serif;
            }

            /* Custom background for the whole page */
            .stApp {
                background-color: #0B0F19;
                background-image: 
                    radial-gradient(at 10% 20%, rgba(30, 41, 59, 0.4) 0px, transparent 50%),
                    radial-gradient(at 90% 80%, rgba(3, 105, 161, 0.15) 0px, transparent 50%);
                background-attachment: fixed;
            }

            /* Buttons custom styling */
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
            }
            div.stButton > button:first-child:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
                background: linear-gradient(135deg, #0ea5e9 0%, #0284C7 100%);
            }
            
            /* Custom headers decoration */
            .section-header {
                border-left: 4px solid #0EA5E9;
                padding-left: 12px;
                margin-bottom: 25px;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 1.1rem;
                color: #38BDF8 !important;
            }
            
            /* Table formatting */
            .dataframe {
                background-color: rgba(15, 23, 42, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
                border-radius: 10px !important;
            }

            /* Workflow Architecture Card Styles */
            .workflow-container {
                display: flex;
                flex-direction: column;
                gap: 6px;
                width: 100%;
                margin-top: 5px;
            }
            .workflow-card {
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid rgba(56, 189, 248, 0.2);
                border-radius: 10px;
                padding: 10px 14px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-sizing: border-box;
            }
            .workflow-card-accent {
                border: 1px solid rgba(16, 185, 129, 0.35);
                background: rgba(15, 23, 42, 0.75);
            }
            .workflow-card-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .workflow-num {
                background: linear-gradient(135deg, #0284C7, #0369A1);
                color: #ffffff;
                width: 32px;
                height: 32px;
                min-width: 32px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 0.85rem;
            }
            .workflow-num-accent {
                background: linear-gradient(135deg, #059669, #10B981);
            }
            .workflow-title {
                color: #F8FAFC;
                font-weight: 600;
                font-size: 0.88rem;
                line-height: 1.2;
            }
            .workflow-desc {
                color: #94A3B8;
                font-size: 0.73rem;
                margin-top: 2px;
                line-height: 1.2;
            }
            .workflow-icon {
                font-size: 1.25rem;
                margin-left: 8px;
            }
            .workflow-arrow {
                text-align: center;
                color: #38BDF8;
                font-size: 0.85rem;
                line-height: 1;
                margin: -2px 0;
                opacity: 0.85;
            }
            .workflow-arrow-accent {
                color: #10B981;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_hero_section():
    """
    Renders the premium platform banner at the top of the Home page.
    """
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(30, 41, 59, 0.4)); margin-bottom: 30px;">
            <div style="font-size: 0.85rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #38BDF8; margin-bottom: 10px;">
                AI-Powered Decision Support
            </div>
            <h1 style="font-size: 3rem; margin: 0; font-weight: 800; background: linear-gradient(90deg, #F8FAFC, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                VisionBench AI
            </h1>
            <p style="font-size: 1.15rem; color: #94A3B8; max-width: 700px; margin: 15px auto 0 auto; line-height: 1.6;">
                An intelligent image classification benchmarking platform. Analyze datasets automatically, discover architecture suitability rankings, train comparative ML/DL models, and analyze predictions using explainability matrices.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_workflow_architecture():
    """
    Renders the 6-stage end-to-end workflow architecture cards with guaranteed layout integrity.
    """
    html_markup = (
        '<div class="workflow-container">'
        '  <div class="workflow-card">'
        '    <div class="workflow-card-left">'
        '      <div class="workflow-num">01</div>'
        '      <div>'
        '        <div class="workflow-title">Dataset Ingestion & Validation</div>'
        '        <div class="workflow-desc">ZIP extraction, corrupt file filter & directory verification</div>'
        '      </div>'
        '    </div>'
        '    <span class="workflow-icon">📂</span>'
        '  </div>'
        '  <div class="workflow-arrow">↓</div>'
        '  <div class="workflow-card">'
        '    <div class="workflow-card-left">'
        '      <div class="workflow-num">02</div>'
        '      <div>'
        '        <div class="workflow-title">AI Dataset Intelligence</div>'
        '        <div class="workflow-desc">Blur Laplacian, noise ratio, Shannon entropy & hash duplicates</div>'
        '      </div>'
        '    </div>'
        '    <span class="workflow-icon">📊</span>'
        '  </div>'
        '  <div class="workflow-arrow">↓</div>'
        '  <div class="workflow-card">'
        '    <div class="workflow-card-left">'
        '      <div class="workflow-num">03</div>'
        '      <div>'
        '        <div class="workflow-title">Model Recommendations</div>'
        '        <div class="workflow-desc">Mathematical suitability scoring & AI architecture guidelines</div>'
        '      </div>'
        '    </div>'
        '    <span class="workflow-icon">💡</span>'
        '  </div>'
        '  <div class="workflow-arrow">↓</div>'
        '  <div class="workflow-card">'
        '    <div class="workflow-card-left">'
        '      <div class="workflow-num">04</div>'
        '      <div>'
        '        <div class="workflow-title">Comparative Benchmarking</div>'
        '        <div class="workflow-desc">6-model training loop: Accuracy, F1, latency, memory & size</div>'
        '      </div>'
        '    </div>'
        '    <span class="workflow-icon">⚙️</span>'
        '  </div>'
        '  <div class="workflow-arrow">↓</div>'
        '  <div class="workflow-card">'
        '    <div class="workflow-card-left">'
        '      <div class="workflow-num">05</div>'
        '      <div>'
        '        <div class="workflow-title">Explainable AI (XAI)</div>'
        '        <div class="workflow-desc">Grad-CAM attention heatmaps & Pixel Occlusion sensitivity</div>'
        '      </div>'
        '    </div>'
        '    <span class="workflow-icon">🔍</span>'
        '  </div>'
        '  <div class="workflow-arrow workflow-arrow-accent">↓</div>'
        '  <div class="workflow-card workflow-card-accent">'
        '    <div class="workflow-card-left">'
        '      <div class="workflow-num workflow-num-accent">06</div>'
        '      <div>'
        '        <div class="workflow-title">AI Decision Support & Chatbot</div>'
        '        <div class="workflow-desc">Speed vs accuracy trade-offs, deployment advisor & Groq QA</div>'
        '      </div>'
        '    </div>'
        '    <span class="workflow-icon">🤖</span>'
        '  </div>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(html_markup)
    else:
        st.markdown(html_markup, unsafe_allow_html=True)

def render_metric_card(title: str, value: Any, subtitle: str = "") -> str:
    """
    Generates HTML string for a glassmorphic metric card.
    """
    subtitle_html = f"<div style='font-size: 0.75rem; color: #64748B; margin-top: 5px;'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {subtitle_html}
    </div>
    """

def render_status_pill(status: str, message: str):
    """
    Renders a clean styled status block.
    """
    colors = {
        "success": ("rgba(16, 185, 129, 0.1)", "rgba(16, 185, 129, 0.2)", "#34D399"),
        "warning": ("rgba(245, 158, 11, 0.1)", "rgba(245, 158, 11, 0.2)", "#FBBF24"),
        "error": ("rgba(239, 68, 68, 0.1)", "rgba(239, 68, 68, 0.2)", "#F87171"),
        "info": ("rgba(59, 130, 246, 0.1)", "rgba(59, 130, 246, 0.2)", "#60A5FA")
    }
    bg, border, text = colors.get(status, colors["info"])
    
    st.markdown(
        f"""
        <div style="background-color: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px;">
            <div style="color: {text}; font-weight: 500; font-size: 0.95rem;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
