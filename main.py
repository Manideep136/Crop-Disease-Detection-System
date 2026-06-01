import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import colorsys
import os
import gdown

# ── Leaf Validator ────────────────────────────────────────────────────────────
def is_leaf_image(uploaded_file):
    """
    Validates whether the uploaded image is a plant leaf using two checks:
    1. Green pixel ratio — real leaves have significant green content
    2. Confidence gate — if model confidence is too low, it's likely not a leaf
    Returns (is_valid: bool, reason: str)
    """
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        img_small = img.resize((100, 100))
        pixels = list(img_small.getdata())

        green_count = 0
        brown_tan_count = 0
        total = len(pixels)

        for r, g, b in pixels:
            # Normalize
            rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
            h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
            h_deg = h * 360

            # Green range (healthy leaves): hue 60°–165°, decent saturation
            if 60 <= h_deg <= 165 and s > 0.15 and v > 0.15:
                green_count += 1
            # Brown/tan range (diseased/dry leaves): hue 15°–55°, low-mid saturation
            elif 15 <= h_deg <= 55 and 0.1 < s < 0.75 and v > 0.15:
                brown_tan_count += 1

        plant_ratio = (green_count + brown_tan_count) / total

        # Reset file pointer for downstream use
        uploaded_file.seek(0)

        if plant_ratio < 0.10:
            return False, f"Only {plant_ratio*100:.1f}% of the image contains plant-like colors (green/brown). Please upload a clear leaf photo."

        return True, "ok"

    except Exception as e:
        uploaded_file.seek(0)
        return False, f"Could not process image: {str(e)}"


# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model Prediction ────────────────────────────────────────────────
def model_prediction(test_image):
    model_path = "trained_model.keras"
    
    # Check if the model already exists locally on the Streamlit cloud server
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model weights from Google Drive... Please wait."):
            # Your exact verified shareable Google Drive File ID:
            file_id = "1ic_q416Mabj1W83YB379DwMfcKgA_hB2"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            
    # Load the model cleanly from the verified local path
    model = tf.keras.models.load_model(model_path)
    
    # Load and resize image target package
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    # ── CRITICAL FIX: RESCALE PIXELS TO MATCH TRAINING (0-1) ──
    input_arr = input_arr / 255.0
    
    input_arr = np.array([input_arr])  # Convert single image to a batch row
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    return result_index, confidence

# ── Disease Info Database ─────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple Scab": {
        "severity": "Moderate",
        "cause": "Fungal (Venturia inaequalis)",
        "symptoms": "Olive-green to brown spots on leaves and fruit",
        "treatment": ["Apply fungicides (Captan, Mancozeb)", "Remove infected leaves", "Improve air circulation", "Avoid overhead irrigation"],
        "prevention": "Plant resistant varieties; prune trees regularly",
        "color": "#f59e0b"
    },
    "Black Rot": {
        "severity": "High",
        "cause": "Fungal (Botryosphaeria obtusa)",
        "symptoms": "Circular brown lesions with purple borders on leaves",
        "treatment": ["Prune infected branches", "Apply copper-based fungicides", "Remove mummified fruits", "Disinfect pruning tools"],
        "prevention": "Maintain good sanitation; remove dead wood",
        "color": "#ef4444"
    },
    "Cedar Apple Rust": {
        "severity": "Moderate",
        "cause": "Fungal (Gymnosporangium juniperi-virginianae)",
        "symptoms": "Bright orange-yellow spots on upper leaf surfaces",
        "treatment": ["Apply myclobutanil or triadimefon fungicides", "Remove nearby cedar/juniper trees", "Apply preventive sprays in spring"],
        "prevention": "Plant resistant apple varieties",
        "color": "#f97316"
    },
    "Powdery Mildew": {
        "severity": "Moderate",
        "cause": "Fungal (various Erysiphe spp.)",
        "symptoms": "White powdery coating on leaves and stems",
        "treatment": ["Apply sulfur-based fungicides", "Use neem oil spray", "Improve air circulation", "Avoid excess nitrogen fertilizer"],
        "prevention": "Choose resistant varieties; water at base of plants",
        "color": "#8b5cf6"
    },
    "Cercospora Leaf Spot": {
        "severity": "Moderate",
        "cause": "Fungal (Cercospora zeae-maydis)",
        "symptoms": "Rectangular gray lesions with tan centers",
        "treatment": ["Apply strobilurin or triazole fungicides", "Plant resistant hybrids", "Rotate crops", "Reduce leaf wetness"],
        "prevention": "Crop rotation with non-host plants",
        "color": "#6b7280"
    },
    "Common Rust": {
        "severity": "Moderate",
        "cause": "Fungal (Puccinia sorghi)",
        "symptoms": "Circular to elongated brown pustules on both leaf surfaces",
        "treatment": ["Apply foliar fungicides early", "Use resistant corn hybrids", "Monitor crops regularly"],
        "prevention": "Plant resistant hybrids; early planting",
        "color": "#b45309"
    },
    "Northern Leaf Blight": {
        "severity": "High",
        "cause": "Fungal (Exserohilum turcicum)",
        "symptoms": "Long cigar-shaped gray-green to tan lesions",
        "treatment": ["Apply triazole or strobilurin fungicides", "Plant resistant varieties", "Practice crop rotation"],
        "prevention": "Avoid continuous corn planting; use resistant hybrids",
        "color": "#ef4444"
    },
    "Black Rot (Grape)": {
        "severity": "High",
        "cause": "Fungal (Guignardia bidwellii)",
        "symptoms": "Brown circular leaf spots with black borders; shriveled fruit",
        "treatment": ["Apply mancozeb or myclobutanil", "Remove mummified berries", "Prune for air circulation"],
        "prevention": "Remove infected plant material promptly",
        "color": "#dc2626"
    },
    "Esca (Black Measles)": {
        "severity": "High",
        "cause": "Fungal complex (Phaeomoniella, Phaeoacremonium)",
        "symptoms": "Tiger-stripe pattern on leaves; dark berries",
        "treatment": ["No cure; manage with pruning", "Apply wound sealants after pruning", "Remove heavily infected vines"],
        "prevention": "Avoid large pruning wounds; use sterilized tools",
        "color": "#7c3aed"
    },
    "Leaf Blight": {
        "severity": "Moderate",
        "cause": "Fungal/Bacterial",
        "symptoms": "Brown, water-soaked lesions spreading from leaf margins",
        "treatment": ["Apply copper-based fungicides", "Improve drainage", "Remove infected tissue"],
        "prevention": "Adequate plant spacing; avoid leaf wetness",
        "color": "#d97706"
    },
    "Huanglongbing (Citrus Greening)": {
        "severity": "Critical",
        "cause": "Bacterial (Candidatus Liberibacter)",
        "symptoms": "Yellowing of shoots; blotchy mottled leaves; lopsided fruit",
        "treatment": ["No cure available", "Remove and destroy infected trees", "Control psyllid vector with insecticides", "Use certified disease-free nursery stock"],
        "prevention": "Strict psyllid management; quarantine measures",
        "color": "#dc2626"
    },
    "Bacterial Spot": {
        "severity": "Moderate",
        "cause": "Bacterial (Xanthomonas spp.)",
        "symptoms": "Water-soaked spots that turn brown/black with yellow halos",
        "treatment": ["Apply copper hydroxide sprays", "Use bactericides", "Avoid working with wet plants", "Remove infected plant parts"],
        "prevention": "Use disease-free seeds; crop rotation",
        "color": "#f59e0b"
    },
    "Early Blight": {
        "severity": "Moderate",
        "cause": "Fungal (Alternaria solani)",
        "symptoms": "Dark brown spots with concentric rings (target-board pattern)",
        "treatment": ["Apply chlorothalonil or mancozeb", "Remove lower infected leaves", "Mulch around plants", "Water at soil level"],
        "prevention": "Crop rotation; resistant varieties",
        "color": "#d97706"
    },
    "Late Blight": {
        "severity": "Critical",
        "cause": "Oomycete (Phytophthora infestans)",
        "symptoms": "Dark water-soaked lesions with white mold on undersides",
        "treatment": ["Apply metalaxyl or cymoxanil immediately", "Remove and destroy infected plants", "Avoid overhead watering", "Apply copper fungicides preventively"],
        "prevention": "Use certified blight-resistant seed; monitor humidity",
        "color": "#dc2626"
    },
    "Leaf Mold": {
        "severity": "Moderate",
        "cause": "Fungal (Passalora fulva)",
        "symptoms": "Yellow patches on upper leaf surface; olive-brown mold below",
        "treatment": ["Apply fungicides (chlorothalonil)", "Improve greenhouse ventilation", "Reduce humidity", "Remove infected leaves"],
        "prevention": "Maintain low humidity; resistant varieties",
        "color": "#65a30d"
    },
    "Septoria Leaf Spot": {
        "severity": "Moderate",
        "cause": "Fungal (Septoria lycopersici)",
        "symptoms": "Small circular spots with dark borders and light centers",
        "treatment": ["Apply mancozeb or copper fungicides", "Remove lower infected leaves", "Avoid splashing water"],
        "prevention": "Crop rotation; mulching",
        "color": "#6b7280"
    },
    "Spider Mites": {
        "severity": "Moderate",
        "cause": "Arachnid pest (Tetranychus urticae)",
        "symptoms": "Fine webbing; stippled, yellowing leaves; bronzing",
        "treatment": ["Apply miticides or neem oil", "Increase humidity", "Introduce predatory mites", "Strong water spray to dislodge mites"],
        "prevention": "Monitor regularly; avoid dusty conditions",
        "color": "#ef4444"
    },
    "Target Spot": {
        "severity": "Moderate",
        "cause": "Fungal (Corynespora cassiicola)",
        "symptoms": "Concentric ring spots; lesions coalesce in severe cases",
        "treatment": ["Apply azoxystrobin or difenoconazole", "Remove infected tissue", "Improve air circulation"],
        "prevention": "Avoid overcrowding plants",
        "color": "#f97316"
    },
    "Yellow Leaf Curl Virus": {
        "severity": "High",
        "cause": "Viral (Tomato Yellow Leaf Curl Virus — TYLCV)",
        "symptoms": "Upward leaf curling; yellowing; stunted growth",
        "treatment": ["No cure; remove infected plants", "Control whitefly vector aggressively", "Apply reflective mulches"],
        "prevention": "Use virus-resistant varieties; insect netting",
        "color": "#eab308"
    },
    "Mosaic Virus": {
        "severity": "High",
        "cause": "Viral (Tomato Mosaic Virus — ToMV)",
        "symptoms": "Mosaic pattern of light/dark green; distorted leaves",
        "treatment": ["Remove infected plants immediately", "Disinfect tools with bleach", "Control aphid vectors"],
        "prevention": "Use certified virus-free seeds; wash hands before handling",
        "color": "#a16207"
    },
    "Leaf Scorch": {
        "severity": "Low",
        "cause": "Fungal (Diplocarpon earlianum)",
        "symptoms": "Purple-bordered spots that merge; scorched appearance",
        "treatment": ["Apply captan or myclobutanil fungicides", "Remove infected leaves", "Avoid overhead irrigation"],
        "prevention": "Proper plant spacing; good air circulation",
        "color": "#f97316"
    },
}

# ── Class Names ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Huanglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Bell Pepper - Bacterial Spot', 'Bell Pepper - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites',
    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

def get_disease_key(disease_name):
    """Match disease name to info database."""
    for key in DISEASE_INFO:
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return key
    # Partial word match
    for key in DISEASE_INFO:
        words = key.lower().split()
        for w in words:
            if len(w) > 4 and w in disease_name.lower():
                return key
    return None

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f35 0%, #0a2840 100%) !important;
    border-right: 1px solid rgba(34, 197, 94, 0.2);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Main text */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f0fdf4 !important; }
p, li, div { color: #cbd5e1; }

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}
.card-green {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.06));
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-red {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Logo text */
.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #22c55e, #10b981, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.logo-sub {
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 500;
}

/* Stat boxes */
.stat-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #22c55e !important;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Confidence bar */
.conf-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #22c55e, #10b981);
    transition: width 0.8s ease;
}

/* Treatment steps */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
    padding: 0.6rem 1rem;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    border-left: 3px solid #22c55e;
}
.step-num {
    background: #22c55e;
    color: #000 !important;
    font-weight: 700;
    font-size: 0.75rem;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}

/* Severity badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.badge-critical { background: rgba(220,38,38,0.2); color: #fca5a5; border: 1px solid rgba(220,38,38,0.4); }
.badge-high     { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.badge-moderate { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
.badge-low      { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }
.badge-healthy  { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }

/* Feature cards on home */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(34,197,94,0.12);
    border-radius: 14px;
    padding: 1.4rem;
    height: 100%;
    transition: border-color 0.2s;
}

/* Divider */
.green-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #22c55e, transparent);
    margin: 1.5rem 0;
    border: none;
}

/* Hide streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Stagger animation */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate { animation: fadeSlideUp 0.6s ease forwards; }
.d1 { animation-delay: 0.1s; opacity: 0; }
.d2 { animation-delay: 0.2s; opacity: 0; }
.d3 { animation-delay: 0.3s; opacity: 0; }
.d4 { animation-delay: 0.4s; opacity: 0; }

/* ── REMOVED ILLEGITIMATE BLANK GAP FROM FILE UPLOADER DRAWER ── */
div[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, rgba(34, 197, 94, 0.04) 0%, rgba(15, 23, 42, 0.4) 100%) !important;
    border: 2px dashed rgba(34, 197, 94, 0.3) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: -25px !important;  /* Bypasses layout padding issues */
}
div[data-testid="stFileUploader"]:hover {
    border-color: #22c55e !important;
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.12) !important;
}
div[data-testid="stFileUploader"] section {
    padding: 0px !important;
    background: transparent !important;
}
div[data-testid="stFileUploader"] label {
    display: none !important; /* Hides duplicate empty text blocks causing gaps */
}
div[data-testid="stFileUploader"] button {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* ── PREMIUM GLOWING SEAMLESS BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(52, 211, 153, 0.3) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.7px !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #34d399 0%, #10b981 50%, #059669 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.45) !important;
    border-color: #34d399 !important;
}

/* Radio */
.stRadio > label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #22c55e !important; border-bottom-color: #22c55e !important; }

/* Expander */
.streamlit-expanderHeader { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size:2.8rem; margin-bottom:0.3rem;">🌾</div>
        <div class="logo-text">CropSense AI</div>
        <div class="logo-sub">Plant Intelligence Platform</div>
    </div>
    <hr style="border-color:rgba(34,197,94,0.2); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    app_mode = st.radio(
        "Navigate",
        ["🏠  Home", "🔬  Diagnose", "📊  Disease Library", "ℹ️  About"],
        index=0
    )
    st.markdown("<hr style='border-color:rgba(34,197,94,0.1);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 1rem; background: rgba(34,197,94,0.06); border-radius:12px; border: 1px solid rgba(34,197,94,0.15);">
        <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.7rem;">Quick Stats</div>
        <div style="color:#22c55e; font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;">38 Diseases</div>
        <div style="color:#64748b; font-size:0.75rem;">Detected & Classified</div>
        <div style="color:#22c55e; font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-top:0.7rem;">98.7%</div>
        <div style="color:#64748b; font-size:0.75rem;">Validation Accuracy</div>
        <div style="color:#22c55e; font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-top:0.7rem;">14 Crops</div>
        <div style="color:#64748b; font-size:0.75rem;">Supported Species</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem; font-size:0.7rem; color:#334155; text-align:center;">
        © 2025 CropSense AI · v2.0<br>Built for precision agriculture
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE ─────────────────────────────────────────────────────────────────
if "Home" in app_mode:

    st.markdown("""
    <div class="animate d1" style="padding: 1.5rem 0 0.5rem 0;">
        <div style="font-family:'Syne',sans-serif; font-size:0.75rem; letter-spacing:4px; color:#22c55e; text-transform:uppercase; margin-bottom:0.8rem;">
            AI-Powered Precision Agriculture
        </div>
        <h1 style="font-size:2.8rem; font-weight:800; line-height:1.1; margin-bottom:0.8rem;">
            Detect Crop Diseases<br>
            <span style="background:linear-gradient(135deg,#22c55e,#06b6d4); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Before They Spread
            </span>
        </h1>
        <p style="font-size:1.05rem; color:#94a3b8; max-width:600px; line-height:1.7;">
            Upload a leaf image and get instant AI diagnosis across 38 diseases in 14 crop species — 
            powered by deep convolutional neural networks trained on 87,000+ images.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hero image
    try:
        st.image("homeIMG.jpg", use_container_width=True, caption="")
    except:
        pass

    st.markdown("<hr class='green-divider'>", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("98.7%", "Model Accuracy"),
        ("87K+", "Training Images"),
        ("38", "Disease Classes"),
        ("<5s", "Diagnosis Time"),
    ]
    for col, (num, lbl) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div class="stat-box animate d2">
            <div class="stat-number">{num}</div>
            <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown('<h2 style="font-size:1.5rem; margin-bottom:1rem;">How It Works</h2>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📸", "Capture", "Take a clear, well-lit photo of the affected plant leaf"),
        ("⬆️", "Upload", "Go to Diagnose and upload your JPG or PNG image"),
        ("🧠", "Analyze", "Our CNN model processes and classifies the disease instantly"),
        ("💊", "Treat", "Receive diagnosis, severity rating, and treatment steps"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        col.markdown(f"""
        <div class="feature-card animate d3">
            <div style="font-size:1.8rem; margin-bottom:0.6rem;">{icon}</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#f0fdf4; margin-bottom:0.4rem;">{title}</div>
            <div style="font-size:0.82rem; color:#64748b; line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Supported crops
    st.markdown('<h2 style="font-size:1.5rem; margin-bottom:0.8rem;">Supported Crops</h2>', unsafe_allow_html=True)
    crops = ["🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn", "🍇 Grape",
             "🍊 Orange", "🍑 Peach", "🫑 Bell Pepper", "🥔 Potato",
             "🍓 Strawberry", "🍅 Tomato", "🫐 Raspberry", "🌱 Soybean", "🎃 Squash"]
    crop_html = "".join([
        f'<span style="display:inline-block; background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2); border-radius:20px; padding:0.35rem 0.9rem; margin:0.25rem; font-size:0.83rem; color:#94a3b8;">{c}</span>'
        for c in crops
    ])
    st.markdown(f'<div>{crop_html}</div>', unsafe_allow_html=True)

# ── DIAGNOSE PAGE ─────────────────────────────────────────────────────────────
elif "Diagnose" in app_mode:

    st.markdown("""
    <h1 style="font-size:2.2rem; margin-bottom:0.3rem;">🔬 Disease Diagnosis</h1>
    <p style="color:#64748b; margin-bottom:1.5rem;">Upload a clear leaf image for instant AI-powered analysis</p>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size:1.1rem; margin-bottom:1.5rem;">📤 Upload Leaf Image</h3>', unsafe_allow_html=True)

        test_image = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            help="Best results with clear, well-lit, single-leaf photos"
        )

        if test_image:
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(test_image, use_container_width=True, caption="Uploaded Image Asset")

            st.markdown("""
            <div style="margin-top:0.8rem; padding:0.7rem 1rem; background:rgba(34, 197, 94, 0.08); border-radius:10px; font-size:0.82rem; color:#cbd5e1; border: 1px solid rgba(34,197,94,0.2);">
                ✅ Image package validated — click <strong style="color:#34d399;">Analyze Leaf</strong> below
            </div>""", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips
        with st.expander("📷 Photo Tips for Best Results"):
            tips = [
                "Use natural daylight — avoid harsh shadows",
                "Keep the leaf flat and fill the frame",
                "Capture both healthy and affected areas",
                "Avoid blurry or dark images",
                "One leaf per photo for accurate results"
            ]
            for t in tips:
                st.markdown(f"• {t}")

    with col_result:
        if test_image:
            if st.button("Analyze Leaf", type="primary", use_container_width=True):
                # ── Step 1: Validate it's actually a leaf ──────────────────
                with st.spinner("Checking image structure..."):
                    valid, reason = is_leaf_image(test_image)

                if not valid:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
                        border: 1px solid rgba(239,68,68,0.4);
                        border-radius: 16px;
                        padding: 1.8rem;
                        text-align: center;
                        margin-top: 1rem;
                    ">
                        <div style="font-size:3rem; margin-bottom:0.8rem;">🚫</div>
                        <h3 style="color:#fca5a5 !important; font-size:1.2rem; margin-bottom:0.6rem;">
                            Not a Plant Leaf Image
                        </h3>
                        <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:1rem; line-height:1.7;">
                            {reason}
                        </p>
                        <div style="
                            background: rgba(255,255,255,0.04);
                            border-radius: 12px;
                            padding: 1rem;
                            text-align: left;
                        ">
                            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.7rem;">
                                📷 Tips for a valid image
                            </div>
                            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.9;">
                                ✅ &nbsp; Use a photo of a <strong style="color:#e2e8f0;">single plant leaf</strong><br>
                                ✅ &nbsp; Ensure <strong style="color:#e2e8f0;">good lighting</strong> — natural daylight works best<br>
                                ✅ &nbsp; The leaf should <strong style="color:#e2e8f0;">fill most of the frame</strong><br>
                                ✅ &nbsp; Supported crops: Apple, Corn, Tomato, Potato, Grape &amp; more<br>
                                ❌ &nbsp; Do not upload people, animals, food, or unrelated objects
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # ── Step 2: Run disease model ──────────────────────────
                    with st.spinner("Analyzing leaf patterns with AI..."):
                        result_index, confidence = model_prediction(test_image)
                        diagnosis = CLASS_NAMES[result_index]
                        plant, disease = diagnosis.split(" - ", 1)
                        is_healthy = "Healthy" in disease
                        disease_key = get_disease_key(disease)
                        info = DISEASE_INFO.get(disease_key, {})

                    # ── Result Header
                    if is_healthy:
                        st.markdown(f"""
                        <div class="card-green animate d1">
                            <div style="font-size:2rem; margin-bottom:0.5rem;">✅</div>
                            <h2 style="font-size:1.5rem; color:#4ade80 !important; margin:0 0 0.3rem 0;">{plant} is Healthy!</h2>
                            <p style="color:#86efac; font-size:0.9rem; margin:0;">No disease detected. Your crop looks great.</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        severity = info.get("severity", "Unknown")
                        badge_class = f"badge-{severity.lower()}" if severity in ["Critical","High","Moderate","Low"] else "badge-moderate"
                        color = info.get("color", "#f59e0b")
                        st.markdown(f"""
                        <div class="card-red animate d1">
                            <div style="font-size:2rem; margin-bottom:0.5rem;">⚠️</div>
                            <div style="margin-bottom:0.5rem;"><span class="badge {badge_class}">{severity} Severity</span></div>
                            <h2 style="font-size:1.4rem; color:#fca5a5 !important; margin:0 0 0.3rem 0;">{disease}</h2>
                            <p style="color:#94a3b8; font-size:0.88rem; margin:0;">Detected in: <strong style="color:#e2e8f0;">{plant}</strong></p>
                        </div>""", unsafe_allow_html=True)

                    # ── Confidence Meter
                    conf_color = "#22c55e" if confidence > 80 else "#f59e0b" if confidence > 60 else "#ef4444"
                    st.markdown(f"""
                    <div class="card animate d2" style="padding:1rem 1.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:1px;">Confidence Score</span>
                            <span style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:{conf_color};">{confidence:.1f}%</span>
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{confidence}%; background: linear-gradient(90deg, {conf_color}, {conf_color}aa);"></div>
                        </div>
                        <div style="font-size:0.75rem; color:#475569; margin-top:0.3rem;">
                            {"High confidence — reliable diagnosis" if confidence > 80 else "Moderate confidence — consider retaking photo" if confidence > 60 else "Low confidence — please retake with better lighting"}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # ── Disease Details (only if diseased)
                    if not is_healthy and info:
                        tabs = st.tabs(["🧪 Disease Info", "💊 Treatment", "🛡️ Prevention"])

                        with tabs[0]:
                            st.markdown(f"""
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-top:0.5rem;">
                                <div style="padding:0.8rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                                    <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Cause</div>
                                    <div style="font-size:0.88rem; color:#e2e8f0;">{info.get('cause','—')}</div>
                                </div>
                                <div style="padding:0.8rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                                    <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Severity</div>
                                    <div style="font-size:0.88rem; color:#e2e8f0;">{info.get('severity','—')}</div>
                                </div>
                            </div>
                            <div style="margin-top:0.8rem; padding:0.8rem 1rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                                <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">Key Symptoms</div>
                                <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.6;">{info.get('symptoms','—')}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with tabs[1]:
                            st.markdown('<div style="margin-top:0.5rem;">', unsafe_allow_html=True)
                            for i, step in enumerate(info.get("treatment", []), 1):
                                st.markdown(f"""
                                <div class="step-item">
                                    <div class="step-num">{i}</div>
                                    <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.5;">{step}</div>
                                </div>""", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with tabs[2]:
                            st.markdown(f"""
                            <div style="margin-top:0.5rem; padding:1rem; background:rgba(34,197,94,0.06); border-radius:12px; border: 1px solid rgba(34,197,94,0.15);">
                                <div style="font-size:0.75rem; color:#22c55e; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.6rem;">🛡️ Prevention Strategy</div>
                                <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.7;">{info.get('prevention','Maintain good agricultural practices and regular monitoring.')}</div>
                            </div>""", unsafe_allow_html=True)

                    elif is_healthy:
                        st.markdown("""
                        <div class="card-green" style="margin-top:0.5rem;">
                            <h4 style="color:#4ade80 !important; margin-bottom:0.7rem;">🌱 Maintenance Tips</h4>
                            <div class="step-item"><div class="step-num">✓</div><div style="font-size:0.85rem; color:#cbd5e1;">Continue regular watering and fertilization schedule</div></div>
                            <div class="step-item"><div class="step-num">✓</div><div style="font-size:0.85rem; color:#cbd5e1;">Monitor for early signs of pest activity</div></div>
                            <div class="step-item"><div class="step-num">✓</div><div style="font-size:0.85rem; color:#cbd5e1;">Maintain adequate spacing between plants for airflow</div></div>
                            <div class="step-item"><div class="step-num">✓</div><div style="font-size:0.85rem; color:#cbd5e1;">Apply preventive fungicide during high-humidity seasons</div></div>
                        </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="display:flex; align-items:center; justify-content:center; height:300px; border:2px dashed rgba(34,197,94,0.15); border-radius:16px; flex-direction:column; gap:1rem;">
                <div style="font-size:3rem;">🌾</div>
                <div style="color:#334155; font-size:0.9rem; text-align:center;">Upload a leaf image on the left<br>to start diagnosis</div>
            </div>""", unsafe_allow_html=True)

# ── DISEASE LIBRARY ───────────────────────────────────────────────────────────
elif "Library" in app_mode:

    st.markdown("""
    <h1 style="font-size:2.2rem; margin-bottom:0.3rem;">📊 Disease Library</h1>
    <p style="color:#64748b; margin-bottom:1.5rem;">Reference guide for all 38 detectable diseases</p>
    """, unsafe_allow_html=True)

    # Group by plant
    plant_diseases = {}
    for cls in CLASS_NAMES:
        plant, disease = cls.split(" - ", 1)
        if plant not in plant_diseases:
            plant_diseases[plant] = []
        plant_diseases[plant].append(disease)

    # Filter
    col_s, col_f = st.columns([2, 1])
    with col_s:
        search = st.text_input("🔍 Search disease or crop", placeholder="e.g. Tomato, Blight, Rust...")
    with col_f:
        filter_type = st.selectbox("Filter", ["All", "Diseases Only", "Healthy Only"])

    for plant, diseases in sorted(plant_diseases.items()):
        filtered = []
        for d in diseases:
            if search and search.lower() not in plant.lower() and search.lower() not in d.lower():
                continue
            if filter_type == "Diseases Only" and "Healthy" in d:
                continue
            if filter_type == "Healthy Only" and "Healthy" not in d:
                continue
            filtered.append(d)

        if not filtered:
            continue

        with st.expander(f"🌿 {plant}  ({len(filtered)} conditions)", expanded=False):
            for d in filtered:
                is_h = "Healthy" in d
                key = get_disease_key(d)
                info = DISEASE_INFO.get(key, {})
                sev = info.get("severity", "Healthy" if is_h else "—")
                badge = "badge-healthy" if is_h else f"badge-{sev.lower()}" if sev in ["Critical","High","Moderate","Low"] else "badge-moderate"
                cause = info.get("cause", "N/A")
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:0.7rem 1rem; background:rgba(255,255,255,0.03); border-radius:10px; margin-bottom:0.5rem; border:1px solid rgba(255,255,255,0.05);">
                    <div>
                        <div style="font-size:0.9rem; color:#e2e8f0; font-weight:500;">{"✅ " if is_h else "⚠️ "}{d}</div>
                        <div style="font-size:0.75rem; color:#475569; margin-top:0.2rem;">{cause}</div>
                    </div>
                    <span class="badge {badge}">{"Healthy" if is_h else sev}</span>
                </div>""", unsafe_allow_html=True)

# ── ABOUT PAGE ────────────────────────────────────────────────────────────────
elif "About" in app_mode:

    st.markdown("""
    <h1 style="font-size:2.2rem; margin-bottom:0.3rem;">ℹ️ About CropSense AI</h1>
    <p style="color:#64748b; margin-bottom:1.5rem;">Precision plant disease intelligence for modern agriculture</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="font-size:1.1rem; margin-bottom:0.8rem; color:#22c55e !important;">🌐 Mission</h3>
            <p style="color:#94a3b8; line-height:1.8; font-size:0.9rem;">
                CropSense AI bridges the gap between cutting-edge deep learning research and 
                real-world agricultural needs. Farmers and agronomists deserve tools that are 
                as powerful as they are accessible — instant, accurate, and actionable.
            </p>
        </div>

        <div class="card">
            <h3 style="font-size:1.1rem; margin-bottom:0.8rem; color:#22c55e !important;">📊 Dataset</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.7rem;">
                <div style="padding:0.7rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <div style="font-size:0.72rem; color:#64748b; margin-bottom:0.2rem;">Source</div>
                    <div style="font-size:0.85rem; color:#e2e8f0;">PlantVillage (Kaggle)</div>
                </div>
                <div style="padding:0.7rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <div style="font-size:0.72rem; color:#64748b; margin-bottom:0.2rem;">Total Images</div>
                    <div style="font-size:0.85rem; color:#e2e8f0;">87,000+ RGB</div>
                </div>
                <div style="padding:0.7rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <div style="font-size:0.72rem; color:#64748b; margin-bottom:0.2rem;">Training Set</div>
                    <div style="font-size:0.85rem; color:#e2e8f0;">70,295 images</div>
                </div>
                <div style="padding:0.7rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <div style="font-size:0.72rem; color:#64748b; margin-bottom:0.2rem;">Validation Set</div>
                    <div style="font-size:0.85rem; color:#e2e8f0;">17,572 images</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3 style="font-size:1.1rem; margin-bottom:0.8rem; color:#22c55e !important;">🛠️ Technical Stack</h3>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
        """ + "".join([
            f'<span style="background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2); border-radius:6px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#86efac;">{t}</span>'
            for t in ["TensorFlow 2.0", "Keras", "Streamlit", "Python 3.10", "CNN 16-Layer", "NumPy", "PIL", "Adam Optimizer"]
        ]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏆</div>
            <h3 style="font-size:1.1rem; color:#22c55e !important; margin-bottom:1.5rem;">Model Performance</h3>
            <div class="stat-box" style="margin-bottom:0.7rem;">
                <div class="stat-number">98.7%</div>
                <div class="stat-label">Validation Accuracy</div>
            </div>
            <div class="stat-box" style="margin-bottom:0.7rem;">
                <div class="stat-number">50</div>
                <div class="stat-label">Training Epochs</div>
            </div>
            <div class="stat-box" style="margin-bottom:0.7rem;">
                <div class="stat-number">128×128</div>
                <div class="stat-label">Input Resolution</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">&lt;5s</div>
                <div class="stat-label">Inference Time</div>
            </div>
        </div>

        <div style="margin-top:1.2rem; padding:1.2rem; background:rgba(34,197,94,0.06); border-radius:14px; border:1px solid rgba(34,197,94,0.15); text-align:center;">
            <div style="font-size:1.3rem; margin-bottom:0.4rem;">👨‍💻</div>
            <div style="font-size:0.85rem; color:#94a3b8;">Built with ❤️ for Indian farmers<br>and global agriculture</div>
            <div style="font-size:0.75rem; color:#334155; margin-top:0.6rem;">© 2025 CropSense AI · v2.0</div>
        </div>
        """, unsafe_allow_html=True)