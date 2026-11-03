"""
MamaCare Kenya — Maternal Health Risk Prediction
DSA 4900 Final Year Project
Author: SelmahT | Data: KDHS 2022
AI: Google Gemini 3.8 Flash (free tier)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings, io, datetime, textwrap
warnings.filterwarnings("ignore")

# ── Optional: Gemini ──
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# ── Optional: PDF ──
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MOUT = ROOT / "03_Models_Saved"
FOUT = ROOT / "04_Outputs" / "figures"
ROUT = ROOT / "04_Outputs" / "results"

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
TEAL       = "#0D6E6E"
CREAM      = "#F5F0E8"
AMBER      = "#E8A020"
RED_RISK   = "#C0392B"
GREEN_SAFE = "#1A7A4A"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {CREAM};
    color: #1A1A1A;
}}
[data-testid="stSidebar"] {{
    background-color: {TEAL};
}}
[data-testid="stSidebar"] * {{ color: white !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.25); }}
.block-container {{ padding-top: 1.5rem; max-width: 1100px; }}
h1 {{ font-family: 'DM Serif Display', serif; color: {TEAL}; }}
h2, h3 {{ color: {TEAL}; }}

.risk-badge {{
    display: inline-block; padding: 10px 28px; border-radius: 50px;
    font-size: 1.5rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 12px 0;
}}
.risk-low  {{ background:#D4EDDA; color:{GREEN_SAFE}; border:2px solid {GREEN_SAFE}; }}
.risk-mod  {{ background:#FFF3CD; color:#856404;      border:2px solid {AMBER}; }}
.risk-high {{ background:#FDECEA; color:{RED_RISK};   border:2px solid {RED_RISK}; }}

.card {{
    background:white; border-radius:14px; padding:22px 26px;
    box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-bottom:18px;
}}
.metric-value {{ font-size:2rem; font-weight:700; color:{TEAL}; }}
.metric-label {{ font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:.05em; }}

.disclaimer {{
    background:#FFF3CD; border-left:4px solid {AMBER};
    padding:10px 16px; border-radius:6px;
    font-size:0.82rem; color:#5A4010; margin-bottom:16px;
}}
.stButton > button {{
    background-color:{TEAL}; color:white; border:none;
    border-radius:8px; padding:10px 28px;
    font-size:1rem; font-weight:600;
}}
.stButton > button:hover {{ background-color:#0A5858; }}
thead tr th {{ background:{TEAL} !important; color:white !important; }}
</style>
"""

# ─────────────────────────────────────────────
# TRANSLATIONS  (English / Kiswahili)
# ─────────────────────────────────────────────
T = {
    "en": {
        "nav_home": "🏠 Home", "nav_assess": "📋 Risk Assessment",
        "nav_insights": "📊 Model Insights", "nav_about": "ℹ️ About",
        "disclaimer": ("⚠️ <b>Clinical Disclaimer:</b> MamaCare Kenya is a research and "
                       "educational tool only. Predictions must not replace clinical judgment. "
                       "No patient data is stored or transmitted."),
        "clinician": "Clinician", "researcher": "Researcher",
        "role_q": "Who are you?",
        "clinician_desc": "Enter patient indicators and receive a risk prediction with AI explanation.",
        "researcher_desc": "Explore model performance, SHAP explanations, and Bayesian uncertainty.",
        "enter_clinician": "Enter as Clinician", "enter_researcher": "Enter as Researcher",
        "switch_role": "🔄 Switch Role",
        "assess_title": "# 📋 Risk Assessment",
        "patient_indicators": "### Patient Indicators",
        "demographic": "**Demographic & Socioeconomic**",
        "clinical": "**Clinical & Obstetric**",
        "predict_btn": "🔍 Predict Risk Level",
        "result": "### Assessment Result",
        "predicted": "Predicted Risk Level",
        "confidence": "Model Confidence",
        "probabilities": "Class probabilities",
        "ai_summary": "### AI Clinical Summary",
        "generating": "Generating AI summary…",
        "shap_title": "### Why this prediction? — SHAP Explanation",
        "shap_desc": ("The waterfall chart shows which features pushed the prediction towards "
                      "<b>{risk}</b>. Red bars increase risk; blue bars decrease it."),
        "download_btn": "⬇ Download Referral Letter (PDF)",
        "download_note": "Includes: patient profile · risk result · AI summary · SHAP chart · referral instructions",
        "risk_low": "Low Risk", "risk_mod": "Moderate Risk", "risk_high": "High Risk",
        "referral_letter": "REFERRAL LETTER",
        "referral_to": "To: Facility In-Charge",
        "referral_from": "From: Community Health Worker",
        "referral_intro": "Please attend to the patient described below who has been assessed using the MamaCare Kenya maternal risk tool (KDHS 2022, XGBoost model).",
        "referral_action": {
            0: "Routine follow-up at nearest health centre as per ANC schedule.",
            1: "Enhanced antenatal follow-up recommended. Please review ANC history and counsel on facility delivery.",
            2: "URGENT REFERRAL. Patient presents multiple high-risk indicators. Immediate assessment at a Level 4/5 facility is strongly recommended.",
        },
    },
    "sw": {
        "nav_home": "🏠 Nyumbani", "nav_assess": "📋 Tathmini ya Hatari",
        "nav_insights": "📊 Maarifa ya Mfano", "nav_about": "ℹ️ Kuhusu",
        "disclaimer": ("⚠️ <b>Onyo la Kimatibabu:</b> MamaCare Kenya ni zana ya utafiti na elimu tu. "
                       "Matokeo hayapaswi kuchukua nafasi ya uamuzi wa kimatibabu. "
                       "Hakuna data ya mgonjwa inayohifadhiwa au kutumwa."),
        "clinician": "Kliniki", "researcher": "Mtafiti",
        "role_q": "Wewe ni nani?",
        "clinician_desc": "Ingiza viashiria vya mgonjwa na upate utabiri wa hatari na maelezo ya AI.",
        "researcher_desc": "Chunguza utendaji wa mfano, maelezo ya SHAP, na uchambuzi wa Bayesian.",
        "enter_clinician": "Ingia kama Kliniki", "enter_researcher": "Ingia kama Mtafiti",
        "switch_role": "🔄 Badilisha Jukumu",
        "assess_title": "# 📋 Tathmini ya Hatari ya Uzazi",
        "patient_indicators": "### Viashiria vya Mgonjwa",
        "demographic": "**Idadi ya watu na Hali ya Kiuchumi**",
        "clinical": "**Kliniki na Uzazi**",
        "predict_btn": "🔍 Tathmini Kiwango cha Hatari",
        "result": "### Matokeo ya Tathmini",
        "predicted": "Kiwango cha Hatari Kilichotabiriwa",
        "confidence": "Imara ya Mfano",
        "probabilities": "Uwezekano wa darasa",
        "ai_summary": "### Muhtasari wa AI wa Kimatibabu",
        "generating": "Inaunda muhtasari wa AI…",
        "shap_title": "### Kwa nini utabiri huu? — Maelezo ya SHAP",
        "shap_desc": ("Chati inaonyesha vipengele vilivyosukuma utabiri kuelekea "
                      "<b>{risk}</b>. Mistari nyekundu huongeza hatari; bluu hupunguza."),
        "download_btn": "⬇ Pakua Barua ya Rufaa (PDF)",
        "download_note": "Inajumuisha: wasifu wa mgonjwa · matokeo ya hatari · muhtasari wa AI · chati ya SHAP · maelekezo ya rufaa",
        "risk_low": "Hatari Ndogo", "risk_mod": "Hatari ya Wastani", "risk_high": "Hatari Kubwa",
        "referral_letter": "BARUA YA RUFAA",
        "referral_to": "Kwa: Msimamizi wa Kituo",
        "referral_from": "Kutoka: Mfanyakazi wa Afya ya Jamii",
        "referral_intro": "Tafadhali msaidie mgonjwa aliyeelezwa hapa chini ambaye amepimwa kwa kutumia zana ya MamaCare Kenya (KDHS 2022, mfano wa XGBoost).",
        "referral_action": {
            0: "Ufuatiliaji wa kawaida katika kituo cha afya karibu nawe kulingana na ratiba ya ANC.",
            1: "Ufuatiliaji wa ziada wa ANC unapendekezwa. Tafadhali kagua historia ya ANC na ushauri kuhusu kujifungulia kituo.",
            2: "RUFAA YA HARAKA. Mgonjwa ana viashiria vingi vya hatari kubwa. Tathmini ya haraka katika kituo cha Ngazi ya 4/5 inapendekezwa sana.",
        },
    },
}

RISK_IDX  = {0: "risk_low", 1: "risk_mod", 2: "risk_high"}
RISK_CSS  = {0: "risk-low", 1: "risk-mod",  2: "risk-high"}
RISK_ICON = {0: "✅", 1: "⚠️", 2: "🔴"}

FEATURE_CODES = ["v012","v201","v106","v190","v025","v501","v212","v024",
                 "m14_1","m15_1","m17_1","m3a_1","b11_01"]

# Only these 5 were passed through StandardScaler during training
NUMERIC_FEATURES = ["v012","v201","m14_1","b11_01","v212"]

FEAT_MAP_EN = {
    "v012":"Age of Woman","v201":"Total Children Born","v106":"Education Level",
    "v190":"Wealth Index","v025":"Residence","v501":"Marital Status",
    "v212":"Age at First Birth","v024":"Region","m14_1":"ANC Visits",
    "m15_1":"Place of Delivery","m17_1":"Caesarean Section",
    "m3a_1":"Doctor at Delivery","b11_01":"Birth Interval (months)",
}
FEAT_MAP_SW = {
    "v012":"Umri wa Mwanamke","v201":"Jumla ya Watoto","v106":"Kiwango cha Elimu",
    "v190":"Kiwango cha Utajiri","v025":"Makazi","v501":"Hali ya Ndoa",
    "v212":"Umri wa Kuzaa Kwanza","v024":"Mkoa","m14_1":"Ziara za ANC",
    "m15_1":"Mahali pa Kujifungua","m17_1":"Upasuaji wa Cesarean",
    "m3a_1":"Daktari Wakati wa Kujifungua","b11_01":"Muda Kati ya Vizazi (miezi)",
}

# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    scaler  = joblib.load(MOUT / "scaler.pkl")
    model   = joblib.load(MOUT / "xgb_model.pkl")
    results = pd.read_csv(ROUT / "all_model_results.csv")

    # Regenerate SHAP explainer at runtime to avoid pickle version conflicts
    dummy = pd.DataFrame([
        [28, 3, 2, 3, 2, 1, 20, 1, 4, 2, 0, 1, 24],
        [22, 1, 1, 1, 2, 1, 18, 3, 2, 1, 0, 0, 14],
        [35, 6, 0, 1, 2, 1, 17, 5, 1, 1, 0, 0, 10],
    ], columns=FEATURE_CODES, dtype=float)
    dummy_out = dummy.copy()
    dummy_out[NUMERIC_FEATURES] = scaler.transform(dummy[NUMERIC_FEATURES])
    explainer = shap.TreeExplainer(model)
    explainer(dummy_out.values)  # warm-up

    return scaler, model, explainer, results

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(scaler, model, values: dict):
    df = pd.DataFrame([values], columns=FEATURE_CODES)
    # Scale only the 5 numeric features — matches training pipeline
    df_out = df.copy().astype(float)
    df_out[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    df_s = df_out.values
    pred  = int(model.predict(df_s)[0])
    proba = model.predict_proba(df_s)[0]
    return pred, proba, df_s

# ─────────────────────────────────────────────
# SHAP WATERFALL
# ─────────────────────────────────────────────
def shap_waterfall(explainer, df_s, pred_class: int, lang: str):
    feat_map = FEAT_MAP_SW if lang == "sw" else FEAT_MAP_EN
    sv  = explainer(df_s)
    sv_cls = shap.Explanation(
        values      = sv.values[0, :, pred_class],
        base_values = sv.base_values[0, pred_class],
        data        = df_s[0],
        feature_names = [feat_map.get(f, f) for f in FEATURE_CODES],
    )
    shap.plots.waterfall(sv_cls, max_display=10, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    return fig, sv.values[0, :, pred_class]

# ─────────────────────────────────────────────
# AI NARRATIVE  (Gemini 1.5 Flash)
# ─────────────────────────────────────────────
def generate_narrative(risk_label: str, confidence: float,
                        top_feats: list, api_key: str, lang: str) -> str:
    pos = [f for f, v in top_feats if v > 0]
    neg = [f for f, v in top_feats if v <= 0]

    if lang == "sw":
        fallback = (
            f"Mfano unatoa utabiri wa **{risk_label}** kwa imara ya {confidence:.1f}%. "
        )
        if pos:
            fallback += f"Viashiria vya kuongeza hatari: {', '.join(pos[:3])}. "
        if neg:
            fallback += f"Viashiria vya ulinzi: {', '.join(neg[:2])}. "
        if "Kubwa" in risk_label:
            fallback += "Rufaa ya haraka kwa kituo cha ngazi ya juu inapendekezwa."
        elif "Wastani" in risk_label:
            fallback += "Ufuatiliaji wa ziada wa ANC unashauriwa."
        else:
            fallback += "Endelea na ufuatiliaji wa kawaida wa ANC."
    else:
        fallback = (
            f"The model predicts **{risk_label}** with {confidence:.1f}% confidence. "
        )
        if pos:
            fallback += f"Key risk-elevating factors: {', '.join(pos[:3])}. "
        if neg:
            fallback += f"Protective factors present: {', '.join(neg[:2])}. "
        if "High" in risk_label:
            fallback += ("Immediate referral to a Level 4/5 facility and intensified "
                         "antenatal surveillance are strongly recommended.")
        elif "Moderate" in risk_label:
            fallback += "Enhanced antenatal follow-up and facility delivery counselling are advised."
        else:
            fallback += "Continue routine antenatal monitoring per national guidelines."

    if not api_key or not _GEMINI_AVAILABLE:
        return fallback

    try:
        genai.configure(api_key=api_key)
        model_g = genai.GenerativeModel("gemini-3.8-flash")

        feat_text = "\n".join(
            f"  - {f}: {'increases' if v>0 else 'decreases'} risk (SHAP={v:+.3f})"
            for f, v in top_feats
        )
        lang_instruction = (
            "Respond in Kiswahili only." if lang == "sw"
            else "Respond in English only."
        )
        prompt = textwrap.dedent(f"""
            You are a maternal health clinical decision-support assistant serving
            Community Health Workers (CHWs) in Kenya.

            A patient was assessed by an XGBoost model trained on KDHS 2022 data:
            Prediction : {risk_label} ({confidence:.1f}% confidence)

            Top SHAP feature contributions:
            {feat_text}

            Write a concise 3-sentence clinical summary:
            1. State the risk level and confidence in plain language.
            2. Name the 2-3 most important driving factors using clinical terminology
               a CHW would understand.
            3. Give one specific, actionable recommendation aligned with Kenya MoH
               antenatal care guidelines.

            {lang_instruction}
            Do not use bullet points or markdown. Plain prose only.
        """).strip()

        response = model_g.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return fallback + f"\n\n*(AI narrative unavailable: {e})*"


# ─────────────────────────────────────────────
# REFERRAL LETTER PDF
# ─────────────────────────────────────────────
def generate_referral_pdf(patient_values: dict, risk_label: str,
                           confidence: float, proba: np.ndarray,
                           narrative: str, shap_fig: plt.Figure,
                           lang: str, pred: int) -> bytes:
    tr = T[lang]
    buf = io.BytesIO()
    feat_map = FEAT_MAP_SW if lang == "sw" else FEAT_MAP_EN

    if not _PDF_AVAILABLE:
        lines = [
            f"MAMACARE KENYA — {tr['referral_letter']}",
            f"{datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
            "", tr["referral_to"], tr["referral_from"], "",
            tr["referral_intro"], "",
            f"{tr['predicted']}: {risk_label} ({confidence:.1f}%)", "",
            narrative, "",
            tr["referral_action"][pred], "",
            "--- Patient Indicators ---",
        ]
        for k, v in patient_values.items():
            lines.append(f"  {feat_map.get(k, k)}: {v}")
        lines += ["", "DISCLAIMER: Research prototype. Not for clinical use without validation."]
        buf.write("\n".join(lines).encode())
        return buf.getvalue()

    TEAL_RL = rl_colors.HexColor("#0D6E6E")
    risk_color = {
        "Low Risk":      rl_colors.HexColor("#1A7A4A"),
        "Moderate Risk": rl_colors.HexColor("#856404"),
        "High Risk":     rl_colors.HexColor("#C0392B"),
        "Hatari Ndogo":  rl_colors.HexColor("#1A7A4A"),
        "Hatari ya Wastani": rl_colors.HexColor("#856404"),
        "Hatari Kubwa":  rl_colors.HexColor("#C0392B"),
    }.get(risk_label, TEAL_RL)

    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2.5*cm, rightMargin=2.5*cm,
                               topMargin=2.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_s = ParagraphStyle("t", parent=styles["Title"],
                              textColor=TEAL_RL, fontSize=16, spaceAfter=2)
    sub_s   = ParagraphStyle("s", parent=styles["Normal"],
                              textColor=rl_colors.grey, fontSize=9, spaceAfter=10)
    h2_s    = ParagraphStyle("h2", parent=styles["Heading2"],
                              textColor=TEAL_RL, fontSize=11, spaceBefore=12, spaceAfter=4)
    body_s  = ParagraphStyle("b",  parent=styles["Normal"], fontSize=10, leading=15)
    warn_s  = ParagraphStyle("w",  parent=styles["Normal"], fontSize=8,
                              textColor=rl_colors.HexColor("#5A4010"),
                              backColor=rl_colors.HexColor("#FFF3CD"),
                              borderPadding=6, leading=12)
    action_s = ParagraphStyle("a", parent=styles["Normal"], fontSize=10, leading=14,
                               textColor=risk_color,
                               backColor=rl_colors.HexColor("#F8F8F8"),
                               borderPadding=8)

    story.append(Paragraph("MamaCare Kenya", title_s))
    story.append(Paragraph(
        f"{tr['referral_letter']} &nbsp;·&nbsp; "
        f"{datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
        sub_s))

    story.append(Paragraph(tr["referral_to"],   body_s))
    story.append(Paragraph(tr["referral_from"], body_s))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(tr["referral_intro"], body_s))
    story.append(Spacer(1, 0.3*cm))

    badge_data = [[
        Paragraph(f"<b>{risk_label.upper()}</b>",
                  ParagraphStyle("bd", fontSize=13, textColor=risk_color,
                                 alignment=TA_CENTER)),
        Paragraph(f"<b>{confidence:.1f}%</b>",
                  ParagraphStyle("cd", fontSize=12, textColor=TEAL_RL,
                                 alignment=TA_CENTER)),
        Paragraph(
            f"<font size='8'>Low {proba[0]*100:.0f}% | "
            f"Mod {proba[1]*100:.0f}% | High {proba[2]*100:.0f}%</font>",
            ParagraphStyle("pd", fontSize=8, alignment=TA_CENTER)),
    ]]
    badge_tbl = Table(badge_data, colWidths=[5.5*cm, 3.5*cm, 5*cm])
    badge_tbl.setStyle(TableStyle([
        ("BOX",           (0,0), (-1,-1), 1, TEAL_RL),
        ("LINEAFTER",     (0,0), (1,0),   0.4, rl_colors.lightgrey),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [rl_colors.HexColor("#F5F0E8")]),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(badge_tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("AI Clinical Summary" if lang=="en" else "Muhtasari wa AI", h2_s))
    story.append(Paragraph(narrative, body_s))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Recommended Action" if lang=="en" else "Hatua Inayopendekezwa", h2_s))
    story.append(Paragraph(f"<b>{tr['referral_action'][pred]}</b>", action_s))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Patient Indicators" if lang=="en" else "Viashiria vya Mgonjwa", h2_s))
    tdata = [["Indicator" if lang=="en" else "Kiashiria",
              "Value" if lang=="en" else "Thamani"]]
    for k, v in patient_values.items():
        tdata.append([feat_map.get(k, k), str(v)])
    t = Table(tdata, colWidths=[9*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), TEAL_RL),
        ("TEXTCOLOR",     (0,0), (-1,0), rl_colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor("#F5F0E8")]),
        ("GRID",          (0,0), (-1,-1), 0.4, rl_colors.lightgrey),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("SHAP Explanation" if lang=="en" else "Maelezo ya SHAP", h2_s))
    shap_buf = io.BytesIO()
    shap_fig.savefig(shap_buf, format="png", dpi=120, bbox_inches="tight")
    shap_buf.seek(0)
    story.append(RLImage(shap_buf, width=14*cm, height=7.5*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "DISCLAIMER: MamaCare Kenya is a research prototype (DSA 4900, USIU-Africa, 2026). "
        "This letter is AI-generated and must be reviewed by a qualified healthcare provider "
        "before clinical action. No patient data is stored or transmitted.",
        warn_s))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [("role", None), ("researcher_agreed", False),
                     ("lang", "en"), ("gemini_key", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# Pre-fill Gemini key from Streamlit Cloud secrets if available
if not st.session_state["gemini_key"]:
    try:
        st.session_state["gemini_key"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

lang = st.session_state["lang"]
tr   = T[lang]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
        <span style='font-size:2.4rem;'>🌿</span>
        <h2 style='margin:4px 0 0; font-family:DM Serif Display; font-size:1.4rem;'>MamaCare Kenya</h2>
        <p style='font-size:0.78rem; opacity:0.8; margin:2px 0 0;'>Maternal Risk Intelligence</p>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate",
                    [tr["nav_home"], tr["nav_assess"], tr["nav_insights"], tr["nav_about"]],
                    label_visibility="collapsed")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.78rem; opacity:0.85; margin-bottom:4px;'>🌍 Language / Lugha</p>",
                unsafe_allow_html=True)
    new_lang = st.radio("lang", ["English", "Kiswahili"],
                        index=0 if lang == "en" else 1,
                        label_visibility="collapsed",
                        horizontal=True)
    if (new_lang == "English") != (lang == "en"):
        st.session_state["lang"] = "en" if new_lang == "English" else "sw"
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.78rem; opacity:0.85; margin-bottom:4px;'>🤖 Gemini API Key</p>",
                unsafe_allow_html=True)
    key_input = st.text_input("Gemini key", type="password",
                               placeholder="AIza…",
                               value=st.session_state["gemini_key"],
                               label_visibility="collapsed")
    if key_input != st.session_state["gemini_key"]:
        st.session_state["gemini_key"] = key_input
    if key_input:
        st.markdown("<p style='font-size:0.7rem; color:#90EE90;'>✓ Key saved — AI summaries active</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:0.7rem; opacity:0.65;'>No key → rule-based summary<br/>"
                    "<a href='https://aistudio.google.com/apikey' "
                    "style='color:#90EE90;'>Get free key →</a></p>",
                    unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; opacity:0.75; line-height:1.8;'>
        <b>Data:</b> KDHS 2022<br>
        <b>Model:</b> XGBoost + SHAP<br>
        <b>AI:</b> Gemini 3.8 Flash<br>
        <b>Author:</b> SelmahT · DSA 4900
    </div>
    """, unsafe_allow_html=True)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────
if page in (T["en"]["nav_home"], T["sw"]["nav_home"]):

    if st.session_state.role is None:
        st.markdown("""
        <div style='text-align:center; padding:40px 0 20px;'>
            <h1 style='font-size:3rem; margin-bottom:6px;'>MamaCare Kenya</h1>
            <p style='font-size:1.1rem; color:#444; max-width:560px; margin:auto;'>
                AI-powered maternal health risk stratification for Kenyan Community Health
                Workers and researchers — grounded in KDHS 2022 population data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Hero image — centred, not full-bleed
        col_l, col_c, col_r = st.columns([1, 6, 1])
        with col_c:
            st.image(
                "https://images.unsplash.com/photo-1551601651-2a8555f1a136?w=900&h=360&q=80&fit=crop",
                width="stretch",
                caption="Maternal healthcare in Kenya · Photo: Unsplash (free licence)"
            )

        st.markdown(f"<br/><h3>{tr['role_q']}</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='card' style='text-align:center; border-top:4px solid {TEAL};'>
                <div style='font-size:2rem;'>🩺</div>
                <h3 style='margin:8px 0 4px;'>{tr['clinician']}</h3>
                <p style='font-size:0.88rem; color:#555;'>{tr['clinician_desc']}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(tr["enter_clinician"], use_container_width=True):
                st.session_state.role = "clinician"; st.rerun()

        with col2:
            st.markdown(f"""
            <div class='card' style='text-align:center; border-top:4px solid {AMBER};'>
                <div style='font-size:2rem;'>🔬</div>
                <h3 style='margin:8px 0 4px;'>{tr['researcher']}</h3>
                <p style='font-size:0.88rem; color:#555;'>{tr['researcher_desc']}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(tr["enter_researcher"], use_container_width=True):
                st.session_state.role = "researcher"; st.rerun()

    else:
        role_label = tr["clinician"] if st.session_state.role == "clinician" else tr["researcher"]
        st.markdown(f"""
        <div style='text-align:center; padding:40px 0 20px;'>
            <h1 style='font-size:2.8rem;'>MamaCare Kenya</h1>
            <p style='color:#444;'>Logged in as <b>{role_label}</b> · KDHS 2022 · XGBoost + Gemini AI</p>
        </div>""", unsafe_allow_html=True)

        col_l, col_c, col_r = st.columns([1, 6, 1])
        with col_c:
            st.image(
                "https://images.unsplash.com/photo-1551601651-2a8555f1a136?w=900&h=360&q=80&fit=crop",
                width="stretch",
                caption="Maternal healthcare in Kenya · Photo: Unsplash"
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, val, lbl in [
            (col1, "9,054", "Training women (KDHS 2022)"),
            (col2, "99.9%", "High-Risk Recall (XGBoost)"),
            (col3, "7",     "ML models benchmarked"),
        ]:
            with col:
                st.markdown(f"""
                <div class='card'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='disclaimer'>{tr['disclaimer']}</div>",
                    unsafe_allow_html=True)

        if st.button(tr["switch_role"]):
            st.session_state.role = None
            st.session_state.researcher_agreed = False
            st.rerun()


# ─────────────────────────────────────────────
# PAGE 2 — RISK ASSESSMENT
# ─────────────────────────────────────────────
elif page in (T["en"]["nav_assess"], T["sw"]["nav_assess"]):

    scaler, model, explainer, _ = load_artifacts()
    st.markdown(tr["assess_title"])
    st.markdown(f"<div class='disclaimer'>{tr['disclaimer']}</div>",
                unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown(tr["patient_indicators"])
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(tr["demographic"])
            v012  = st.number_input("Age / Umri (yrs)", 15, 49, 28)
            v201  = st.number_input("Total Children / Jumla ya Watoto", 0, 15, 3)
            v212  = st.number_input("Age at First Birth / Umri wa Kuzaa Kwanza", 10, 45, 20)
            v106  = st.selectbox("Education / Elimu", [0,1,2,3],
                                  format_func=lambda x: {0:"None/Hakuna",1:"Primary/Msingi",
                                                         2:"Secondary/Sekondari",3:"Higher/Zaidi"}[x])
            v190  = st.selectbox("Wealth Index / Utajiri", [1,2,3,4,5],
                                  format_func=lambda x: {1:"Poorest/Maskini Sana",2:"Poorer/Maskini",
                                                         3:"Middle/Kati",4:"Richer/Tajiri",
                                                         5:"Richest/Tajiri Sana"}[x])
            v025  = st.selectbox("Residence / Makazi", [1,2],
                                  format_func=lambda x: {1:"Urban/Mjini",2:"Rural/Vijijini"}[x])
            v501  = st.selectbox("Marital Status / Ndoa", [0,1,2,3,4,5],
                                  format_func=lambda x: {0:"Never/Hajaoa",1:"Married/Ameoa",
                                                         2:"Cohabiting/Pamoja",3:"Widowed/Mjane",
                                                         4:"Divorced/Talaka",5:"Separated/Ameachana"}[x])
            v024  = st.selectbox("Region / Mkoa", list(range(1,9)),
                                  format_func=lambda x: {1:"Nairobi",2:"Central",3:"Coast/Pwani",
                                                         4:"Eastern/Mashariki",5:"Nyanza",
                                                         6:"Rift Valley",7:"Western/Magharibi",
                                                         8:"North Eastern/Kaskazini Mashariki"}[x])
        with col2:
            st.markdown(tr["clinical"])
            m14_1 = st.number_input("ANC Visits / Ziara za ANC", 0, 20, 4)
            m15_1 = st.selectbox("Delivery Place / Mahali pa Kujifungua", [1,2,3,4,5,6],
                                  format_func=lambda x: {1:"Home/Nyumbani",
                                                         2:"Gov't Hospital/Hospitali ya Serikali",
                                                         3:"Gov't Health Centre/Kituo",
                                                         4:"Gov't Post/Zahanati",
                                                         5:"Private Hospital/Hospitali Binafsi",
                                                         6:"Other/Nyingine"}[x])
            m17_1 = st.selectbox("Caesarean / Upasuaji", [0,1],
                                  format_func=lambda x: {0:"No/Hapana",1:"Yes/Ndiyo"}[x])
            m3a_1 = st.selectbox("Doctor at Delivery / Daktari", [0,1],
                                  format_func=lambda x: {0:"No/Hapana",1:"Yes/Ndiyo"}[x])
            b11_01 = st.number_input("Birth Interval (months) / Muda (miezi)", 0, 120, 24)

        submitted = st.form_submit_button(tr["predict_btn"], use_container_width=True)

    if submitted:
        values = {"v012":v012,"v201":v201,"v106":v106,"v190":v190,
                  "v025":v025,"v501":v501,"v212":v212,"v024":v024,
                  "m14_1":m14_1,"m15_1":m15_1,"m17_1":m17_1,
                  "m3a_1":m3a_1,"b11_01":b11_01}

        pred, proba, df_s = predict(scaler, model, values)
        risk_key   = RISK_IDX[pred]
        risk_label = tr[risk_key]
        risk_css   = RISK_CSS[pred]
        conf       = proba[pred] * 100

        st.markdown("---")
        st.markdown(tr["result"])

        with st.spinner("Computing SHAP & AI summary…"):
            fig, shap_vals = shap_waterfall(explainer, df_s, pred, lang)
            feat_map = FEAT_MAP_SW if lang=="sw" else FEAT_MAP_EN
            feat_names = [feat_map.get(f, f) for f in FEATURE_CODES]
            top_idx  = np.argsort(np.abs(shap_vals))[::-1][:5]
            top_feats = [(feat_names[i], float(shap_vals[i])) for i in top_idx]
            narrative = generate_narrative(risk_label, conf, top_feats,
                                           st.session_state["gemini_key"], lang)

        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            st.markdown(f"""
            <div class='card' style='text-align:center;'>
                <p style='font-size:0.78rem; color:#666; margin-bottom:4px; text-transform:uppercase;'>
                    {tr['predicted']}
                </p>
                <div class='risk-badge {risk_css}'>{RISK_ICON[pred]} {risk_label}</div>
                <br/>
                <div class='metric-value' style='font-size:2.4rem;'>{conf:.1f}%</div>
                <div class='metric-label'>{tr['confidence']}</div>
                <br/>
                <div style='font-size:0.78rem; color:#666;'>
                    {tr['probabilities']}<br/>
                    Low {proba[0]*100:.1f}% · Mod {proba[1]*100:.1f}% · High {proba[2]*100:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

        with rcol2:
            ai_mode = ("Gemini 3.8 Flash" if (st.session_state["gemini_key"] and _GEMINI_AVAILABLE)
                       else "Rule-based")
            border_color = RED_RISK if pred==2 else AMBER if pred==1 else GREEN_SAFE
            st.markdown(f"""
            <div class='card' style='border-left:4px solid {border_color};'>
                <p style='font-size:0.72rem; color:#888; margin:0 0 8px;'>
                    🤖 {ai_mode} · {datetime.datetime.now().strftime('%H:%M')}
                </p>
                <p style='font-size:0.95rem; line-height:1.75; margin:0;'>{narrative}</p>
                <hr style='border-color:#eee; margin:12px 0 8px;'/>
                <p style='font-size:0.8rem; font-weight:600; color:{border_color}; margin:0;'>
                    {tr['referral_action'][pred]}
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown(tr["shap_title"])
        st.markdown(f"""
        <div style='background:white; border-radius:10px; padding:10px 14px; margin-bottom:12px;
                    font-size:0.83rem; color:#555;'>
            {tr['shap_desc'].format(risk=risk_label)}
        </div>""", unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        with st.spinner("Preparing referral letter…"):
            pdf_bytes = generate_referral_pdf(
                patient_values=values, risk_label=risk_label,
                confidence=conf, proba=proba, narrative=narrative,
                shap_fig=fig, lang=lang, pred=pred)
        plt.close(fig)

        ext  = "pdf" if _PDF_AVAILABLE else "txt"
        mime = "application/pdf" if _PDF_AVAILABLE else "text/plain"
        fname = f"mamacare_referral_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"

        st.download_button(label=tr["download_btn"], data=pdf_bytes,
                           file_name=fname, mime=mime, use_container_width=True)
        st.markdown(f"<p style='font-size:0.78rem; color:#888;'>{tr['download_note']}</p>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────
elif page in (T["en"]["nav_insights"], T["sw"]["nav_insights"]):

    if st.session_state.role == "researcher" and not st.session_state.researcher_agreed:
        st.markdown("## 🔬 Researcher Access")
        st.markdown("""
        <div class='card'>
            <h3 style='margin-top:0;'>Data Use Agreement</h3>
            <ul style='line-height:2;'>
                <li>This prototype is for <b>academic research only</b> — not validated for clinical deployment.</li>
                <li>KDHS 2022 data was obtained under DHS Program licence and may not be redistributed.</li>
                <li>Model outputs must not be used to identify, profile, or discriminate against individuals.</li>
                <li>Findings should be reported with uncertainty (Bayesian HDI is provided for this purpose).</li>
                <li>You will cite this work appropriately in any publication or presentation.</li>
            </ul>
        </div>""", unsafe_allow_html=True)
        agreed = st.checkbox("I have read and agree to the data use terms above.")
        if st.button("Access Model Insights", disabled=not agreed):
            st.session_state.researcher_agreed = True; st.rerun()
        st.stop()

    scaler, model, explainer, results_df = load_artifacts()
    st.markdown("# 📊 Model Insights")

    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🔍 Global SHAP", "📐 Bayesian Uncertainty"])

    with tab1:
        st.markdown("### Model Performance Leaderboard")
        data = {
            "Model": ["XGBoost","CatBoost","Random Forest","TabTransformer",
                      "TabNet","Decision Tree","Logistic Regression"],
            "Accuracy": [0.9994,0.9987,0.9968,0.9962,0.9897,0.9423,0.8114],
            "Macro F1": [0.9992,0.9985,0.9962,0.9958,0.9884,0.9388,0.8109],
            "High-Risk F1": [0.9985,0.9970,0.9940,0.9940,0.9821,0.9292,0.8314],
            "AUC": ["1.000","1.000","1.000","0.9997","0.9997","—","—"],
        }
        df_lb = pd.DataFrame(data)
        st.dataframe(df_lb, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class='card' style='margin-top:14px;'>
            <h4 style='margin-top:0;'>Why XGBoost for deployment?</h4>
            <ul style='line-height:1.9; font-size:0.9rem;'>
                <li><b>Performance:</b> Best High-Risk F1 (0.9985) and AUC (1.000)</li>
                <li><b>Explainability:</b> Exact SHAP via TreeExplainer — no approximation</li>
                <li><b>Speed:</b> Sub-millisecond inference — ideal for CHW real-time use</li>
                <li><b>Uncertainty:</b> Bayesian HDI [0.9909, 0.9995] at 94% credibility</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Global Feature Importance (SHAP) — High-Risk Class")
        col1, col2 = st.columns(2)
        for col, path, cap in [
            (col1, FOUT/"shap_global_importance.png",   "Mean |SHAP| — High Risk"),
            (col2, FOUT/"shap_beeswarm_high_risk.png",  "SHAP Beeswarm — High Risk"),
        ]:
            with col:
                if path.exists():
                    st.image(str(path), caption=cap, width="stretch")
                else:
                    st.info(f"Run Notebook 07 to generate {path.name}")
        st.markdown("""
        <div class='card'>
            <b>Top finding:</b> Total Children Born, Wealth Index, and ANC Visits are the
            three strongest predictors (Mean |SHAP| ≈ 3.97–3.98). High parity and low
            wealth consistently elevate risk; ≥4 ANC visits and a skilled birth attendant
            are strongly protective.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### Bayesian Credible Intervals — 94% HDI")
        bayes_img = FOUT / "bayesian_posterior_f1.png"
        if bayes_img.exists():
            st.image(str(bayes_img), caption="Posterior F1 — 94% HDI", width="stretch")
        else:
            st.info("Run Notebook 06 to generate bayesian_posterior_f1.png")
        st.dataframe(pd.DataFrame({
            "Model": ["XGBoost","TabTransformer","Stacking Ensemble"],
            "Mean F1": [0.9955, 0.9911, 0.9955],
            "HDI Lower (94%)": [0.9909, 0.9841, 0.9909],
            "HDI Upper (94%)": [0.9995, 0.9969, 0.9995],
            "R̂": ["1.00","1.00","1.00"], "ESS": [">1800",">1800",">1800"],
        }), use_container_width=True, hide_index=True)
        st.markdown("""
        <div class='card'>
            All R̂ = 1.00 and ESS > 1800 confirm perfect MCMC convergence.
            The 94% HDI for XGBoost's High-Risk F1 = [0.9909, 0.9995] —
            we are 94% credibly certain the true population performance lies in this range.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────
elif page in (T["en"]["nav_about"], T["sw"]["nav_about"]):

    st.markdown("# ℹ️ About MamaCare Kenya")

    col_l, col_c, col_r = st.columns([1, 6, 1])
    with col_c:
        st.image(
            "https://images.unsplash.com/photo-1584515933487-779824d29309?w=900&h=360&q=80&fit=crop",
            width="stretch",
            caption="Healthcare workers in Kenya · Photo: Unsplash (free licence)"
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='margin-top:0;'>The Dataset — KDHS 2022</h3>
            <p style='font-size:0.9rem; line-height:1.8;'>
                The <b>Kenya Demographic and Health Survey (KDHS) 2022</b> is a nationally
                representative household survey. The Individual Recode file contains
                <b>10,391 women</b> aged 15–49. After preprocessing and SMOTE oversampling,
                the training set comprised <b>9,054 observations</b> across three balanced
                risk classes (Low · Moderate · High).
            </p>
        </div>
        <div class='card'>
            <h3 style='margin-top:0;'>Risk Classification Logic</h3>
            <p style='font-size:0.87rem;'>Eight WHO-based binary clinical risk flags (f1–f8):</p>
            <ul style='font-size:0.86rem; line-height:2;'>
                <li><b>f1</b> — Adolescent (&lt;18) or advanced maternal age (&gt;35)</li>
                <li><b>f2</b> — Grand multipara (≥5 births)</li>
                <li><b>f3</b> — Short birth interval (&lt;24 months)</li>
                <li><b>f4</b> — Fewer than 4 ANC visits</li>
                <li><b>f5</b> — No skilled birth attendant</li>
                <li><b>f6</b> — Home delivery</li>
                <li><b>f7</b> — Poorest wealth quintile</li>
                <li><b>f8</b> — No formal education</li>
            </ul>
            <p style='font-size:0.87rem;'>
                Composite score →
                <b style='color:#1A7A4A;'>Low (0–1)</b> ·
                <b style='color:#856404;'>Moderate (2–3)</b> ·
                <b style='color:#C0392B;'>High (≥4)</b>
            </p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='margin-top:0;'>AI Integration</h3>
            <p style='font-size:0.9rem; line-height:1.8;'>
                <b>Google Gemini 3.8 Flash</b> generates personalised clinical narratives
                in <b>English or Kiswahili</b> from each patient's top SHAP drivers.
                The prompt is tailored for Community Health Workers and references
                Kenya MoH antenatal care guidelines.<br/><br/>
                If no API key is provided, a deterministic rule-based summary is used —
                the app remains fully functional in offline/low-connectivity settings.
            </p>
        </div>
        <div class='card'>
            <h3 style='margin-top:0;'>Privacy & Ethics</h3>
            <p style='font-size:0.87rem; line-height:1.8;'>
                <b>No real patient data is stored.</b> All form inputs are processed
                in-memory and discarded on session end.<br/><br/>
                KDHS 2022 data was accessed under the DHS Program standard academic
                licence. This tool is a <b>research prototype only</b> and must not be
                used for clinical decisions without appropriate validation.<br/><br/>
                Researcher access is gated behind an explicit data-use agreement.
            </p>
        </div>
        <div class='card'>
            <h3 style='margin-top:0;'>Author</h3>
            <p style='font-size:0.9rem; line-height:1.9;'>
                <b>Selmah Tzindori</b><br/>
                B.Sc. Data Science &amp; Analytics · USIU-Africa<br/>
                Final Year Research Project · 2026<br/><br/>
                <a href='https://github.com/SelmahT' style='color:#0D6E6E;'>GitHub: SelmahT</a>
            </p>
        </div>""", unsafe_allow_html=True)