# ─────────────────────────────────────────────────────────────
# 08_Pilot_App.py
# Maternal Health Risk Prediction — CHW Decision Support App
# DSA 3900 | USIU-Africa | Selmah Tzindori Mise | ID: 669602
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import requests
import json
import os
import sys

# ── Add paths ─────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Maternal Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
.risk-high   { background:#FDECEA; border-left:5px solid #E53935;
               padding:12px; border-radius:6px; }
.risk-mod    { background:#FFF3E0; border-left:5px solid #FB8C00;
               padding:12px; border-radius:6px; }
.risk-low    { background:#E8F5E9; border-left:5px solid #43A047;
               padding:12px; border-radius:6px; }
.metric-box  { background:#F5F5F5; padding:10px;
               border-radius:6px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__),
                            '../../03_Models_Saved')
RISK_NAMES  = {0:'Low Risk', 1:'Moderate Risk', 2:'High Risk'}
RISK_COLORS = {0:'risk-low', 1:'risk-mod', 2:'risk-high'}
RISK_ICONS  = {0:'🟢', 1:'🟡', 2:'🔴'}
CAT_COLS    = ['v106','v190','v025','v501',
               'v024','m15_1','m17_1','m3a_1']
CONT_COLS   = ['v012','v201','v212','m14_1','b11_01']
ALL_COLS    = ['v012','v201','v106','v190','v025','v501',
               'v212','v024','m14_1','m15_1','m17_1',
               'm3a_1','b11_01']

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    cat_model    = joblib.load(f'{MODELS_DIR}/cat_model.pkl')
    xgb_model    = joblib.load(f'{MODELS_DIR}/xgb_model.pkl')
    meta_learner = joblib.load(
        f'{MODELS_DIR}/stacking_meta_learner_weighted.pkl')
    scaler       = joblib.load(f'{MODELS_DIR}/scaler.pkl')

    # Rebuild TabTransformer
    cats = [4, 6, 3, 6, 9, 22, 2, 2]

    class TransformerTabular(nn.Module):
        def __init__(self, categories, num_cont=5,
                     dim=32, depth=4, heads=8, num_classes=3):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(c, dim) for c in categories])
            enc = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads,
                dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc, depth)
            self.cont_norm   = nn.LayerNorm(num_cont)
            in_dim = dim * len(categories) + num_cont
            self.head = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, num_classes))

        def forward(self, x_cat, x_cont):
            emb = torch.stack([
                self.embeddings[i](x_cat[:, i])
                for i in range(x_cat.shape[1])], dim=1)
            out = self.transformer(emb)
            flat = out.reshape(out.size(0), -1)
            return self.head(torch.cat(
                [flat, self.cont_norm(x_cont)], dim=-1))

    tt = TransformerTabular(cats)
    tt.load_state_dict(torch.load(
        f'{MODELS_DIR}/tabtransformer_best.pt',
        weights_only=True))
    tt.eval()

    return cat_model, xgb_model, tt, meta_learner, scaler

cat_model, xgb_model, tt_model, meta_learner, scaler = load_models()

# ── Prediction function ───────────────────────────────────────
def predict_risk(inputs: dict):
    row = pd.DataFrame([inputs])[ALL_COLS]

    # Scale continuous columns
    cont_scaled = scaler.transform(
        row[['v012','v201','v212','m14_1','b11_01']])
    row_scaled = row.copy()
    row_scaled[['v012','v201','v212','m14_1','b11_01']] = \
        cont_scaled

    X = row_scaled.values

    # Base model probabilities
    cat_p  = cat_model.predict_proba(X)
    xgb_p  = xgb_model.predict_proba(X)

    x_cat  = torch.tensor(
        row[CAT_COLS].values.astype(int), dtype=torch.long)
    x_cont = torch.tensor(
        cont_scaled.astype(np.float32), dtype=torch.float)
    with torch.no_grad():
        logits = tt_model(x_cat, x_cont)
        tt_p   = torch.softmax(logits, dim=1).numpy()

    meta   = np.hstack([cat_p, xgb_p, tt_p])
    proba  = meta_learner.predict_proba(meta)[0]
    pred   = int(np.argmax(proba))

    return pred, proba

# ── AI explanation ────────────────────────────────────────────
def get_ai_explanation(risk_level, proba, inputs, language):
    lang = "Swahili" if language == "Swahili" else "English"

    driver_map = {
        'm14_1': f"ANC visits: {inputs['m14_1']}",
        'v190':  f"Wealth quintile: {inputs['v190']}",
        'm15_1': f"Delivery: {'Home' if inputs['m15_1']==21 else 'Facility'}",
        'v012':  f"Age: {inputs['v012']} years",
        'v201':  f"Parity: {inputs['v201']} births",
        'm3a_1': f"Complication: {'Yes' if inputs['m3a_1']==1 else 'No'}",
        'b11_01':f"Birth interval: {inputs['b11_01']} months",
    }

    prompt = f"""You are a maternal health clinical decision 
support assistant helping a Community Health Worker in Kenya.

Patient profile:
- Age: {inputs['v012']} years
- Parity: {inputs['v201']} births
- ANC visits: {inputs['m14_1']}
- Delivery place: {'Home' if inputs['m15_1']==21 else 'Facility'}
- Complication reported: {'Yes' if inputs['m3a_1']==1 else 'No'}
- Birth interval: {inputs['b11_01']} months
- Wealth quintile: {inputs['v190']} of 5
- Education level: {inputs['v106']} (0=none, 3=higher)

ML Model Result:
- Risk Level: {risk_level}
- Probabilities: Low={proba[0]:.1%}, 
  Moderate={proba[1]:.1%}, High={proba[2]:.1%}

Respond in {lang} with EXACTLY this JSON:
{{
  "explanation": "2-3 sentences explaining the risk to a CHW",
  "recommendations": ["action 1", "action 2", "action 3"],
  "referral_letter": "Short formal referral letter 3-4 sentences"
}}
Return only valid JSON, no markdown."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 800,
                "messages": [{"role":"user","content":prompt}]
            },
            timeout=30
        )
        text = resp.json()['content'][0]['text']
        return json.loads(text)
    except Exception as e:
        return {
            "explanation": f"AI guidance unavailable: {e}",
            "recommendations": [
                "Conduct thorough clinical assessment",
                "Follow standard ANC protocol",
                "Consult nearest health facility if concerned"
            ],
            "referral_letter": "Referral letter unavailable — please write manually."
        }

# ══════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════
st.title("🏥 Maternal Health Risk Predictor")
st.markdown(
    "**AI-assisted decision support for Community Health Workers · Kenya**")
st.markdown("---")

col1, col2 = st.columns([1, 1.6])

# ── LEFT: Input Form ──────────────────────────────────────────
with col1:
    st.subheader("📋 Patient Details")

    age       = st.slider("Maternal Age (years)", 15, 49, 25)
    parity    = st.slider("Number of Births (Parity)", 0, 10, 2)
    anc       = st.slider("ANC Visits", 0, 12, 3)
    interval  = st.slider("Birth Interval (months)", 0, 60, 24)
    age_first = st.slider("Age at First Birth", 14, 35, 20)

    education = st.selectbox(
        "Education Level",
        [0,1,2,3],
        format_func=lambda x: {
            0:"None",1:"Primary",2:"Secondary",3:"Higher"}[x])

    wealth = st.selectbox(
        "Wealth Quintile",
        [1,2,3,4,5],
        format_func=lambda x: {
            1:"1 — Poorest",2:"2 — Poor",3:"3 — Middle",
            4:"4 — Wealthier",5:"5 — Wealthiest"}[x])

    residence = st.radio(
        "Residence", [1,2],
        format_func=lambda x:{1:"Urban",2:"Rural"}[x],
        horizontal=True)

    marital = st.selectbox(
        "Marital Status",
        [1,2,3,4,5],
        format_func=lambda x:{
            1:"Married",2:"Living together",3:"Widowed",
            4:"Divorced",5:"Never married"}[x])

    region = st.selectbox("Region", list(range(1,9)),
        format_func=lambda x: f"Region {x}")

    delivery = st.radio(
        "Place of Delivery",
        [11,21],
        format_func=lambda x:{11:"Health Facility",
                               21:"Home"}[x],
        horizontal=True)

    cs_hist = st.radio(
        "Previous Caesarean Section",
        [0,1],
        format_func=lambda x:{0:"No",1:"Yes"}[x],
        horizontal=True)

    complication = st.radio(
        "Pregnancy Complication Reported",
        [0,1],
        format_func=lambda x:{0:"No",1:"Yes"}[x],
        horizontal=True)

    language = st.radio(
        "AI Output Language",
        ["English","Swahili"],
        horizontal=True)

    predict_btn = st.button(
        "🔍 Assess Maternal Risk",
        use_container_width=True,
        type="primary")

# ── RIGHT: Results ────────────────────────────────────────────
with col2:
    if predict_btn:
        inputs = {
            'v012':age, 'v201':parity, 'v106':education,
            'v190':wealth, 'v025':residence, 'v501':marital,
            'v212':age_first, 'v024':region, 'm14_1':anc,
            'm15_1':delivery, 'm17_1':cs_hist,
            'm3a_1':complication, 'b11_01':interval
        }

        with st.spinner("Running risk assessment..."):
            pred, proba = predict_risk(inputs)

        # ── Risk badge ─────────────────────────────────────────
        risk_name  = RISK_NAMES[pred]
        risk_class = RISK_COLORS[pred]
        risk_icon  = RISK_ICONS[pred]

        st.markdown(
            f'<div class="{risk_class}">'
            f'<h2>{risk_icon} {risk_name}</h2>'
            f'<p>Predicted probability: <strong>'
            f'{proba[pred]:.1%}</strong></p>'
            f'</div>', unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Low Risk",    f"{proba[0]:.1%}")
        c2.metric("🟡 Moderate",    f"{proba[1]:.1%}")
        c3.metric("🔴 High Risk",   f"{proba[2]:.1%}")

        # ── AI Guidance ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### ✦ AI-Powered Clinical Guidance")

        with st.spinner("Generating AI guidance..."):
            ai = get_ai_explanation(
                risk_name, proba, inputs, language)

        st.markdown("**Explanation:**")
        st.info(ai['explanation'])

        st.markdown("**Recommended Actions:**")
        for i, rec in enumerate(ai['recommendations'], 1):
            st.markdown(f"{i}. {rec}")

        st.markdown("**Auto-Generated Referral Letter:**")
        st.text_area("", ai['referral_letter'],
                      height=120, label_visibility="collapsed")

        if st.button("📋 Copy Letter"):
            st.code(ai['referral_letter'])

    else:
        st.markdown("""
        ### How to use this tool

        1. **Enter patient details** in the form on the left
        2. **Click Assess** to run the risk prediction
        3. **Review the risk level** and probability scores
        4. **Read the AI guidance** — explanation, recommendations,
           and referral letter in your chosen language

        ---

        **Models used:**
        - CatBoost + XGBoost + TabTransformer (Stacking Ensemble)
        - Weighted meta-learner for High Risk sensitivity
        - AI API for plain-language clinical guidance

        **⚠️ Clinical disclaimer:**
        This tool supports — it does not replace — 
        CHW clinical judgement. Always conduct a 
        full clinical assessment alongside this tool.
        """)