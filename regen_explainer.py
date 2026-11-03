"""
Regenerates shap_explainer.pkl using the current environment.
Run from the project root with the venv active:
    python regen_explainer.py
"""
from pathlib import Path
import joblib, shap, numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")

MOUT = Path(__file__).parent / "03_Models_Saved"

FEATURE_CODES   = ["v012","v201","v106","v190","v025","v501","v212","v024",
                   "m14_1","m15_1","m17_1","m3a_1","b11_01"]
NUMERIC_FEATURES = ["v012","v201","m14_1","b11_01","v212"]

print("Loading scaler and model...")
scaler = joblib.load(MOUT / "scaler.pkl")
model  = joblib.load(MOUT / "xgb_model.pkl")
print(f"  scaler feature names: {list(scaler.feature_names_in_)}")
print(f"  model type: {type(model).__name__}")

# Dummy rows that match the 13-feature training schema
dummy = pd.DataFrame([
    [28, 3, 2, 3, 2, 1, 20, 1, 4, 2, 0, 1, 24],
    [22, 1, 1, 1, 2, 1, 18, 3, 2, 1, 0, 0, 14],
    [35, 6, 0, 1, 2, 1, 17, 5, 1, 1, 0, 0, 10],
], columns=FEATURE_CODES, dtype=float)

# Apply scaler only to the 5 numeric features (matches training pipeline)
dummy_out = dummy.copy()
dummy_out[NUMERIC_FEATURES] = scaler.transform(dummy[NUMERIC_FEATURES])
dummy_scaled = dummy_out.values

print("Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# Warm-up to confirm correctness
sv = explainer(dummy_scaled)
print(f"  SHAP values shape: {sv.values.shape}  ✓  (expected (3, 13, 3))")

out = MOUT / "shap_explainer.pkl"
joblib.dump(explainer, out)
print(f"\n✓ Saved: {out}")
print("Done — restart the Streamlit app.")