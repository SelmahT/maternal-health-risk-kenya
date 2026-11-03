# 🏥 Maternal Health Risk Prediction in Kenya
### DSA 4900 — Final Year Project | USIU-Africa

**Student:** Selmah Tzindori Mise | ID: 602
**Programme:** B.Sc. Data Science and Analytics
**Institution:** United States International University – Africa, Nairobi, Kenya
**Supervisor:** Dr Gabriel Okello
**Academic Year:** 2026–2027
**GitHub:** [github.com/SelmahT](https://github.com/SelmahT)
**LinkedIn:** [linkedin.com/in/selmah-tzindori](https://linkedin.com/in/selmah-tzindori)

> 🚀 **Live App:** [mamacare-kenya.streamlit.app](https://mamacare-kenya.streamlit.app)

---

## 📌 Project Overview

A machine learning system that predicts individual maternal health risk
(**Low / Moderate / High**) for pregnant women in Kenya, using the Kenya
Demographic and Health Survey 2022 (KDHS 2022).

Kenya's maternal mortality ratio stands at **355 deaths per 100,000 live births** —
nearly 5× the SDG 3.1 target of 70. Over 100,000 Community Health Workers visit
pregnant women daily with no data-driven risk tool. This project builds one.

---

## 🎯 Research Objectives

| # | Objective |
|---|---|
| **SO1** | Build and evaluate ML models (TabNet, TabTransformer, Stacking Ensemble) to classify individual maternal risk as Low / Moderate / High using KDHS 2022 |
| **SO2** | Apply Bayesian inference (PyMC) to generate calibrated probability scores with 94% credible intervals per prediction |
| **SO3** | Deploy an AI API-integrated Streamlit app generating risk explanations, care recommendations, and referral letters for CHWs in English & Kiswahili |

---

## 🗂️ Repository Structure

```text
maternal-health-risk-kenya/
│
├── 01_Data/
│   ├── raw/                               ← KDHS 2022 files (not in repo — DHS licence)
│   └── processed/
│       ├── 00_kdhs_clean.csv              ← 10,391 women × 24 cols
│       ├── X_train.csv / X_val.csv / X_test.csv
│       └── y_train.csv / y_val.csv / y_test.csv
│
├── 02_Notebooks/                          ← Implementation notebooks (real KDHS data)
│   ├── EDA/
│   │   └── 00_DataLoading_EDA.ipynb
│   ├── Preprocessing/
│   │   └── 01_Preprocessing.ipynb
│   ├── Models/
│   │   ├── 02_Baseline_Models.ipynb
│   │   ├── 03_Classical_Models_Optuna.ipynb
│   │   ├── 04_Novel_Models_Optuna.ipynb
│   │   └── 05_Stacking_Ensemble.ipynb
│   ├── Bayesian/
│   │   └── 06_Bayesian_Inference.ipynb
│   └── SHAP/
│       └── 07_SHAP_Explainability.ipynb
│
├── 03_Models_Saved/                       ← Trained model artefacts
│   ├── scaler.pkl                         ← StandardScaler (5 numeric features)
│   ├── xgb_model.pkl                      ← Final XGBoost classifier
│   └── shap_explainer.pkl                 ← SHAP TreeExplainer (regenerated at runtime)
│
├── 04_Outputs/
│   ├── figures/                           ← SHAP plots, Bayesian posterior images
│   ├── results/
│   │   └── all_model_results.csv          ← Model leaderboard data
│   └── reports/
│
├── 05_App/
│   ├── app.py                             ← MamaCare Kenya Streamlit app
│   └── requirements.txt                   ← Python dependencies
│
├── 06_Documentation/
│   ├── proposal_documentation/
│   ├── presentation/
│   └── reports/
│
├── 07_Pilot_Test/                         ← ✅ COMPLETE — Pilot on 1,000 simulated records
│
├── regen_explainer.py                     ← Regenerates shap_explainer.pkl locally
└── .python-version                        ← Pins Python 3.10 for Streamlit Cloud
```

---

## 📱 Live Application

**URL:** [mamacare-kenya.streamlit.app](https://mamacare-kenya.streamlit.app)

**Run locally:**
```bash
git clone https://github.com/SelmahT/maternal-health-risk-kenya.git
cd maternal-health-risk-kenya
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r 05_App/requirements.txt
streamlit run 05_App/app.py
```

**Features:**
- Role-based access: **Clinician** (risk assessment + referral) vs **Researcher** (full model insights)
- 13-variable CHW patient input form
- XGBoost risk prediction (Low / Moderate / High) with class probabilities
- SHAP waterfall explanation per prediction
- Google Gemini 3.8 Flash AI clinical narrative (English & Kiswahili)
- Rule-based fallback summary (works offline / without API key)
- Auto-generated PDF referral letter
- Language toggle: **English** or **Kiswahili**

---

## 🏆 Final Model Results (KDHS 2022 — Real Data)

| Model | Accuracy | Macro F1 | High-Risk F1 | AUC |
|---|---|---|---|---|
| **XGBoost** ✅ | **99.94%** | **0.9992** | **0.9985** | **1.000** |
| CatBoost | 99.87% | 0.9985 | 0.9970 | 1.000 |
| Random Forest | 99.68% | 0.9962 | 0.9940 | 1.000 |
| TabTransformer | 99.62% | 0.9958 | 0.9940 | 0.9997 |
| TabNet | 98.97% | 0.9884 | 0.9821 | 0.9997 |
| Decision Tree | 94.23% | 0.9388 | 0.9292 | — |
| Logistic Regression | 81.14% | 0.8109 | 0.8314 | — |

**Bayesian HDI (94%):** [0.9909, 0.9995] · R̂ = 1.00 · ESS > 1,800

---

## 🧪 Pilot Test Summary

The pilot test validated the complete pipeline on **1,000 simulated records**
mirroring KDHS 2022 statistical distributions before applying the system to
real KDHS data.

**Best pilot result:** Stacking Ensemble — Accuracy 90%, AUC 0.972, High Recall 0.87 on test set.

> **Critical clinical result:** Zero High Risk women were misclassified as Low Risk on the final test set.

---

## 🔍 SHAP Explainability (Implementation — Real KDHS Data)

| Rank | Variable | Code | Mean \|SHAP\| |
|---|---|---|---|
| 1 | Total Children Born | v201 | ~3.98 |
| 2 | ANC Visits | m14_1 | ~3.97 |
| 3 | Wealth Index | v190 | ~3.97 |
| 4 | Maternal Age | v012 | major |
| 5 | Birth Interval | b11_01 | major |

All SHAP explanations are clinically coherent — the model learned genuine maternal health risk patterns.

---

## 🧮 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data | pandas, numpy |
| ML Models | scikit-learn, xgboost, catboost, pytorch-tabnet, torch |
| Novel Architectures | Custom TabTransformer (pure PyTorch) |
| Bayesian Inference | pymc v5.x, arviz |
| Explainability | shap (TreeSHAP) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Optimisation | optuna |
| Visualisation | matplotlib, seaborn |
| App | streamlit 1.65.0 |
| AI Narrative | Google Gemini 3.8 Flash |
| Deployment | Streamlit Community Cloud |
| Version Control | git, GitHub |

---

## 📊 Dataset

| Detail | Info |
|---|---|
| Name | Kenya Demographic and Health Survey 2022 |
| Source | DHS Program — [dhsprogram.com](https://dhsprogram.com) |
| File | KEIR8CFL.DTA (Individual Recode) |
| Original size | 32,156 rows × 5,925 columns |
| Working dataset | 10,391 rows × 13 columns (after filtering) |
| Training set (post-SMOTE) | 9,054 rows × 13 columns (3,018 per class) |
| Access | Requires DHS Program registration + data use agreement |

> ⚠️ The raw dataset is not included in this repository per the DHS Program
> data use agreement. See `01_Data/raw/DATA_README.md` for access instructions.

**13 Features used:**

| Code | Description | Type |
|---|---|---|
| v012 | Age of woman | Numeric |
| v201 | Total children born | Numeric |
| v106 | Education level | Categorical |
| v190 | Wealth index | Categorical |
| v025 | Residence (urban/rural) | Categorical |
| v501 | Marital status | Categorical |
| v212 | Age at first birth | Numeric |
| v024 | Region | Categorical |
| m14_1 | ANC visits | Numeric |
| m15_1 | Place of delivery | Categorical |
| m17_1 | Caesarean section | Categorical |
| m3a_1 | Doctor at delivery | Categorical |
| b11_01 | Birth interval (months) | Numeric |

---

## 📈 Progress

### ✅ Phase 1 — Proposal (Summer Semester 2026) — COMPLETE

- [x] Research topic defined and research gap confirmed
- [x] Literature review (5 key articles)
- [x] KDHS 2022 dataset accessed and explored
- [x] Methodology chapter with full mathematical equations
- [x] Project scope and Gantt chart submitted
- [x] Concept note presentation delivered
- [x] Full proposal document compiled and defended
- [x] NACOSTI and USIU-Africa IRB ethical approval submitted

### ✅ Phase 2 — Pilot Test (Summer Semester 2026) — COMPLETE

- [x] Notebooks 00–08: full pipeline on 1,000 simulated records
- [x] Stacking Ensemble: 90% accuracy, 0.972 AUC, 0.87 High Recall
- [x] Bayesian inference: R̂=1.003, ESS=3,477
- [x] SHAP: wealth index and ANC visits top drivers
- [x] Streamlit prototype with AI API integration

### ✅ Phase 3 — Implementation (Fall Semester 2026) — COMPLETE

- [x] Full preprocessing on real KDHS 2022 data (10,391 women)
- [x] Optuna hyperparameter tuning (all models)
- [x] XGBoost final model: Accuracy=99.94%, Macro F1=0.9992, AUC=1.000
- [x] Bayesian inference: HDI [0.9909, 0.9995], R̂=1.00, ESS>1,800
- [x] SHAP analysis on real data
- [x] MamaCare Kenya Streamlit app deployed on Streamlit Community Cloud
- [ ] Final report written and defended

---

## ⚖️ Ethical Compliance

| Requirement | Status |
|---|---|
| NACOSTI ethical clearance | ✅ Submitted |
| USIU-Africa IRB approval | ✅ Submitted |
| DHS Program data use agreement | ✅ Signed |
| No PII in dataset | ✅ KDHS data fully anonymised |
| No raw data in repository | ✅ Raw data in .gitignore |
| Clinical disclaimer in app | ✅ Tool supports, does not replace, CHW judgement |
| Role-based access control | ✅ Clinician / Researcher separation |

---

## 📄 Documents

All proposal and report documents are in [`06_Documentation/`](06_Documentation/):

| Document | Description |
|---|---|
| [05_Full_Research_Proposal_Selmah.pdf](06_Documentation/proposal_documentation/05_Full_Research_Proposal_Selmah.pdf) | Complete 3-chapter research proposal |
| [01_LiteratureReview_Selmah.pdf](06_Documentation/proposal_documentation/01_LiteratureReview_Selmah.pdf) | Chapter 2 — Literature Review |
| [02_Methodology_Selmah.pdf](06_Documentation/proposal_documentation/02_Methodology_Selmah.pdf) | Chapter 3 — Methodology |
| [03_ConceptNote_Selmah.pdf](06_Documentation/proposal_documentation/03_ConceptNote_Selmah.pdf) | 1-page concept note |
| [04_Scope_Gantt_Selmah.pdf](06_Documentation/proposal_documentation/04_Scope_Gantt_Selmah.pdf) | Project scope + Gantt chart |
| [01_pilot_report.pdf](06_Documentation/reports/01_pilot_report.pdf) | Pilot test scientific report |
| [02_pilot_poster.pdf](06_Documentation/reports/02_pilot_poster.pdf) | Scientific symposium poster |

> 🔍 **Interactive pilot poster:** [claude.ai/artifact/CcqavYGdjv9khad85s37N3](https://claude.ai/artifact/CcqavYGdjv9khad85s37N3)

---

## 📚 Key References

- Arik & Pfister (2021) — TabNet: AAAI 2021
- Huang et al. (2020) — TabTransformer: arXiv:2012.06678
- Sheridan et al. (2025) — Bayesian Network pregnancy outcomes
- Mamun et al. (2025) — Explainable ML for maternal health (99.51% accuracy)
- Tariq et al. (2026) — TabNet for obstetric prediction (BMC Pregnancy)
- Ombech et al. (2025) — Digital health for maternal care in Kenya
- KNBS & ICF (2023) — Kenya Demographic and Health Survey 2022
- WHO (2023) — Trends in maternal mortality estimates

---

## 📬 Contact

**Selmah Tzindori Mise**
GitHub: [github.com/SelmahT](https://github.com/SelmahT)
LinkedIn: [linkedin.com/in/selmah-tzindori](https://linkedin.com/in/selmah-tzindori)
Institution: USIU-Africa, Nairobi, Kenya