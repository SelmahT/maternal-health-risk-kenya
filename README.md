# 🏥 Maternal Health Risk Prediction in Kenya
### DSA 3900 — Final Year Project | USIU-Africa

**Student:** Selmah Tzindori Mise | ID: 669602  
**Programme:** B.Sc. Data Science and Analytics  
**Institution:** United States International University – Africa, Nairobi, Kenya  
**Supervisor:** ___________________________  
**Academic Year:** 2026–2027  

---

## Project Overview

A machine learning system that predicts individual maternal health risk 
(Low / Moderate / High) for pregnant women in Kenya using the Kenya 
Demographic and Health Survey 2022 (KDHS 2022).

The system combines:
- **Novel tabular deep learning** — TabNet and TabTransformer
- **Classical ensemble models** — Random Forest, XGBoost, CatBoost
- **Bayesian inference** — probabilistic risk scores with credible intervals
- **SHAP explainability** — plain-language risk factor attribution
- **AI API integration** — automated explanations and referral letters
- **Streamlit web app** — deployable CHW-facing decision support tool

---

## Research Objectives

| # | Objective |
|---|---|
| SO1 | Build and evaluate ML models (including TabNet & TabTransformer) for maternal risk classification using KDHS 2022 |
| SO2 | Apply Bayesian inference (PyMC) to generate calibrated probability scores with 95% credible intervals |
| SO3 | Deploy an AI API-integrated Streamlit app generating risk explanations and referral letters for CHWs |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| ML Models | scikit-learn, xgboost, catboost, pytorch-tabnet, tab-transformer-pytorch |
| Bayesian | pymc, arviz |
| Explainability | shap |
| Optimisation | optuna |
| Imbalanced data | imbalanced-learn (SMOTE) |
| Visualisation | matplotlib, seaborn |
| App | streamlit |
| Environment | VS Code + virtualenv |

---

## Dataset

| Detail | Info |
|---|---|
| Name | Kenya Demographic and Health Survey 2022 |
| Source | DHS Program — dhsprogram.com |
| File | KEIR8CFL.DTA (Individual Recode) |
| Size | 32,156 rows × 5,925 columns |
| Working dataset | 10,391 rows × 13 columns |
| Access | Requires DHS Program registration and data use agreement |

> ⚠️ The raw dataset is not included in this repository.  
> See `01_Data/raw/DATA_README.md` for access instructions.

---

## Project Structure
---

## Target Variable

Composite maternal risk score engineered from 8 WHO-based clinical flags:

| Flag | Condition | Variable |
|---|---|---|
| low_anc | Fewer than 4 ANC visits | m14_1 < 4 |
| home_delivery | Delivered outside health facility | m15_1 == 21 |
| complication | Reported pregnancy complication | m3a_1 == 1 |
| young_mother | Age under 18 | v012 < 18 |
| older_mother | Age over 35 | v012 > 35 |
| high_parity | 5 or more births | v201 >= 5 |
| short_interval | Birth interval under 24 months | b11_01 < 24 |
| poor | Wealth quintile 1 or 2 | v190 <= 2 |

**Classification:**
- 0–1 flags → Low Risk (class 0)
- 2–3 flags → Moderate Risk (class 1)
- 4+ flags → High Risk (class 2)

---

## Progress

### ✅ Proposal Phase — Complete
- [x] Research topic defined and defended
- [x] Literature review (Chapter 2) — 26 citations, 2023–2026
- [x] Methodology chapter (Chapter 3) — with full mathematical equations
- [x] Full proposal document compiled
- [x] NACOSTI and USIU-Africa IRB ethical approval submitted
- [x] Concept note and full proposal presentations prepared

### 🟡 Pilot Test Phase — In Progress
- [x] Project folder structure created
- [x] GitHub repository set up
- [x] Virtual environment configured
- [x] `requirements.txt` defined
- [x] Notebook 00 — Simulated data generated (1,000 records)
- [x] Notebook 01 — Preprocessing and target variable engineering
- [ ] Notebook 02 — Baseline models (Logistic Regression, Decision Tree)
- [ ] Notebook 03 — Advanced models (RF, XGBoost, CatBoost)
- [ ] Notebook 04 — Novel models (TabNet, TabTransformer)
- [ ] Notebook 05 — Stacking ensemble
- [ ] Notebook 06 — Bayesian inference (PyMC)
- [ ] Notebook 07 — SHAP explainability
- [ ] Notebook 08 — Streamlit app pilot

### ⬜ Implementation Phase — Pending 
- [ ] Load real KDHS 2022 dataset
- [ ] Full pipeline on real data
- [ ] Final model selection and validation
- [ ] App deployment on Streamlit Community Cloud

---

## Ethical Compliance

- NACOSTI ethical clearance — submitted
- USIU-Africa IRB approval — submitted
- DHS Program data use agreement — signed
- No personally identifiable information used
- All KDHS records are fully anonymised

---

## References

Key methodological references:

- Arik & Pfister (2021) — TabNet: AAAI 2021
- Huang et al. (2020) — TabTransformer: arXiv:2012.06678
- Sheridan et al. (2025) — Bayesian Network pregnancy outcomes
- Mamun et al. (2025) — Explainable ML for maternal health
- Ombech et al. (2025) — Digital health for maternal care in Kenya
- KNBS & ICF (2023) — Kenya Demographic and Health Survey 2022

---

## Contact

**Selmah Tzindori Mise**  
GitHub: [github.com/SelmahT](https://github.com/SelmahT)  
LinkedIn: [linkedin.com/in/selmah-tzindori](https://linkedin.com/in/selmah-tzindori)  
Institution: USIU-Africa, Nairobi, Kenya