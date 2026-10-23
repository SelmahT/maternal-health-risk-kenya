# 🏥 Maternal Health Risk Prediction in Kenya
### DSA 3900 — Final Year Project | USIU-Africa

**Student:** Selmah Tzindori Mise | ID: 602
**Programme:** B.Sc. Data Science and Analytics
**Institution:** United States International University – Africa, Nairobi, Kenya
**Supervisor:** Dr Gabriel Okello
**Academic Year:** 2026–2027
**GitHub:** [github.com/SelmahT](https://github.com/SelmahT)
**LinkedIn:** [linkedin.com/in/selmah-tzindori](https://linkedin.com/in/selmah-tzindori)

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
| **SO2** | Apply Bayesian inference (PyMC) to generate calibrated probability scores with 95% credible intervals per prediction |
| **SO3** | Deploy an AI API-integrated Streamlit app generating risk explanations, care recommendations, and referral letters for CHWs in English & Swahili |

---

## 🗂️ Repository Structure

```text
DSA3900_MaternalHealthRisk/
│
├── 01_Data/
│   ├── raw/                               ← KDHS 2022 files (not in repo)
│   └── processed/
│       ├── 00_kdhs_clean.csv              ← 10,391 women × 24 cols
│       ├── X_train.csv / X_val.csv / X_test.csv
│       └── y_train.csv / y_val.csv / y_test.csv
│
├── 02_Notebooks/                          ← Implementation notebooks (real KDHS data)
│   ├── EDA/
│   │   └── 00_DataLoading_EDA.ipynb       ✅
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
├── 03_Models_Saved/                       ← Implementation models (populated after training)
│
├── 04_Outputs/
│   ├── figures/
│   ├── results/
│   └── reports/
│
├── 05_App/
│   └── app.py
│
├── 06_Documentation/
│   ├── proposal_documentation/
│   ├── presentation/
│   └── reports/
│
└── 07_Pilot_Test/                         ← ✅ COMPLETE — Pilot on 1,000 simulated records
    ├── pilot_data/
    │   ├── raw/simulated_kdhs_1000.csv
    │   └── processed/
    │       ├── X_train.csv (1,269 × 13)
    │       ├── X_val.csv (150 × 13)
    │       ├── X_test.csv (150 × 13)
    │       ├── y_train / y_val / y_test.csv
    │       └── pilot_full_processed.csv
    ├── notebooks/
    │   ├── 00_Pilot_DataSimulation.ipynb
    │   ├── 01_Pilot_Preprocessing.ipynb
    │   ├── 02_Pilot_BaselineModels.ipynb
    │   ├── 03_Pilot_AdvancedModels.ipynb
    │   ├── 04_Pilot_NovelModels.ipynb
    │   ├── 05_Pilot_StackingEnsemble.ipynb
    │   ├── 06_Pilot_Bayesian.ipynb
    │   └── 07_Pilot_SHAP.ipynb
    ├── Models_Saved/                      ← Pilot model artefacts (local only)
    ├── results/
    └── app_test/
        └── 08_Pilot_App.py
```

---

## 🧪 Pilot Test — Full Results Summary

The pilot test validated the complete pipeline on **1,000 simulated records**
mirroring KDHS 2022 statistical distributions before applying the system to
real KDHS data.

### Simulated Data

| Variable | Code | Distribution | KDHS Target | Simulated Mean |
|---|---|---|---|---|
| Maternal age | v012 | Uniform(15,49) | 15–49 range | 32.4 |
| Parity | v201 | Poisson(λ=3) | TFR ≈ 3.4 | 2.97 |
| ANC visits | m14_1 | Poisson(λ=4) | WHO min = 4 | 3.94 |
| Birth interval | b11_01 | Normal(μ=36,σ=18) | 35.7 months | 35.69 |
| CS rate | m17_1 | Binomial(0.15) | 14–16% | 16% |
| Place of delivery | m15_1 | Binomial: 70% facility | 66% facility | 70% |

### Target Variable Engineering

Composite risk score from **8 WHO-based clinical flags**:

| Flag | Condition | Variable |
|---|---|---|
| Low ANC | Fewer than 4 ANC visits | m14_1 < 4 |
| Home delivery | Delivered outside facility | m15_1 == 21 |
| Complication | Reported pregnancy complication | m3a_1 == 1 |
| Young mother | Age under 18 | v012 < 18 |
| Older mother | Age over 35 | v012 > 35 |
| High parity | 5 or more births | v201 >= 5 |
| Short interval | Birth spacing < 24 months | b11_01 < 24 |
| Poor | Wealth quintile 1–2 | v190 <= 2 |

**Final classification thresholds** (calibrated on pilot data):

| Flags | Risk Level | Class | Pilot Distribution |
|---|---|---|---|
| 0–1 | Low Risk | 0 | 24.6% (n=246) |
| 2–3 | Moderate Risk | 1 | 60.3% (n=603) |
| 4+ | High Risk | 2 | 15.1% (n=151) |

### Preprocessing Pipeline

| Step | Method | Detail |
|---|---|---|
| Missing values | Median / zero imputation | v212: median; b11_01: 0 for first births |
| Encoding | Label encoding | Categorical variables retained as integers |
| Scaling | StandardScaler | 5 continuous variables |
| Class imbalance | SMOTE (k=5) | Training set only: 700 → 1,269 records |
| Split | Stratified 70/15/15 | Train=700, Val=150, Test=150 |

### Model Performance — All Tiers

| Model | Tier | Accuracy | AUC-ROC | High Recall | F1 | Split |
|---|---|---|---|---|---|---|
| Logistic Regression | Baseline | 0.660 | 0.860 | 0.783 | 0.651 | Val |
| Decision Tree | Baseline | 0.640 | 0.847 | 0.739 | 0.646 | Val |
| Random Forest | Classical | 0.833 | 0.944 | 0.609 | 0.791 | Val |
| XGBoost | Classical | 0.860 | 0.964 | 0.609 | 0.817 | Val |
| CatBoost | Classical | 0.847 | 0.969 | **0.826** | 0.827 | Val |
| TabNet | Novel | 0.687 | 0.819 | 0.522 | 0.640 | Val |
| TabTransformer | Novel | 0.580 | 0.720 | **0.870** | 0.570 | Val |
| Stacking (Standard) | Final | 0.853 | 0.964 | 0.609 | 0.810 | Val |
| Stacking (Weighted) | Final | 0.853 | 0.964 | 0.652 | 0.820 | Val |
| **Stacking (Weighted)** | **Final** | **0.900** | **0.972** | **0.870** | **0.880** | **Test** |

### 🔑 Critical Clinical Result

> **Zero (0) High Risk women were misclassified as Low Risk on the final test set.**
> 87% of genuinely High Risk women were correctly identified.
> The remaining 13% were classified as Moderate — still receive monitoring.

### Meta-Learner Selection

Three stacking configurations were tested:

| Meta-Learner | Accuracy | AUC-ROC | High Recall |
|---|---|---|---|
| Standard LR | 0.853 | 0.964 | 0.609 |
| Weighted LR (3× High Risk penalty) | 0.853 | 0.964 | 0.652 |
| Weighted + Boosted TabTransformer | 0.853 | 0.964 | 0.652 |

**Selected:** Weighted LR — same performance as boosted version with greater simplicity.
Class weights: `{Low: 1.0, Moderate: 1.0, High: 3.0}`

---

## 📐 Bayesian Inference Results

**Sampler:** PyMC v5.x · NUTS · 2 chains · 1,000 tuning + 1,000 draw steps

| Convergence Diagnostic | Value | Target | Status |
|---|---|---|---|
| Maximum R-hat | 1.0034 | < 1.01 | ✅ Converged |
| Minimum ESS | 3,477 | > 400 | ✅ Excellent |

**Output per prediction:**
- Risk class: Low / Moderate / High
- Posterior mean probability: e.g. 0.847
- 95% Highest Density Interval: e.g. [0.71 – 0.93]
- Confidence label: High / Moderate / Low Confidence

**Three clinical output patterns identified:**

| Pattern | Example CI | CI Width | CHW Action |
|---|---|---|---|
| Definitive High Risk | [0.97 – 1.00] | 0.03 | Immediate referral |
| Borderline High Risk | [0.30 – 0.86] | 0.56 | Additional assessment |
| Confident Low Risk | [0.00 – 0.01] | 0.01 | Routine monitoring |

---

## 🔍 SHAP Explainability Results

**Method:** CatBoost + TreeSHAP · Global and individual-level explanations

| Rank | Variable | Code | Mean \|SHAP\| | Clinical Alignment |
|---|---|---|---|---|
| 1 | Wealth Index | v190 | 1.2048 | ✅ Poverty = access barrier |
| 2 | ANC Visits | m14_1 | 1.0856 | ✅ WHO minimum threshold |
| 3 | Place of Delivery | m15_1 | 0.9053 | ✅ Home delivery = risk |
| 4 | Maternal Age | v012 | 0.7880 | ✅ Age extremes = risk |
| 5 | Birth Interval | b11_01 | 0.7239 | ✅ Short spacing = risk |
| 6 | Parity | v201 | 0.5377 | ✅ Grand multiparity |
| 7 | Complication | m3a_1 | 0.2399 | ✅ Direct clinical flag |
| 8–13 | Remaining | — | < 0.15 | Minor contributors |

> All SHAP explanations are clinically coherent. The model learned genuine
> maternal health risk patterns — not spurious statistical correlations.

---

## 📱 Streamlit Application

**File:** `07_Pilot_Test/app_test/08_Pilot_App.py`

**Run locally:**
```bash
cd 07_Pilot_Test/app_test
streamlit run 08_Pilot_App.py
```

**Features:**
- 13-variable CHW patient input form
- Stacking Ensemble risk prediction (Low / Moderate / High)
- Probability breakdown across all 3 classes
- Bayesian credible interval display
- AI API plain-language explanation of risk drivers
- 3 specific care recommendations
- Auto-generated referral letter
- Language toggle: **English** or **Swahili**

---

## 🧮 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| ML Models | scikit-learn, xgboost, catboost, pytorch-tabnet, torch |
| Novel Architectures | Custom TransformerTabular (pure PyTorch) |
| Bayesian Inference | pymc v5.x, arviz |
| Explainability | shap (TreeSHAP) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Optimisation | optuna |
| Visualisation | matplotlib, seaborn |
| App | streamlit |
| Version Control | git, GitHub |
| Environment | VS Code + virtualenv |

---

## 📦 Setup

```bash
# Clone the repo
git clone https://github.com/SelmahT/maternal-health-risk-kenya.git
cd maternal-health-risk-kenya

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

| Detail | Info |
|---|---|
| Name | Kenya Demographic and Health Survey 2022 |
| Source | DHS Program — [dhsprogram.com](https://dhsprogram.com) |
| File | KEIR8CFL.DTA (Individual Recode) |
| Original size | 32,156 rows × 5,925 columns |
| Working dataset | 10,391 rows × 13 columns (after filtering) |
| Access | Requires DHS Program registration + data use agreement |

> ⚠️ The raw dataset is not included in this repository per the DHS Program
> data use agreement. See `01_Data/raw/DATA_README.md` for access instructions.

---

## 📈 Progress

### ✅ Phase 1 — Proposal (Summer Semester 2026) — COMPLETE

- [x] Research topic defined and research gap confirmed
- [x] 5 research articles summarised (literature review)
- [x] KDHS 2022 dataset accessed and explored
- [x] Methodology chapter with full mathematical equations
- [x] Expected results and contributions drafted
- [x] Project scope and Gantt chart submitted (Week 6)
- [x] Concept note presentation delivered (Week 8)
- [x] Full proposal document compiled
- [x] NACOSTI and USIU-Africa IRB ethical approval submitted
- [x] Full proposal presentation delivered and defended

### ✅ Phase 2 — Pilot Test (Summer Semester 2026) — COMPLETE

- [x] Project folder structure and GitHub repository set up
- [x] Virtual environment configured, `requirements.txt` defined
- [x] **Notebook 00** — Simulated data generated (1,000 records, KDHS-mirroring)
- [x] **Notebook 01** — Preprocessing: target variable engineering, SMOTE, 70/15/15 split
- [x] **Notebook 02** — Baseline models: Logistic Regression (AUC=0.86), Decision Tree (AUC=0.85)
- [x] **Notebook 03** — Advanced models: Random Forest, XGBoost, CatBoost (best: AUC=0.97)
- [x] **Notebook 04** — Novel models: TabNet, TabTransformer (best High Recall=0.87)
- [x] **Notebook 05** — Stacking Ensemble: 90% accuracy, 0.972 AUC-ROC, 0.87 High Recall on test set
- [x] **Notebook 06** — Bayesian Inference: R-hat=1.003, ESS=3,477 — converged ✅
- [x] **Notebook 07** — SHAP: wealth index and ANC visits are top drivers — clinically coherent
- [x] **Notebook 08** — Streamlit app: working prototype with AI API integration

### ⬜ Phase 3 — Implementation (Fall Semester 2026) — PENDING

- [ ] Load real KDHS 2022 dataset (10,391 women)
- [ ] Full preprocessing pipeline on real data
- [ ] Optuna hyperparameter tuning (100 trials per model)
- [ ] TabNet and TabTransformer recalibration on full dataset
- [ ] Final stacking ensemble trained and validated
- [ ] Bayesian inference on real data
- [ ] SHAP analysis on real data
- [ ] Streamlit app deployed on Streamlit Community Cloud
- [ ] Clinical validation with CHW participants
- [ ] Final report written and defended

---

## ⚖️ Ethical Compliance

| Requirement | Status |
|---|---|
| NACOSTI ethical clearance | ✅ Submitted |
| USIU-Africa IRB approval | ✅ Submitted |
| DHS Program data use agreement | ✅ Signed |
| No PII in dataset | ✅ KDHS data fully anonymised |
| No data in repository | ✅ Raw data in .gitignore |
| Clinical disclaimer in app | ✅ Tool supports, does not replace, CHW judgement |

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

## 📄 Documents

All proposal and report documents are in [`06_Documentation/`](06_Documentation/):

| Document | Description | Action |
|---|---|---|
| [05_Full_Research_Proposal_Selmah.pdf](06_Documentation/proposal_documentation/05_Full_Research_Proposal_Selmah.pdf) | Complete 3-chapter research proposal | 📄 View |
| [01_LiteratureReview_Selmah.pdf](06_Documentation/proposal_documentation/01_LiteratureReview_Selmah.pdf) | Chapter 2 — Literature Review | 📄 View |
| [02_Methodology_Selmah.pdf](06_Documentation/proposal_documentation/02_Methodology_Selmah.pdf) | Chapter 3 — Methodology with equations | 📄 View |
| [03_ConceptNote_Selmah.pdf](06_Documentation/proposal_documentation/03_ConceptNote_Selmah.pdf) | 1-page concept note | 📄 View |
| [04_Scope_Gantt_Selmah.pdf](06_Documentation/proposal_documentation/04_Scope_Gantt_Selmah.pdf) | Project scope + Gantt chart | 📄 View |
| [01_ConceptNote_Slides_Selmah.pptx](06_Documentation/presentation/01_ConceptNote_Slides_Selmah.pptx) | 4-slide concept note presentation | ⬇️ Download |
| [01_FullProposal_Slides_Selmah.pptx](06_Documentation/presentation/01_FullProposal_Slides_Selmah.pptx) | 11-slide full proposal presentation | ⬇️ Download |
| [01_pilot_report.pdf](06_Documentation/reports/01_pilot_report.pdf) | Pilot test scientific report (LaTeX-compiled) | 📄 View |
| [02_pilot_poster.pdf](06_Documentation/reports/02_pilot_poster.pdf) | Scientific symposium poster — pilot test results | 📄 View | 

> 🔍 **Can't zoom the pilot poster PDF?** View the interactive version here:
> [claude.ai/artifact/CcqavYGdjv9khad85s37N3](https://claude.ai/artifact/CcqavYGdjv9khad85s37N3)

> 📄 **View** — opens directly in GitHub's PDF viewer in your browser  
> ⬇️ **Download** — GitHub will prompt you to download the file

---

## 📬 Contact

**Selmah Tzindori Mise**
GitHub: [github.com/SelmahT](https://github.com/SelmahT)
LinkedIn: [linkedin.com/in/selmah-tzindori](https://linkedin.com/in/selmah-tzindori)
Institution: USIU-Africa, Nairobi, Kenya    