
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