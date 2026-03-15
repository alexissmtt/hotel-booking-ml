# 🏨 Hotel Booking Cancellation — Customer Segmentation

> **Statistical Learning** — Master 2 TIDE, Université Paris 1 Panthéon-Sorbonne
> Academic year 2025–2026 | Professor: Alain Celisse

---

## 📌 Project Overview

Analysis of hotel booking cancellations on the **INN Hotels Group dataset** (36,275 bookings, 19 variables). The project covers the full ML pipeline from EDA to business impact analysis, combining supervised and unsupervised approaches.

| # | Research Question | Method | Author |
|---|---|---|---|
| Q1 | Predicting cancellations — EDA, business impact | Logistic Regression · LASSO/Ridge · Random Forest | M. Hadmen |
| Q2 | Discriminant analysis & boosting | LDA · QDA · KNN · XGBoost | B. Kessi |
| **Q3** | **Unsupervised customer segmentation** | **GMM · PCA · BIC/AIC** | **A. Mattei** |

---

## 🔬 My Contribution — Q3: Customer Segmentation via GMM

### Objective
Identify natural booking profiles in an **unsupervised** setting — without using the cancellation label at any point — to uncover hidden customer segments and their cancellation behaviour.

### Methodology
- **Preprocessing**: one-hot encoding of categorical features, standardisation → 27-dimensional feature space
- **Dimensionality reduction**: PCA (10 components, 58.7% variance explained)
- **Model selection**: BIC/AIC minimisation over K ∈ [2, 8] → **K = 4** (EM convergence in 24 iterations)
- **Assignment**: MAP rule on posterior probabilities

### Results — 4 Customer Profiles

| Cluster | Profile | Size | Cancellation rate |
|---------|---------|------|------------------|
| C0 | Standard | 31.5% | 33% |
| C1 | Upscale | 34.7% | 36% (highest) |
| C2 | Early Planners | 25.7% | 33% |
| C3 | Loyal Guests | 8.2% | 17% (lowest) |

**96.9%** of observations have a cluster membership probability > 0.9, confirming well-separated clusters.

### Visualisations
- BIC / AIC curve for K selection
- 2D PCA projection of GMM clusters vs actual cancellation status
- Cluster profiling: lead time, price, special requests, loyalty rate
- Membership probability distribution (MAP rule)

---

## 🏆 Key Results (Full Project)

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| AUC-ROC | 0.956 |
| AUC-PR | 0.935 |
| Optimal threshold | 0.35 → Recall = 91.8% |
| Estimated financial gain | **€688,578** on test set |

---

## 🗂️ Repository Structure

```
hotel-booking-ml/
│
├── notebook/
│   └── Q3_GMM_clustering.ipynb        # Q3 unsupervised segmentation
│
├── data/
│   └── INNHotelsGroup.csv             # INN Hotels Group dataset (36K bookings)
│
├── report/
│   └── hotel-booking-cancellation-report.pdf  # Full academic report
│
└── README.md
```

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)

**Methods:** GMM · PCA · BIC/AIC · MAP rule · Unsupervised learning

---

## 👤 Author

**Alexis Mattei** — Data Scientist @ Groupe BPCE | MSc Data Science, Paris 1 Panthéon-Sorbonne

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/alexis-mt)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/alexissmtt)
