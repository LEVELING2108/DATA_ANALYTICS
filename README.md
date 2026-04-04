# Data Analytics Assignment — BTech ECE

A comprehensive data analytics project covering exploratory data analysis, statistical modeling, hypothesis testing, and machine learning — implemented across 6 assignments using Python, pandas, matplotlib, seaborn, and scikit-learn.

---

## Project Structure

Each assignment follows a consistent directory layout:

```
ASSIGNMENT_N/
├── data/          # Raw datasets (CSV files)
├── notebooks/     # Jupyter notebooks with analysis & visualizations
├── outputs/       # Generated plots, reports, and result files
└── scripts/       # Standalone Python scripts (plot generators, report builders)
```

---

## Assignments Overview

### Assignment 1 — Understanding Data & Attribute Classification
**Dataset:** Insurance charges (1,338 records)

| Part | Task | Key Deliverables |
|------|------|-----------------|
| **A** | Attribute Type Identification | Classification of age, sex, bmi, children, smoker, region, charges as discrete/continuous/nominal |
| **B** | Data Extraction & Cleaning | Loading from Google Drive, missing value check, duplicate removal, encoding (binary + one-hot) |
| **C** | Data Viewing | Scatter plot (BMI vs charges, color-coded by smoker), data matrix, histogram with KDE |
| **D** | Correlation Analysis | Pearson correlation heatmap, identification of strongest/weakest predictors |

**Key Finding:** Smoker status is the strongest predictor of insurance charges (r = 0.787).

**Outputs:** `correlation_heatmap.png`, `histogram_charges.png`, `scatter_bmi_vs_charges.png`, `assignment_report.html`, `assignment_report.md`

---

### Assignment 2 — Descriptive Statistics & Probability Distributions
*(Notebook and scripts in `ASSIGNMENT_2/`)*

Covers measures of central tendency, dispersion, probability distributions (normal, binomial, Poisson), and confidence intervals.

---

### Assignment 3 — Sampling & Estimation
*(Notebook and scripts in `ASSIGNMENT_3/`)*

Covers sampling techniques (random, stratified, systematic), Central Limit Theorem demonstrations, and point/interval estimation.

---

### Assignment 4 — Hypothesis Testing
*(Notebook and scripts in `ASSIGNMENT_4/`)*

Covers t-tests, z-tests, chi-square tests, ANOVA, and p-value interpretation for real-world datasets.

---

### Assignment 5 — Regression & Model Building
*(Notebook and scripts in `ASSIGNMENT_5/`)*

Covers simple & multiple linear regression, model evaluation (R², RMSE, MAE), residual analysis, and assumptions validation.

---

### Assignment 6 — Machine Learning Foundations
*(Notebook and scripts in `ASSIGNMENT_6/`)*

Covers classification algorithms (Logistic Regression, Decision Trees, KNN), train-test split, confusion matrix, ROC curves, and cross-validation.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Data Manipulation** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Statistical Analysis** | scipy, statsmodels |
| **Machine Learning** | scikit-learn |
| **Environment** | Jupyter Notebook |

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LEVELING2108/DATA_ANALYTICS.git
   cd DATA_ANALYTICS
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
   ```

3. **Open a notebook:**
   ```bash
   jupyter notebook ASSIGNMENT_1/notebooks/insurance_assignment.ipynb
   ```

4. **Run all cells** to reproduce analysis, visualizations, and outputs.

---

## Dataset Source

The primary dataset (insurance.csv) is loaded from Google Drive:
- **Download URL:** `https://drive.google.com/uc?export=download&id=1oyN6CXzbJq42dL5Jqkn1cP83Hu93CD6q`
- **Records:** 1,338
- **Attributes:** age, sex, bmi, children, smoker, region, charges

---

## Author

**LEVELING2108**  
BTech — Electronics & Communication Engineering  
IIT Madras (Study Programme)
