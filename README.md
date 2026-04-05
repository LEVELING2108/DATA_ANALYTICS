# Data Analytics Assignment — BTech ECE

A comprehensive data analytics project covering exploratory data analysis, statistical modeling, hypothesis testing, and machine learning — implemented across 6 assignments using Python, pandas, matplotlib, seaborn, and scikit-learn.

[![Open Assignment 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1szbUIrya42yP-jTuLPcJLhm_jzIP7Vp4?usp=sharing)

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

### Assignment 2 — Data Objects, Attribute Types & Statistical Descriptions
**Dataset:** HR employee records (14,995 records)
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy (no Tableau/Power BI)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1szbUIrya42yP-jTuLPcJLhm_jzIP7Vp4?usp=sharing)

| Part | Task | Key Deliverables |
|------|------|-----------------|
| **A** | Data Object Classification | Classification of all 10 attributes (quantitative, categorical, binary, dependency analysis) |
| **B** | Central Tendency & Dispersion | Mean, median, mode, range, variance, SD, IQR, CV for 5 numeric columns |
| **C** | Univariate & Multivariate Analysis | Histogram, box plot, Q-Q plot, pairplot, correlation analysis (r = 0.349) |
| **D** | Analytics Dashboard | 4-chart dashboard: attrition by dept, satisfaction by salary, evaluation vs hours, box plots |

**Key Finding:** High-performing but dissatisfied employees who work excessive hours (250+/month) are the most likely to leave — indicating burnout-driven attrition.

**Outputs:**
- **Tables:** `partA_classification.csv`, `partB_statistical_summary.csv`
- **Combined Plots:** `partC1_univariate_satisfaction.png`, `partC2_pairplot.png`, `partC3_correlation_scatter.png`, `partD_dashboard.png`
- **Individual Plots:** `C1a_histogram_satisfaction.png`, `C1b_boxplot_satisfaction.png`, `C1c_qqplot_satisfaction.png`, `D1_barchart_attrition_by_dept.png`, `D2_histogram_satisfaction_by_salary.png`, `D3_scatter_evaluation_vs_hours.png`, `D4_boxplot_satisfaction_by_dept.png`
- **Reports:** `assignment2_report.html`, `assignment2_report.md`

**How to Run:**
```bash
cd ASSIGNMENT_2/notebooks
python assignment2_analysis.py          # Run full analysis
python generate_individual_plots.py     # Generate individual plot images
python generate_report.py               # Generate Markdown + HTML reports
```

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

3. **Open a notebook or run a script:**

   **Option A — Google Colab (recommended, no setup):**
   - Click the **"Open in Colab"** badge above for Assignment 2
   - Or open: [Assignment 2 Notebook](https://colab.research.google.com/drive/1szbUIrya42yP-jTuLPcJLhm_jzIP7Vp4?usp=sharing)
   - Run all cells in order — dataset downloads automatically from Google Drive

   **Option B — Local Jupyter Notebook:**
   ```bash
   # Assignment 1
   jupyter notebook ASSIGNMENT_1/notebooks/insurance_assignment.ipynb

   # Assignment 2
   jupyter notebook ASSIGNMENT_2/notebooks/Assignment2_HR_Analysis.ipynb
   ```

   **Option C — Run Python scripts locally:**
   ```bash
   cd ASSIGNMENT_2/notebooks
   python assignment2_analysis.py          # Run full analysis
   python generate_individual_plots.py     # Generate individual plot images
   python generate_report.py               # Generate Markdown + HTML reports
   ```

4. **Run all cells** to reproduce analysis, visualizations, and outputs.

---

## Dataset Sources

### Assignment 1 — Insurance
- **Download URL:** `https://drive.google.com/uc?export=download&id=1oyN6CXzbJq42dL5Jqkn1cP83Hu93CD6q`
- **Records:** 1,338
- **Attributes:** age, sex, bmi, children, smoker, region, charges

### Assignment 2 — HR Employee Records
- **File:** `ASSIGNMENT_2/data/HR_comma_sep.csv`
- **Records:** 14,995
- **Attributes:** satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, Department, salary

---

## Author

**LEVELING2108**  
BTech — Electronics & Communication Engineering  
IIT Madras (Study Programme)
