# Data Analytics Assignment — BTech ECE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-4_of_6_Complete-brightgreen)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LEVELING2108/DATA_ANALYTICS.git)

A comprehensive data analytics project covering exploratory data analysis, statistical modeling, data pre-processing, hypothesis testing, and machine learning — implemented across 6 assignments using Python, pandas, matplotlib, seaborn, and scikit-learn.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/157O99ZYtR4uFgRoAdOcbkEZgiRpMkBth?usp=sharing)

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

### Assignment 3 — Measuring Data Similarity & Dissimilarity
**Datasets:** Wine (178 records) + Nutrient (27 records)
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy (Google Colab)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing)

| Part | Task | Key Deliverables |
|------|------|-----------------|
| **A** | Data Matrix vs. Dissimilarity Matrix | 10×6 data matrix, Euclidean & Manhattan distance heatmaps, metric comparison |
| **B** | Minkowski Distance Experiment | Pairwise distances for p={1,2,3,10}, line chart, Chebyshev convergence analysis |
| **C** | Nominal & Binary Proximity | Energy/fat group similarity, Jaccard & SMC for binary attributes, discussion |
| **D** | Cloud Deployment | Google Colab setup, reusable distance function, environment verification |

**Key Findings:**
- Manhattan distances are consistently larger than Euclidean (ratio ≈ 1.3-1.5×)
- As p→∞, Minkowski distance converges to Chebyshev distance (max single attribute difference)
- Jaccard similarity is more appropriate than SMC for asymmetric binary attributes

**Outputs:** Euclidean/Manhattan heatmaps, Minkowski line chart, Jaccard/SMC comparison table, Colab screenshot

---

### Assignment 4 — Data Pre-processing Pipeline
**Datasets:** Sonar.csv (208 instances, 60 features) + HR_comma_sep.csv (14,999 records)
**Tools:** Google Colab, sklearn, pandas, matplotlib, seaborn

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i9ZhVnz6NUgthtDbTporzmEYuPfgrzWq?usp=sharing)

| Part | Task | Key Deliverables |
|------|------|-----------------|
| **A** | Data Cleaning | Missing value check, duplicate removal, IQR outlier detection, winsorization (5th-95th %ile), type validation |
| **B** | Data Integration & Transformation | Split/merge simulation, min-max normalization, z-score standardization, equal-width discretization (4 bins) |
| **C** | Dimensionality Reduction (Sonar) | PCA (60→46 components for 90% variance), 2D scatter plot, t-SNE comparison |
| **D** | Power BI Reporting | Export PCA-reduced CSV, top 5 component loadings table, 100% stacked bar chart (class balance) |

**Key Findings:**
- PCA reduces 60 sonar features to 46 components explaining 90.8% variance (23% dimensionality reduction)
- t-SNE provides superior class separation in 2D (KL divergence: 1.39) compared to PCA
- Winsorization at 5th/95th percentile capped extreme values (bounds: 106–300 hrs/month)
- Top PC1 loadings: V31 (0.253), V52 (0.227), V54 (0.227), V48 (0.224), V40 (0.217)
- Class balance (Rock: 53.4%, Mine: 46.6%) maintained after PCA reduction

**Outputs:**
- **Plots (13 individual files):** `partA_winsorization_before.png` (58.1 KB), `partA_winsorization_after.png` (60.7 KB), `partB_boxplot_original.png` (52.6 KB), `partB_boxplot_minmax.png` (64.4 KB), `partB_boxplot_zscore.png` (58.9 KB), `partB_discretization_histogram.png` (37.8 KB), `partB_discretization_bins.png` (53.8 KB), `partC_pca_variance_bar.png` (77.1 KB), `partC_pca_variance_cumulative.png` (64.6 KB), `partC_pca_2d_scatter.png` (116.8 KB), `partC_pca_comparison.png` (99.8 KB), `partC_tsne_scatter.png` (111.9 KB), `partD_class_balance.png` (45.4 KB)
- **Data:** `sonar_pca_reduced.csv` (185.0 KB, 208×48 cols), `sonar_pca_loadings.csv` (55.8 KB, 60×47 cols)
- **Reports:** `assignment4_report.md` (18.2 KB)
- **Scripts:** `scripts/run_analysis.py` — standalone runner (generates all 13 individual plots + 2 CSVs)

**How to Run:**
```bash
cd ASSIGNMENT_4/scripts
python run_analysis.py          # Generates all plots + CSVs locally
```

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
   - Click the **"Open in Colab"** badge above for Assignment 2, 3, or 4
   - Or open: [Assignment 2 Notebook](https://colab.research.google.com/drive/1szbUIrya42yP-jTuLPcJLhm_jzIP7Vp4?usp=sharing) | [Assignment 3 Notebook](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing) | [Assignment 4 Notebook](https://colab.research.google.com/drive/1i9ZhVnz6NUgthtDbTporzmEYuPfgrzWq?usp=sharing)
   - Run all cells in order — dataset downloads automatically from Google Drive

   **Option B — Local Jupyter Notebook:**
   ```bash
   # Assignment 1
   jupyter notebook ASSIGNMENT_1/notebooks/insurance_assignment.ipynb

   # Assignment 2
   jupyter notebook ASSIGNMENT_2/notebooks/Assignment2_HR_Analysis.ipynb

   # Assignment 3
   jupyter notebook ASSIGNMENT_3/notebooks/similarity_dissimilarity_assignment.ipynb

   # Assignment 4
   jupyter notebook ASSIGNMENT_4/notebooks/Assignment4_Data_Preprocessing.ipynb
   ```

   **Option C — Run Python scripts locally:**
   ```bash
   cd ASSIGNMENT_2/notebooks
   python assignment2_analysis.py          # Run full analysis
   python generate_individual_plots.py     # Generate individual plot images
   python generate_report.py               # Generate Markdown + HTML reports

   cd ASSIGNMENT_4/scripts
   python run_analysis.py                  # Generate all plots + CSVs for Assignment 4
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

### Assignment 3 — Wine & Nutrient Datasets
- **Files:** `ASSIGNMENT_3/data/wine.csv`, `ASSIGNMENT_3/data/nutrient.csv`
- **Records:** Wine (178), Nutrient (27)
- **Attributes (Wine):** Alcohol, Malic, Ash, Alcalinity, Magnesium, Phenols, Flavanoids, Nonflavanoid, Proanthocyanins, Intensity, Hue, OD280, Proline
- **Attributes (Nutrient):** Food_Item, energy, protein, fat, calcium, iron

### Assignment 4 — Sonar & HR Datasets
- **Files:** `Sonar.csv` (auto-downloaded from UCI), `HR_comma_sep.csv` (Google Drive)
- **Records:** Sonar (208), HR (14,995)
- **Attributes (Sonar):** V1-V60 (continuous frequency-band features), Class (R=Rock, M=Mine)
- **Attributes (HR):** satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, Department, salary

---

## Author

**LEVELING2108**
BTech — Electronics & Communication Engineering
