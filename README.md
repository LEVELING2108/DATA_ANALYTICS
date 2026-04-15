# Data Analytics Assignment — BTech ECE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-6_of_6_Complete-brightgreen)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LEVELING2108/DATA_ANALYTICS)

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

### Assignment 5 — Exploratory Data Analysis & Pattern Mining
**Dataset:** Satellite dataset (6,435 records)
**Tools:** Python, Pandas, K-Means, Decision Tree

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qvlruii_o9K_WHOYnZGZ51hNH0sIdF_8?usp=sharing)

*(Notebook and scripts in `ASSIGNMENT_5/`)*

| Part | Task | Key Deliverables |
|------|------|-----------------|
| **A** | Initial EDA | Data loading, class distribution (pie/bar charts), missing value check |
| **B** | Clustering | K-Means with Elbow method (k=2-10), silhouette score (k=6: 0.352), cross-tabulation |
| **C** | Classification | Decision Tree (Acc: 83.9%), confusion matrix, outlier detection (Isolation Forest) |
| **D** | Visualization | Dashboard insights (spectral separability), average band intensity heatmaps |

**Key Findings:** K-Means clustering identifies 3 major spectral groups, while Decision Tree achieves 83.9% accuracy in classifying land-use types.

**Outputs:** `partA_class_distribution.png`, `partB_elbow_method.png`, `partB_kmeans_scatter.png`, `partC_confusion_matrix.png`, `partC_outliers_scatter.png`, `assignment5_report.md`

---

### Assignment 6 — Integrative Mini-Project: End-to-End Analytics Pipeline
**Datasets:** Insurance & HR Employee Records
**Tools:** Python, Google Colab, Tableau Public, GitHub

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LEVELING2108/DATA_ANALYTICS/blob/main/ASSIGNMENT_6/notebooks/Assignment6_Colab.ipynb)

*(Notebook and scripts in `ASSIGNMENT_6/`)*

| Step | Task | Key Deliverables |
|------|------|-----------------|
| **1** | Problem Formulation | Domain context for Healthcare/HR and analytical questions |
| **2** | Data Pre-processing | Cleaning, encoding, and attribute classification for both datasets |
| **3** | EDA & Visualization | Correlation heatmaps, histograms, box plots, and similarity measures |
| **4** | Model Building | HR attrition classification (Decision Tree, Macro F1: 0.96) |
| **5** | Deployment | GitHub repository, Tableau dashboard, and Cloud deployment |

**Key Findings:** Smoking is the primary driver of insurance costs (r=0.79), and employee satisfaction is the strongest predictor of attrition.

**Outputs:** `cleaned_hr.csv`, `cleaned_insurance.csv`, `plot1_hist_charges.png`, `plot2_box_satisfaction.png`, `plot3_heatmap_insurance.png`, `assignment6_report.md`, `model_narrative.txt`

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
   - Click the **"Open in Colab"** badge above for Assignment 2, 3, 4, or 5
   - Or open: [Assignment 2](https://colab.research.google.com/drive/1szbUIrya42yP-jTuLPcJLhm_jzIP7Vp4?usp=sharing) | [Assignment 3](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing) | [Assignment 4](https://colab.research.google.com/drive/1i9ZhVnz6NUgthtDbTporzmEYuPfgrzWq?usp=sharing) | [Assignment 5](https://colab.research.google.com/drive/1Qvlruii_o9K_WHOYnZGZ51hNH0sIdF_8?usp=sharing)
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

   # Assignment 5
   jupyter notebook ASSIGNMENT_5/notebooks/Assignment5_Colab.ipynb

   # Assignment 6
   jupyter notebook ASSIGNMENT_6/notebooks/Assignment6_Colab.ipynb
   ```

   **Option C — Run Python scripts locally:**
   ```bash
   cd ASSIGNMENT_2/notebooks
   python assignment2_analysis.py          # Run full analysis

   cd ASSIGNMENT_4/scripts
   python run_analysis.py                  # Generate all plots + CSVs for Assignment 4

   cd ASSIGNMENT_5/notebooks
   python assignment5_analysis.py          # Run full analysis for Assignment 5

   cd ASSIGNMENT_6/notebooks
   python assignment6_analysis.py          # Run full analysis for Assignment 6
   ```

4. **Run all cells** to reproduce analysis, visualizations, and outputs.

---

## Dataset Sources

### Assignment 1 — Insurance
- **Download URL:** `https://drive.google.com/file/d/1oyN6CXzbJq42dL5Jqkn1cP83Hu93CD6q/view?usp=sharing`
- **Records:** 1,338
- **Attributes:** age, sex, bmi, children, smoker, region, charges

### Assignment 2 — HR Employee Records
- **File:** `ASSIGNMENT_2/data/HR_comma_sep.csv`
- **Download URL:** `https://drive.google.com/file/d/1bviXba_EF5Sqv_RzUUtquON5KUrdrjjj/view?usp=sharing`
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

### Assignment 5 — Satellite Dataset
- **Download URL:** `https://drive.google.com/uc?export=download&id=1ykwDH-9nGWR-8pIilhiD7PcIhvIDy3OF`
- **Records:** 6,435
- **Attributes:** x.1-x.36 (spectral bands), classes (land-use)

### Assignment 6 — Integrative (Insurance + HR)
- **Files:** Combined analysis of datasets from Assignment 1 and Assignment 2.

---

## Author

**LEVELING2108**
BTech — Electronics & Communication Engineering
