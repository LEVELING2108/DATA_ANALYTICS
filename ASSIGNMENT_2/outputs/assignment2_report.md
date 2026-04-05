# Assignment 2 — Data Objects, Attribute Types & Statistical Descriptions

**Course:** Data Analytics | **Program:** B.Tech ECE — IIT Madras  
**Dataset:** HR_comma_sep.csv (14,995 employee records)  
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy  
**Date:** 05 April 2026  
**Marks:** [20 Marks]

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1szbUIrya42yP-jTuLPcJLhm_jzIP7Vp4?usp=sharing)

---

## Executive Summary

This report presents a comprehensive statistical analysis of the HR employee dataset containing 14,995 records. The analysis covers data object classification, measures of central tendency and dispersion, univariate and multivariate analysis, and a full analytics dashboard — all implemented in Python without reliance on Tableau or Power BI.

**Key Finding:** Employees who leave the company exhibit a distinct pattern — **low satisfaction (mean ≈ 0.44)** combined with **high evaluation scores (mean ≈ 0.78)** and **excessive monthly hours (mean ≈ 257)**, suggesting burnout among high performers is the primary attrition driver.

---

## Table of Contents

1. [Part A — Data Object Classification](#part-a--data-object-classification)
2. [Part B — Central Tendency & Dispersion](#part-b--central-tendency--dispersion)
3. [Part C — Univariate & Multivariate Analysis](#part-c--univariate--multivariate-analysis)
4. [Part D — Analytics Dashboard](#part-d--analytics-dashboard)
5. [Key HR Insights & Recommendations](#key-hr-insights--recommendations)

---

## Part A — Data Object Classification [4 Marks]

### A1. Classification Table

The HR dataset contains 14,995 independent employee records with 10 attributes. Each attribute is classified below according to the data type taxonomy from Unit II.

| Data Type Category | Columns | Justification |
|---|---|---|
| **Quantitative — Continuous** | `satisfaction_level`, `last_evaluation`, `average_montly_hours` | Continuous numeric measurements on a scale. `satisfaction_level` (0–1), `last_evaluation` (0–1), and `average_montly_hours` (96–310) represent measurable quantities with meaningful arithmetic operations. |
| **Quantitative — Discrete** | `number_project`, `time_spend_company` | Discrete numeric values (counts/years). `number_project` is a count of projects (2–7), `time_spend_company` is years of tenure (2–10). They support arithmetic but take only integer values. |
| **Categorical — Nominal** | `Department` | Named categories with no inherent ordering: sales, accounting, hr, technical, support, management, IT, product_mng, marketing, RandD. |
| **Categorical — Ordinal** | `salary` | Categories (low, medium, high) with a natural ordering from low → high, making it an ordinal categorical attribute. |
| **Binary (Dichotomous)** | `left`, `Work_accident`, `promotion_last_5years` | Binary attributes encoded as 0/1. `left` = attrition (1=left, 0=stayed), `Work_accident` = work accident occurrence, `promotion_last_5years` = promotion in last 5 years. |

### A2. Dependency Analysis

**Classification:** The dataset is primarily **non-dependency-oriented**.

**Explanation:**
- Each row represents an **independent cross-sectional snapshot** of an employee at a point in time.
- There is **no explicit temporal sequence** (no timestamps, no panel data structure).
- There are **no network relationships** encoded between employees.

**However, potential implicit dependencies exist:**

| Dependency Type | Evidence | Impact |
|---|---|---|
| **Temporal** | `time_spend_company` and `promotion_last_5years` imply historical accumulation | Employees with longer tenure may have different attrition patterns |
| **Causal** | `Work_accident` → `satisfaction_level` → `left` | Accidents may reduce satisfaction, increasing attrition risk |
| **Network/Group** | Employees within the same `Department` share management, culture, workload | Department-level effects may create correlated responses |
| **Selection Bias** | The dataset only includes current + recently departed employees | Surviving employees may not represent the full historical population |

**Conclusion:** While the dataset is treated as independent observations, real-world HR data likely contains unobserved dependencies that should be considered in advanced modeling.

---

## Part B — Central Tendency & Dispersion [6 Marks]

### B1. Statistical Summary Table

| Statistic | satisfaction_level | last_evaluation | number_project | average_montly_hours | time_spend_company |
|---|---|---|---|---|---|
| **Mean** | 0.6129 | 0.7161 | 3.803 | 201.0502 | 3.4982 |
| **Median** | 0.64 | 0.72 | 4.0 | 200.0 | 3.0 |
| **Mode** | 0.1 | 0.55 | 4 | 135 | 3 |
| **Range** | 0.91 | 0.64 | 5 | 214 | 8 |
| **Variance** | 0.0618 | 0.0293 | 1.5185 | 2494.3388 | 2.1324 |
| **Std Deviation** | 0.2486 | 0.1712 | 1.2323 | 49.9434 | 1.4603 |
| **IQR** | 0.38 | 0.31 | 2.0 | 89.0 | 1.0 |
| **Min** | 0.09 | 0.36 | 2 | 96 | 2 |
| **Max** | 1.0 | 1.0 | 7 | 310 | 10 |
| **CV (%)** | 40.56% | 23.9% | 32.4% | 24.84% | 41.74% |

### B2. Dispersion Analysis

The **Coefficient of Variation (CV)** measures relative dispersion as a percentage of the mean:

| Column | CV | Dispersion Level | Interpretation |
|---|---|---|---|
| `time_spend_company` | 41.74% | **Moderately Dispersed** | Employee tenure varies significantly — mix of new joiners and long-tenured staff |
| `satisfaction_level` | 40.56% | **Moderately Dispersed** | Wide spread in satisfaction — bimodal distribution with clusters at very low (~0.1) and high (~0.8) |
| `number_project` | 32.4% | **Moderately Dispersed** | Employees handle 2–7 projects; moderate workload variation |
| `average_montly_hours` | 24.84% | **Tightly Clustered** | Most employees work 150–250 hours/month; relatively consistent workload |
| `last_evaluation` | 23.9% | **Tightly Clustered** | Evaluation scores concentrate around 0.7–0.9; most employees rated similarly |

**Key Observations:**
- **Most dispersed:** `time_spend_company` (CV = 41.74%) — indicates diverse tenure distribution
- **Least dispersed:** `last_evaluation` (CV = 23.9%) — evaluation scores are tightly grouped, suggesting limited differentiation in performance ratings
- **Bimodal satisfaction:** The mean (0.6129) is lower than the median (0.64), indicating left-skew — a significant portion of employees are highly dissatisfied

---

## Part C — Univariate & Multivariate Analysis [5 Marks]

### C1. Univariate Analysis — `satisfaction_level`

#### Distribution Characteristics

| Metric | Value |
|---|---|
| Mean | 0.6129 |
| Median | 0.64 |
| Std Deviation | 0.2486 |
| Skewness | -0.4766 |
| Kurtosis | -0.6704 |

**Skewness Interpretation:** The negative skew (-0.4766) indicates a **left-skewed distribution** — the tail extends toward lower satisfaction values, meaning more employees cluster at higher satisfaction levels, but a significant minority report very low satisfaction.

#### Outlier Analysis (IQR Method)

| Metric | Value |
|---|---|
| Q1 (25th percentile) | 0.44 |
| Q3 (75th percentile) | 0.82 |
| IQR | 0.38 |
| Lower Bound (Q1 − 1.5×IQR) | -0.13 |
| Upper Bound (Q3 + 1.5×IQR) | 1.39 |
| Outliers Detected | 0 |

**Finding:** No statistical outliers detected by the IQR method because the bounds (-0.13 to 1.39) encompass the full data range (0.09 to 1.0). However, the **bimodal shape** of the histogram reveals two distinct employee groups:
1. **Highly dissatisfied** (satisfaction ≈ 0.09–0.20) — likely candidates for attrition
2. **Satisfied** (satisfaction ≈ 0.70–0.95) — likely retained employees

#### Visualizations

- **Histogram:** Shows bimodal distribution with peaks near 0.1 and 0.8
- **Box Plot:** Symmetric box with median at 0.64, no extreme outliers
- **Q-Q Plot:** Deviation from the diagonal line confirms **non-normal distribution** — the data is bimodal, not Gaussian

> 📊 **See:** `outputs/partC1_univariate_satisfaction.png` (combined) or individual plots:
> - `outputs/C1a_histogram_satisfaction.png`
> - `outputs/C1b_boxplot_satisfaction.png`
> - `outputs/C1c_qqplot_satisfaction.png`

---

### C2. Multivariate Analysis — Pair Plot

A Seaborn pairplot was generated for all numeric columns, color-coded by the `left` (attrition) variable.

#### Key Separations Identified

**Pair 1: `satisfaction_level` vs `last_evaluation`**
- **Departed employees** (left=1) cluster in the **bottom-right quadrant**: LOW satisfaction (< 0.4) + HIGH evaluation (> 0.7)
- **Retained employees** (left=0) spread across the full range but concentrate in the **top-left quadrant**: HIGH satisfaction + MODERATE evaluation
- **Interpretation:** High performers who are dissatisfied are the most likely to leave — suggesting unrecognized or unrewarded talent

**Pair 2: `average_montly_hours` vs `satisfaction_level`**
- **Departed employees** cluster at **HIGH hours** (250–300+) + **LOW satisfaction** (< 0.4)
- **Retained employees** concentrate at **MODERATE hours** (150–220) + **MODERATE-HIGH satisfaction** (> 0.5)
- **Interpretation:** Overwork is strongly associated with dissatisfaction and attrition — a classic burnout pattern

**Pair 3: `number_project` vs `average_montly_hours`**
- Clear positive trend: more projects → more hours worked
- Departed employees concentrate in the **4–7 project** range with **250+ hours/month**
- **Interpretation:** Heavy workload (many projects + long hours) without corresponding satisfaction drives attrition

> 📊 **See:** `outputs/partC2_pairplot.png` (combined matrix)

---

### C3. Correlation Analysis

**Pearson Correlation: `last_evaluation` vs `number_project`**

| Metric | Value |
|---|---|
| Correlation Coefficient (r) | 0.3491 |
| p-value | 0.00e+00 |
| Strength | Weak |
| Direction | Positive |
| R² (Variance Explained) | 12.18% |

**Interpretation:**

The correlation coefficient of **r = 0.3491** indicates a **weak positive relationship** between the number of projects an employee handles and their evaluation score.

- **R² = 12.18%** of the variance in evaluation scores is explained by the number of projects
- **p-value < 0.001** confirms this relationship is **statistically significant** (not due to chance)
- **Practical meaning:** Employees assigned more projects receive higher performance ratings, suggesting that **visibility and workload drive recognition**
- **Caveat:** This is a **weak-to-moderate** correlation (not strong), meaning many other factors (quality of work, manager bias, team dynamics) also influence evaluations

> 📊 **See:** `outputs/partC3_correlation_scatter.png`

---

## Part D — Analytics Dashboard [5 Marks]

### D1. Dashboard Overview

A comprehensive 4-panel dashboard was built using Matplotlib and Seaborn as an alternative to Tableau/Power BI.

#### Chart 1: Employee Attrition by Department (Grouped Bar Chart)

**What it shows:** Count of employees who left vs. stayed, broken down by department.

**Key observations:**
- **Sales, Technical, and Support** have the highest absolute attrition counts (due to larger department sizes)
- **Management and RandD** show relatively lower attrition
- Attrition rates across departments range from ~14.4% to ~29.1%

#### Chart 2: Satisfaction Level Distribution by Salary Band (Overlaid Histogram)

**What it shows:** Distribution of `satisfaction_level` separated by salary band (low / medium / high).

**Key observations:**

| Salary Band | Mean Satisfaction | Std Dev | Shape |
|---|---|---|---|
| Low | 0.601 | 0.255 | Bimodal — peak at ~0.1 and ~0.8 |
| Medium | 0.622 | 0.245 | Near-normal — centered at ~0.6 |
| High | 0.637 | 0.227 | Right-skewed — concentrated at ~0.7–0.9 |

- **Low-salary employees** show a bimodal split: some are very satisfied, many are very dissatisfied
- **High-salary employees** are consistently more satisfied with less variance

#### Chart 3: Last Evaluation vs Monthly Hours (Scatter Plot, Color-coded by Attrition)

**What it shows:** Relationship between evaluation score and monthly hours, with red dots for departed employees.

**Key observations:**
- A distinct **red cluster** appears at **high hours (250–310)** + **high evaluation (0.7–1.0)** — these are departed high-performers
- Retained employees (blue) are spread across moderate hours (150–250) with varied evaluations
- **Insight:** The company is losing its hardest-working, highest-evaluated employees — a critical retention risk

#### Chart 4: Satisfaction by Department & Attrition Status (Box Plot)

**What it shows:** Box plots of `satisfaction_level` for each department, split by stayed vs. left.

**Key observations:**
- Across **all departments**, departed employees have **lower median satisfaction** than retained employees
- The satisfaction gap is consistent (~0.2–0.3 points) regardless of department
- **Insight:** Attrition drivers are **organization-wide**, not department-specific

> 📊 **Full Dashboard:** `outputs/partD_dashboard.png` (combined) or individual charts:
> - `outputs/D1_barchart_attrition_by_dept.png`
> - `outputs/D2_histogram_satisfaction_by_salary.png`
> - `outputs/D3_scatter_evaluation_vs_hours.png`
> - `outputs/D4_boxplot_satisfaction_by_dept.png`

---

### D2. Key HR Insights Summary

1. **Burnout of High Performers:** The most alarming pattern is employees who left had **low satisfaction** (mean ≈ 0.44) but **high evaluation scores** (mean ≈ 0.78) and **worked excessive hours** (mean ≈ 257 hrs/month). These are top performers who are burning out and leaving.

2. **Salary is a Key Satisfaction Driver:** Low-salary employees show a bimodal satisfaction distribution — either very unhappy (~0.1) or surprisingly happy (~0.8). Medium and high-salary employees show more consistent, moderate-to-high satisfaction.

3. **Workload–Attrition Link:** Employees handling 5+ projects and working 250+ hours/month are significantly more likely to leave, regardless of their evaluation scores.

4. **Universal Attrition Pattern:** The satisfaction gap between stayed and left employees is consistent across all 10 departments, suggesting organization-wide issues (compensation, recognition, work-life balance) rather than department-specific problems.

5. **Critical Tenure Window:** Most attrition occurs at 3–5 years of tenure, suggesting employees reassess their career growth after the initial adjustment period. Retention interventions should target this window.

6. **Evaluation System Bias:** The weak correlation (r = 0.3491) between projects and evaluations suggests that employees who take on more work get rated higher, but this may not reflect actual quality — potentially demotivating employees with fewer project opportunities.

---

## Recommendations for HR Management

| Priority | Action | Expected Impact |
|---|---|---|
| **1** | Implement workload caps (max 4–5 projects per employee) | Reduce burnout-driven attrition |
| **2** | Review compensation for high-performing, low-satisfaction employees | Retain top talent before they leave |
| **3** | Introduce wellness programs for employees working 250+ hrs/month | Address work-life balance concerns |
| **4** | Conduct stay interviews at the 2–3 year tenure mark | Proactively address concerns before the 3–5 year attrition window |
| **5** | Revise evaluation criteria to reward quality over quantity | Ensure fair recognition for all employees |

---

## Deliverables Checklist

| # | Deliverable | File Location |
|---|---|---|
| 1 | Data object classification table | `outputs/partA_classification.csv` |
| 2 | Statistical summary table (mean, median, mode, variance, SD, IQR) | `outputs/partB_statistical_summary.csv` |
| 3 | Q-Q plot, histogram, and box plot for `satisfaction_level` (combined) | `outputs/partC1_univariate_satisfaction.png` |
| 4 | Q-Q plot (individual) | `outputs/C1c_qqplot_satisfaction.png` |
| 5 | Histogram (individual) | `outputs/C1a_histogram_satisfaction.png` |
| 6 | Box plot (individual) | `outputs/C1b_boxplot_satisfaction.png` |
| 7 | Pair plot (Seaborn, color-coded by `left`) | `outputs/partC2_pairplot.png` |
| 8 | Correlation scatter plot | `outputs/partC3_correlation_scatter.png` |
| 9 | Full analytics dashboard (combined) | `outputs/partD_dashboard.png` |
| 10 | Dashboard Chart 1: Attrition by Department | `outputs/D1_barchart_attrition_by_dept.png` |
| 11 | Dashboard Chart 2: Satisfaction by Salary | `outputs/D2_histogram_satisfaction_by_salary.png` |
| 12 | Dashboard Chart 3: Evaluation vs Hours | `outputs/D3_scatter_evaluation_vs_hours.png` |
| 13 | Dashboard Chart 4: Satisfaction by Dept Box Plot | `outputs/D4_boxplot_satisfaction_by_dept.png` |
| 14 | This report (Markdown) | `outputs/assignment2_report.md` |
| 15 | This report (HTML) | `outputs/assignment2_report.html` |

---

## Conclusion

This analysis demonstrates that the HR dataset reveals clear, actionable patterns in employee attrition. Using only Python-based tools (Pandas, NumPy, Matplotlib, Seaborn, SciPy), we successfully replicated all functionality typically requiring Tableau or Power BI, including interactive dashboards, statistical summaries, and multivariate visualizations.

The central finding — that **high-performing, overworked employees are the most likely to leave** — should be a priority for HR intervention. Addressing workload balance, recognition, and compensation for this group could significantly reduce attrition rates.

---

*Report generated automatically using Python. All statistics computed from HR_comma_sep.csv (14,995 records).*
