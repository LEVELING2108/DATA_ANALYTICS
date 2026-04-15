# Assignment 3 — Measuring Data Similarity & Dissimilarity

**Course:** Data Analytics | **Program:** B.Tech ECE
**Datasets:** [wine.csv](https://drive.google.com/uc?export=download&id=1_2u0kE9Q5W4m3S7R8-X0N1o2p3L4K5J6) and [nutrient.csv](https://drive.google.com/uc?export=download&id=1_3v1lF0R6X5n4T8S9-Y1O2p3q4M5N6O7) (Loaded from Google Drive)
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy
**Date:** 06 April 2026
**Marks:** [20 Marks]

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing)

---

## Executive Summary

This report presents a comprehensive analysis of data proximity measures across two contrasting datasets: the wine dataset (all continuous chemical measurements) and the nutrient dataset (mixed nominal and continuous nutritional attributes). The analysis covers Euclidean and Manhattan dissimilarity matrices, Minkowski distance experiments with varying parameter p, nominal and binary attribute similarity measures, and cloud deployment on Google Colab.

**Key Finding:** Manhattan distances are consistently ~15% larger than Euclidean distances on the wine data (mean ratio 1.154), and Minkowski distance converges toward Chebyshev distance as p → ∞, demonstrating that the largest single attribute difference dominates dissimilarity at high p values.

---

## Table of Contents

1. [Part A — Data Matrix vs. Dissimilarity Matrix](#part-a--data-matrix-vs-dissimilarity-matrix)
2. [Part B — Minkowski Distance Experiment](#part-b--minkowski-distance-experiment)
3. [Part C — Proximity for Nominal & Binary Attributes](#part-c--proximity-for-nominal--binary-attributes)
4. [Part D — Cloud Deployment](#part-d--cloud-deployment)
5. [Key Insights & Conclusions](#key-insights--conclusions)

---

## Part A — Data Matrix vs. Dissimilarity Matrix [6 Marks]

### A1. Data Matrix (10 × 6)

The first 10 wine samples were selected with six numeric chemical attributes: Alcohol, Malic Acid, Ash, Magnesium, Phenols, and Flavanoids.

| Sample | Alcohol | Malic | Ash | Magnesium | Phenols | Flavanoids |
|---|---|---|---|---|---|---|
| Wine 1 | 14.23 | 1.71 | 2.43 | 127 | 2.80 | 3.06 |
| Wine 2 | 13.20 | 1.78 | 2.14 | 100 | 2.65 | 2.76 |
| Wine 3 | 13.16 | 2.36 | 2.67 | 101 | 2.80 | 3.24 |
| Wine 4 | 14.37 | 1.95 | 2.50 | 113 | 3.85 | 3.49 |
| Wine 5 | 13.24 | 2.59 | 2.87 | 118 | 2.80 | 2.69 |
| Wine 6 | 14.20 | 1.76 | 2.45 | 112 | 3.27 | 3.39 |
| Wine 7 | 14.39 | 1.87 | 2.45 | 96 | 2.50 | 2.52 |
| Wine 8 | 14.06 | 2.15 | 2.61 | 121 | 2.60 | 2.51 |
| Wine 9 | 14.83 | 1.64 | 2.17 | 97 | 2.80 | 2.98 |
| Wine 10 | 13.86 | 1.35 | 2.27 | 98 | 2.98 | 3.15 |

**Matrix Properties:**
- **Rows:** 10 observations (wine samples)
- **Columns:** 6 attributes (chemical measurements)
- **Total data points:** 60
- **Attribute scales vary widely:** Magnesium (96–127) vs. Flavanoids (2.51–3.49)

> 📊 **See:** Full data matrix in `outputs/partA_euclidean_distance.csv`

---

### A2. Euclidean Dissimilarity Matrix

The 10 × 10 Euclidean distance matrix was computed using `scipy.spatial.distance.cdist` with metric `'euclidean'`.

**Formula:** $d_E(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

#### Distance Statistics

| Metric | Value |
|---|---|
| **Mean distance** | 13.392 |
| **Max distance** | 31.007 (Wine 1 vs Wine 7) |
| **Min distance** | 1.189 (Wine 4 vs Wine 6) |
| **Std deviation** | 8.742 |

**Key Observations:**
- **Most similar pair:** Wine 4 and Wine 6 (d = 1.189) — both have similar chemical profiles
- **Most dissimilar pair:** Wine 1 and Wine 7 (d = 31.007) — significantly different across multiple attributes
- **Diagonal is zero:** Each sample has zero distance to itself (identity property)
- **Matrix is symmetric:** d(x,y) = d(y,x) (symmetry property)

> 📊 **See:** `outputs/partA_euclidean_heatmap.png`

---

### A3. Manhattan (L1) Dissimilarity Matrix

The same 10 × 10 matrix was computed using Manhattan (city block) distance.

**Formula:** $d_M(x, y) = \sum_{i=1}^{n} |x_i - y_i|$

#### Distance Statistics

| Metric | Value |
|---|---|
| **Mean distance** | 15.460 |
| **Max distance** | 32.180 (Wine 1 vs Wine 7) |
| **Min distance** | 2.090 (Wine 4 vs Wine 6) |
| **Std deviation** | 9.234 |

**Key Observations:**
- **Most similar pair:** Wine 4 and Wine 6 (d = 2.090) — consistent with Euclidean
- **Most dissimilar pair:** Wine 1 and Wine 7 (d = 32.180) — same pair as Euclidean
- **All Manhattan distances ≥ Euclidean distances** — mathematically guaranteed

> 📊 **See:** `outputs/partA_manhattan_heatmap.png`

---

### A4. Comparison: Euclidean vs. Manhattan

| Metric | Euclidean | Manhattan | Ratio (M/E) |
|---|---|---|---|
| **Mean** | 13.392 | 15.460 | 1.154 |
| **Max** | 31.007 | 32.180 | 1.038 |
| **Min** | 1.189 | 2.090 | 1.758 |
| **Std** | 8.742 | 9.234 | 1.056 |

#### Which Metric Produces More Extreme Values?

**Answer:** **Manhattan (L1) distance** produces consistently larger dissimilarity values.

**Why?**

1. **Mathematical reason:** Euclidean distance applies a square-root operation after squaring differences, which moderates the impact of multiple small differences. Manhattan distance simply sums absolute differences without any dampening effect.

2. **Geometric interpretation:** Euclidean distance measures the straight-line (shortest) path between two points. Manhattan distance measures the path along axis-aligned segments (like navigating city blocks), which is always longer or equal.

3. **High-dimensional behavior:** In multi-dimensional spaces, Manhattan distance gives equal weight to differences in all dimensions, while Euclidean distance is more influenced by large differences in fewer dimensions due to the squaring operation.

4. **Observed ratio:** The mean Manhattan/Euclidean ratio of **1.154** (15.4% larger) is consistent with theoretical expectations for 6-dimensional data with mixed-scale attributes.

> 📊 **See:** Side-by-side comparison at `outputs/partA_comparison_heatmaps.png`

---

## Part B — Minkowski Distance Experiment [5 Marks]

### B1. Minkowski Distance Table

Three wine samples were selected (Wine 1, Wine 5, Wine 9) and pairwise Minkowski distances were computed for p = {1, 2, 3, 10}.

**Formula:** $d_p(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$

| p Value | Wine 1 vs Wine 5 | Wine 1 vs Wine 9 | Wine 5 vs Wine 9 |
|---|---|---|---|
| **p = 1** (Manhattan) | 11.6800 | 31.0100 | 24.5300 |
| **p = 2** (Euclidean) | 9.1151 | 30.0073 | 21.0951 |
| **p = 3** | 9.0074 | 30.0001 | 21.0040 |
| **p = 10** | 9.0000 | 30.0000 | 21.0000 |

**Observations:**
- Distance values **increase** from p=1 to p=2, then **stabilize** for p ≥ 3
- The change from p=1 to p=2 is significant (Manhattan → Euclidean transition)
- For p ≥ 3, distances converge rapidly toward a limiting value

> 📊 **See:** `outputs/partB_minkowski_distances.csv`

---

### B2. Minkowski Distance vs. Parameter p

The line chart plots Minkowski distance on the y-axis against the parameter p on the x-axis, with one line per sample pair.

**Key Trends:**
- **Wine 1 vs Wine 5:** Distance drops from 11.68 (p=1) to 9.00 (p=10), a **22.9% decrease**
- **Wine 1 vs Wine 9:** Distance drops from 31.01 (p=1) to 30.00 (p=10), a **3.3% decrease**
- **Wine 5 vs Wine 9:** Distance drops from 24.53 (p=1) to 21.00 (p=10), a **14.4% decrease**

**Interpretation:** Pairs with larger initial differences (Wine 1 vs Wine 9) show less sensitivity to p, while pairs with moderate differences (Wine 1 vs Wine 5) show more pronounced changes.

> 📊 **See:** `outputs/partB_minkowski_chart.png`

---

### B3. Effect of Increasing p — Convergence to Chebyshev Distance

#### Mathematical Explanation

As **p → ∞**, the Minkowski distance converges to the **Chebyshev distance** (L∞ norm):

$$\lim_{p \to \infty} d_p(x, y) = \max_i(|x_i - y_i|)$$

**Why does this happen?**

When raising differences to increasingly large powers, the largest difference dominates the sum exponentially more than smaller differences. For example, if the differences are [2, 5, 1, 3]:

- At p=1: Sum = 2 + 5 + 1 + 3 = 11
- At p=2: Sum = 4 + 25 + 1 + 9 = 39 → (39)^(1/2) ≈ 6.24
- At p=10: Sum = 1024 + 9765625 + 1 + 59049 = 9825699 → (9825699)^(1/10) ≈ 5.00
- At p→∞: Result → 5 (the maximum difference)

The term 5^10 = 9,765,625 completely dominates the sum, making all other terms negligible.

#### Chebyshev Distance Values

| Pair | Chebyshev Distance (max |x_i - y_i|) | Dominant Attribute |
|---|---|---|
| Wine 1 vs Wine 5 | 9.00 | Magnesium (|127 - 118| = 9) |
| Wine 1 vs Wine 9 | 30.00 | Magnesium (|127 - 97| = 30) |
| Wine 5 vs Wine 9 | 21.00 | Magnesium (|118 - 97| = 21) |

**Verification:** The Minkowski distances at p=10 match the Chebyshev distances to 4 decimal places, confirming convergence.

#### Practical Implications

| p Value | Behavior | Use Case |
|---|---|---|
| **p = 1** | All differences weighted equally | When every attribute difference matters equally |
| **p = 2** | Moderate emphasis on larger differences | General-purpose distance (most common) |
| **p = 3–10** | Increasing focus on largest differences | When worst-case difference is most important |
| **p → ∞** | Only the maximum difference matters | Chess (king moves), warehouse logistics, worst-case analysis |

---

## Part C — Proximity for Nominal & Binary Attributes [5 Marks]

### C1. Nutrient Dataset Overview

The nutrient dataset contains 27 food items with 5 continuous attributes: energy, protein, fat, calcium, and iron.

| Food Item | Energy | Protein | Fat | Calcium | Iron |
|---|---|---|---|---|---|
| BEEF BRAISED | 340 | 20 | 28 | 9 | 2.6 |
| HAMBURGER | 245 | 21 | 17 | 9 | 2.7 |
| CHICKEN BROILED | 115 | 20 | 3 | 8 | 1.4 |
| SARDINES CANNED | 180 | 22 | 9 | 367 | 2.5 |
| CLAMS RAW | 70 | 11 | 1 | 82 | 6.0 |

> 📊 **Full dataset:** `outputs/partC_nutrient_groups.csv`

---

### C2. Nominal Attribute Similarity (Task 25)

#### Step 1: Categorize Energy and Fat into Groups

Energy and fat values were divided into **Low / Medium / High** groups using tertiles (3-quantiles).

**Energy Group Distribution:**

| Group | Count | Range (kcal) | Example Foods |
|---|---|---|---|
| **Low** | 9 | 45–135 | Clams, Shrimp, Chicken Broiled |
| **Medium** | 9 | 155–200 | Mackerel, Tuna, Pork Simmered |
| **High** | 9 | 245–420 | Beef, Lamb, Hamburger |

**Fat Group Distribution:**

| Group | Count | Range (g) | Example Foods |
|---|---|---|---|
| **Low** | 9 | 1–5 | Clams, Shrimp, Chicken Broiled |
| **Medium** | 9 | 7–14 | Tuna, Mackerel, Beef Canned |
| **High** | 9 | 17–39 | Beef, Lamb, Pork |

> 📊 **See:** `outputs/partC_nutrient_groups.png`

---

#### Step 2: Compute Simple Matching Similarity

**Formula:** $Similarity = \frac{\text{Number of matching attributes}}{\text{Total attributes}}$

**Pair 1: BEEF BRAISED vs HAMBURGER**

| Attribute | BEEF BRAISED | HAMBURGER | Match? |
|---|---|---|---|
| Energy Group | High | High | ✓ (1) |
| Fat Group | High | High | ✓ (1) |
| **Total** | | | **2/2 = 1.00** |

**Result:** Similarity = **1.00** (identical groups)

---

**Pair 2: BEEF BRAISED vs BEEF ROAST**

| Attribute | BEEF BRAISED | BEEF ROAST | Match? |
|---|---|---|---|
| Energy Group | High | High | ✓ (1) |
| Fat Group | High | High | ✓ (1) |
| **Total** | | | **2/2 = 1.00** |

**Result:** Similarity = **1.00** (identical groups)

---

**Pair 3: HAMBURGER vs CHICKEN BROILED**

| Attribute | HAMBURGER | CHICKEN BROILED | Match? |
|---|---|---|---|
| Energy Group | High | Low | ✗ (0) |
| Fat Group | High | Low | ✗ (0) |
| **Total** | | | **0/2 = 0.00** |

**Result:** Similarity = **0.00** (completely different groups)

> 📊 **See:** `outputs/partC_nominal_similarity.csv`

---

### C3. Binary Attributes — Jaccard & SMC (Task 26)

#### Step 1: Create Binary Attributes

Binary attributes were created using median thresholds:

| Attribute | Median Threshold | Rule |
|---|---|---|
| **high_protein** | 19.00 | 1 if protein > 19, else 0 |
| **high_iron** | 2.50 | 1 if iron > 2.5, else 0 |

**Distribution:**

| Binary Attribute | Count (0) | Count (1) | Interpretation |
|---|---|---|---|
| **high_protein** | 14 | 13 | Nearly balanced split |
| **high_iron** | 15 | 12 | Slightly more low-iron foods |

**Example Classifications:**

| Food Item | Protein | high_protein | Iron | high_iron |
|---|---|---|---|---|
| BEEF BRAISED | 20 | 1 | 2.6 | 1 |
| HAMBURGER | 21 | 1 | 2.7 | 1 |
| CHICKEN BROILED | 20 | 1 | 1.4 | 0 |
| SARDINES CANNED | 22 | 1 | 2.5 | 0 |
| CLAMS RAW | 11 | 0 | 6.0 | 1 |

> 📊 **See:** `outputs/partC_binary_attributes.csv` and `outputs/partC_binary_attributes.png`

---

#### Step 2: Compute Jaccard Similarity and SMC

**Notation:**
- **M11:** Both items have value 1 (both high)
- **M00:** Both items have value 0 (both not high)
- **M01:** Item A = 0, Item B = 1
- **M10:** Item A = 1, Item B = 0

**Formulas:**

| Measure | Formula | Treatment of M00 |
|---|---|---|
| **Jaccard** | $J = \frac{M_{11}}{M_{01} + M_{10} + M_{11}}$ | **Ignores** M00 |
| **SMC** | $SMC = \frac{M_{11} + M_{00}}{M_{00} + M_{01} + M_{10} + M_{11}}$ | **Includes** M00 |

---

**Pair 1: BEEF BRAISED vs HAMBURGER**

| | high_protein | high_iron |
|---|---|---|
| BEEF BRAISED | 1 | 1 |
| HAMBURGER | 1 | 1 |

| Metric | Value | Calculation |
|---|---|---|
| M11 | 2 | Both high in protein AND iron |
| M00 | 0 | No mutual absences |
| M01 | 0 | No mismatches |
| M10 | 0 | No mismatches |
| **Jaccard** | **1.0000** | 2 / (0 + 0 + 2) = 1.0 |
| **SMC** | **1.0000** | (2 + 0) / (0 + 0 + 0 + 2) = 1.0 |

**Interpretation:** Both items are identical in binary attributes — perfect similarity.

---

**Pair 2: BEEF BRAISED vs CHICKEN BROILED**

| | high_protein | high_iron |
|---|---|---|
| BEEF BRAISED | 1 | 1 |
| CHICKEN BROILED | 1 | 0 |

| Metric | Value | Calculation |
|---|---|---|
| M11 | 1 | Both high in protein |
| M00 | 0 | No mutual absences |
| M01 | 0 | No cases where A=0, B=1 |
| M10 | 1 | A high iron, B not |
| **Jaccard** | **0.5000** | 1 / (0 + 1 + 1) = 0.5 |
| **SMC** | **0.5000** | (1 + 0) / (0 + 0 + 1 + 1) = 0.5 |

**Interpretation:** Moderate similarity — shared high protein but differ on iron.

---

**Pair 3: CHICKEN BROILED vs SARDINES CANNED**

| | high_protein | high_iron |
|---|---|---|
| CHICKEN BROILED | 1 | 0 |
| SARDINES CANNED | 1 | 0 |

| Metric | Value | Calculation |
|---|---|---|
| M11 | 1 | Both high in protein |
| M00 | 1 | Both not high in iron |
| M01 | 0 | No mismatches |
| M10 | 0 | No mismatches |
| **Jaccard** | **1.0000** | 1 / (0 + 0 + 1) = 1.0 |
| **SMC** | **1.0000** | (1 + 1) / (1 + 0 + 0 + 1) = 1.0 |

**Interpretation:** Both items share identical binary profile — perfect similarity.

> 📊 **See:** `outputs/partC_jaccard_smc.csv`

---

### C4. Discussion: Jaccard vs. SMC for Asymmetric Binary Attributes

#### Key Differences

| Aspect | Jaccard Similarity | Simple Matching Coefficient |
|---|---|---|
| **Formula** | M11 / (M01 + M10 + M11) | (M11 + M00) / Total |
| **Treatment of M00** | Excluded (ignored) | Included (counted as match) |
| **Range** | [0, 1] | [0, 1] |
| **Best for** | Asymmetric binary attributes | Symmetric binary attributes |
| **Focus** | Shared presences only | Both shared presences and absences |

#### Why Jaccard is Preferred for Asymmetric Attributes

**Asymmetric binary attributes** are those where the presence of a trait (value = 1) is more informative than its absence (value = 0).

**In our nutrient context:**
- Being **high in protein** (M11) is a meaningful positive trait
- Being **not high in iron** (M00) is a lack of information — many foods share this

**Example scenario:** Consider two foods that are both NOT high in protein and NOT high in iron (M00 = 2, M11 = 0):
- **Jaccard:** 0 / (0 + 0 + 0) = undefined → treated as 0 (no similarity)
- **SMC:** (0 + 2) / 2 = 1.0 (perfect similarity)

The SMC would incorrectly classify two nutritionally poor foods as "similar" just because they both lack positive traits. Jaccard correctly recognizes that **shared absence is not meaningful similarity** for asymmetric attributes.

#### When to Use Each Measure

| Scenario | Recommended Measure | Reason |
|---|---|---|
| Nutritional profiles (high/low nutrients) | **Jaccard** | Presence of nutrients is more informative |
| Disease symptoms (present/absent) | **Jaccard** | Having a symptom is more significant than not having it |
| Gender (male/female) | **SMC** | Both values are equally informative (symmetric) |
| Color preferences (like/dislike) | **SMC** | Liking and disliking are equally meaningful |

**Conclusion:** For the nutrient dataset's binary attributes (high_protein, high_iron), **Jaccard similarity is the more appropriate measure** because it emphasizes shared positive nutritional traits rather than shared deficiencies.

---

## Part D — Cloud Deployment [4 Marks]

### D1. Cloud Environment Setup

This assignment was designed to run on **Google Colab** (free tier) with the following configuration:

**Environment Details:**

| Component | Specification |
|---|---|
| **Platform** | Google Colab (colab.research.google.com) |
| **Python Version** | 3.10+ |
| **Key Libraries** | pandas, numpy, matplotlib, seaborn, scipy |
| **Data Source** | Uploaded CSV files (wine.csv, nutrient.csv) |
| **Storage** | Google Drive (optional) or local Colab storage |

### D2. Reusable Minkowski Distance Function

```python
def minkowski_distance(x, y, p=2):
    """
    Compute Minkowski distance between two vectors for parameter p.
    
    Parameters:
    -----------
    x, y : array-like
        Input vectors (lists, numpy arrays, or pandas Series)
    p : int or float, default=2
        Minkowski parameter (p >= 1)
        - p=1: Manhattan distance
        - p=2: Euclidean distance
        - p→∞: Chebyshev distance
    
    Returns:
    --------
    float : Minkowski distance value
    
    Examples:
    >>> minkowski_distance([1, 2, 3], [4, 5, 6], p=1)
    9.0
    >>> minkowski_distance([1, 2, 3], [4, 5, 6], p=2)
    5.196
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.power(np.sum(np.abs(x - y) ** p), 1/p)
```

**Function Features:**
- Works with lists, numpy arrays, and pandas Series
- Default p=2 (Euclidean) for common use case
- Supports any p ≥ 1, including non-integer values
- Vectorized for efficient batch computation

### D3. Cloud Deployment Verification

The assignment includes environment verification code that:
1. Detects if running in Google Colab
2. Verifies all required libraries are available
3. Tests the reusable distance function
4. Confirms datasets are loaded correctly

> 📸 **Screenshot:** Students should capture a screenshot of the running Colab notebook showing successful execution and include it in their submission.

---

## Key Insights & Conclusions

### 1. Distance Metric Selection Matters

The choice between Euclidean and Manhattan distance affects dissimilarity values by ~15% on average. **Manhattan distance** is more appropriate when:
- Attributes are on different scales (as in the wine data)
- You want to treat all attribute differences equally
- Outliers should not be overly penalized

**Euclidean distance** is preferred when:
- Attributes are on similar scales
- Large differences in single attributes should dominate
- Geometric interpretation is meaningful

### 2. Minkowski Parameter p Controls Sensitivity

The Minkowski distance provides a **tunable proximity measure**:
- **Low p (1–2):** Considers all attribute differences — good for overall similarity
- **High p (3+):** Focuses on the worst-case difference — good for risk assessment
- **p → ∞:** Reduces to Chebyshev distance — useful when only the maximum difference matters

This tunability makes Minkowski distance a versatile tool for clustering, classification, and anomaly detection.

### 3. Nominal and Binary Attributes Require Different Measures

- **Simple matching coefficient** works well for nominal groupings (Low/Medium/High)
- **Jaccard similarity** is superior for asymmetric binary attributes where presence is more informative than absence
- **SMC** can be misleading for asymmetric data by inflating similarity through shared absences

### 4. Practical Applications

| Application | Recommended Measure | Reason |
|---|---|---|
| Wine classification | Euclidean/Minkowski (p=2) | Continuous chemical measurements |
| Food similarity | Jaccard (binary) + Simple matching (nominal) | Mixed attribute types |
| Anomaly detection | Manhattan (L1) | Robust to outliers |
| Worst-case analysis | Chebyshev (p→∞) | Focus on maximum difference |

---

## Deliverables Checklist

| # | Deliverable | File Location |
|---|---|---|
| 1 | 10 × 6 data matrix | `outputs/partA_euclidean_distance.csv` |
| 2 | Euclidean dissimilarity matrix | `outputs/partA_euclidean_distance.csv` |
| 3 | Euclidean heatmap | `outputs/partA_euclidean_heatmap.png` |
| 4 | Manhattan dissimilarity matrix | `outputs/partA_manhattan_distance.csv` |
| 5 | Manhattan heatmap | `outputs/partA_manhattan_heatmap.png` |
| 6 | Comparison visualization | `outputs/partA_comparison_heatmaps.png` |
| 7 | Minkowski distance table | `outputs/partB_minkowski_distances.csv` |
| 8 | Minkowski line chart | `outputs/partB_minkowski_chart.png` |
| 9 | Nutrient groups table | `outputs/partC_nutrient_groups.csv` |
| 10 | Nutrient groups chart | `outputs/partC_nutrient_groups.png` |
| 11 | Nominal similarity results | `outputs/partC_nominal_similarity.csv` |
| 12 | Binary attributes table | `outputs/partC_binary_attributes.csv` |
| 13 | Binary attributes chart | `outputs/partC_binary_attributes.png` |
| 14 | Jaccard & SMC results | `outputs/partC_jaccard_smc.csv` |
| 15 | This report (Markdown) | `outputs/assignment3_report.md` |
| 16 | This report (HTML) | `outputs/assignment3_report.html` |

---

## Conclusion

This assignment demonstrates the fundamental importance of selecting appropriate proximity measures based on data characteristics. Continuous attributes (wine chemistry) are well-suited to Euclidean and Minkowski distances, while nominal attributes (energy/fat groups) require simple matching, and asymmetric binary attributes (high protein/iron) are best handled by Jaccard similarity.

The Minkowski distance experiment reveals a key mathematical property: as p increases, the distance metric transitions from considering all differences equally (Manhattan) to focusing exclusively on the largest single difference (Chebyshev). This tunability makes Minkowski distance a powerful tool for data analytics applications ranging from clustering to anomaly detection.

All analysis was implemented using Python-based tools (Pandas, NumPy, Matplotlib, Seaborn, SciPy) and is fully reproducible on Google Colab, demonstrating cloud-ready, open-source data analytics workflows.

---

*Report generated automatically using Python. All statistics computed from wine.csv (178 records) and nutrient.csv (27 records).*
