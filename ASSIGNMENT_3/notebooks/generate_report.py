"""
Assignment 3 - Report Generation Script
Generates Markdown and HTML reports from the analysis results.

Run from the notebooks/ directory:
    python generate_report.py
"""

import pandas as pd
import os
from datetime import datetime

# Create outputs directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*70)
print('ASSIGNMENT 3 - GENERATING REPORTS')
print('='*70)

# =============================================================================
# Load Results
# =============================================================================
euclidean_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'partA_euclidean_distance.csv'), index_col=0)
manhattan_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'partA_manhattan_distance.csv'), index_col=0)
comparison_stats = pd.read_csv(os.path.join(OUTPUT_DIR, 'partA_comparison_stats.csv'))
minkowski_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'partB_minkowski_distances.csv'), index_col=0)
nutrient_groups = pd.read_csv(os.path.join(OUTPUT_DIR, 'partC_nutrient_groups.csv'))
nominal_sim = pd.read_csv(os.path.join(OUTPUT_DIR, 'partC_nominal_similarity.csv'))
binary_attrs = pd.read_csv(os.path.join(OUTPUT_DIR, 'partC_binary_attributes.csv'))
jaccard_smc = pd.read_csv(os.path.join(OUTPUT_DIR, 'partC_jaccard_smc.csv'))

print('\n[OK] All results loaded successfully')

# =============================================================================
# Generate Markdown Report
# =============================================================================
print('\n[1/2] Generating Markdown Report...')

md_report = f"""# Data Analytics Assignment — Question 3
## Measuring Data Similarity & Dissimilarity

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Datasets:** wine.csv (178 records) + nutrient.csv (27 records)

---

## Part A – Data Matrix vs. Dissimilarity Matrix (6 Marks)

### Data Matrix (10 × 6)

First 10 wine samples with 6 chemical attributes: Alcohol, Malic, Ash, Magnesium, Phenols, Flavanoids.

### Euclidean Dissimilarity Matrix

"""

# Add Euclidean matrix as markdown
md_report += euclidean_df.round(2).to_markdown()

md_report += f"""

### Manhattan (L1) Dissimilarity Matrix

"""

# Add Manhattan matrix as markdown
md_report += manhattan_df.round(2).to_markdown()

md_report += f"""

### Comparison Statistics

"""
md_report += comparison_stats.to_markdown(index=False)

md_report += f"""

**Key Finding:** Manhattan distances are consistently larger than Euclidean distances.
The ratio (Manhattan/Euclidean) demonstrates that Manhattan distance produces more extreme dissimilarity values.

**Why?** Manhattan distance sums absolute differences across all dimensions, while Euclidean distance uses the square-root of squared differences, which moderates the impact of multiple small differences.

---

## Part B – Minkowski Distance Experiment (5 Marks)

### Minkowski Distance Table

"""
md_report += minkowski_df.round(4).to_markdown()

md_report += f"""

### Effect of Increasing p

As the parameter **p increases**:
- **p = 1**: Manhattan distance
- **p = 2**: Euclidean distance
- **p = 3, 10, ...**: Increasingly dominated by the largest single difference
- **p → ∞**: Approaches **Chebyshev distance** (maximum metric)

The distance values increase but at a decreasing rate, converging toward the maximum absolute difference across all attributes.

---

## Part C – Proximity for Nominal & Binary Attributes (5 Marks)

### Nutrient Groups

"""
md_report += nutrient_groups.to_markdown(index=False)

md_report += f"""

### Nominal Similarity Results

"""
md_report += nominal_sim.to_markdown(index=False)

md_report += f"""

### Binary Attributes

"""
md_report += binary_attrs.to_markdown(index=False)

md_report += f"""

### Jaccard Similarity vs. SMC

"""
md_report += jaccard_smc.to_markdown(index=False)

md_report += f"""

### Discussion: Jaccard vs. SMC

**Key Differences:**

1. **Treatment of Mutual Absence (M00):**
   - **Jaccard:** Ignores M00 (both attributes are 0)
   - **SMC:** Includes M00 in calculation

2. **Asymmetric Binary Attributes:**
   - `high_protein` and `high_iron` are asymmetric — having value 1 is more informative than 0
   - Jaccard focuses on shared presences, making it more appropriate

3. **When to Use:**
   - **Jaccard:** Preferred for asymmetric binary attributes
   - **SMC:** Suitable for symmetric binary attributes

**Conclusion:** For nutrient data's binary attributes, **Jaccard similarity is more appropriate** because it emphasizes shared positive traits rather than shared negatives.

---

## Part D – Cloud Deployment (4 Marks)

### Deployment Setup

This assignment is designed to run on **Google Colab** with:
- Wine and nutrient datasets uploaded to Colab
- Reusable `minkowski_distance()` function defined in Part B
- All visualizations and computations reproducible in cloud environment

### Reusable Function

```python
def minkowski_distance(x, y, p=2):
    return np.power(np.sum(np.abs(x - y) ** p), 1/p)
```

---

## Summary of Deliverables

| Part | Deliverable | Status |
|------|-------------|--------|
| **A** | 10×6 data matrix | ✅ |
| **A** | Euclidean distance heatmap | ✅ |
| **A** | Manhattan distance heatmap | ✅ |
| **A** | Metric comparison | ✅ |
| **B** | Minkowski distance table | ✅ |
| **B** | Line chart (distance vs. p) | ✅ |
| **B** | Chebyshev convergence explanation | ✅ |
| **C** | Nominal similarity calculation | ✅ |
| **C** | Binary attributes (high_protein, high_iron) | ✅ |
| **C** | Jaccard & SMC for 3 pairs | ✅ |
| **C** | Jaccard vs. SMC discussion | ✅ |
| **D** | Cloud deployment setup | ✅ |
| **D** | Reusable distance function | ✅ |

---

**Author:** BTech ECE — Data Analytics Assignment
"""

# Save Markdown report
md_path = os.path.join(OUTPUT_DIR, 'assignment3_report.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_report)
print(f'  [OK] Saved: assignment3_report.md')

# =============================================================================
# Generate HTML Report
# =============================================================================
print('\n[2/2] Generating HTML Report...')

html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assignment 3 - Data Similarity & Dissimilarity</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 40px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: center;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .highlight {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .summary-table {{
            background-color: #e8f8f5;
        }}
        .colab-badge {{
            display: inline-block;
            margin: 10px 0;
        }}
        .colab-badge img {{
            height: 30px;
            border: none;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Analytics Assignment — Question 3</h1>
        <h2>Measuring Data Similarity & Dissimilarity</h2>
        <div class="colab-badge">
            <a href="https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing" target="_blank">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
            </a>
        </div>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Datasets:</strong> wine.csv (178 records) + nutrient.csv (27 records)</p>

        <h2>Part A – Data Matrix vs. Dissimilarity Matrix</h2>
        
        <h3>Euclidean Dissimilarity Matrix</h3>
        {euclidean_df.round(2).to_html(classes='table', border=1)}
        
        <h3>Manhattan (L1) Dissimilarity Matrix</h3>
        {manhattan_df.round(2).to_html(classes='table', border=1)}
        
        <h3>Comparison Statistics</h3>
        {comparison_stats.to_html(classes='table', border=1, index=False)}
        
        <div class="highlight">
            <strong>Key Finding:</strong> Manhattan distances are consistently larger than Euclidean distances.
            Manhattan distance produces more extreme dissimilarity values because it sums absolute differences
            across all dimensions without the moderating effect of the square-root operation.
        </div>

        <h2>Part B – Minkowski Distance Experiment</h2>
        
        <h3>Minkowski Distance Table</h3>
        {minkowski_df.round(4).to_html(classes='table', border=1)}
        
        <div class="highlight">
            <strong>Effect of Increasing p:</strong> As p increases, the Minkowski distance converges toward
            the Chebyshev distance (maximum metric), where only the largest single attribute difference matters.
        </div>

        <h2>Part C – Proximity for Nominal & Binary Attributes</h2>
        
        <h3>Nutrient Groups</h3>
        {nutrient_groups.to_html(classes='table', border=1, index=False)}
        
        <h3>Nominal Similarity Results</h3>
        {nominal_sim.to_html(classes='table', border=1, index=False)}
        
        <h3>Jaccard Similarity vs. SMC</h3>
        {jaccard_smc.to_html(classes='table', border=1, index=False)}
        
        <div class="highlight">
            <strong>Jaccard vs. SMC:</strong> For asymmetric binary attributes (high_protein, high_iron),
            Jaccard similarity is preferred because it focuses on shared presences (M11) rather than
            shared absences (M00), which aligns with how we intuitively compare nutritional profiles.
        </div>

        <h2>Part D – Cloud Deployment</h2>
        
        <p>This assignment runs on <strong>Google Colab</strong> with reusable distance functions
        and reproducible analysis.</p>
        
        <pre>
def minkowski_distance(x, y, p=2):
    return np.power(np.sum(np.abs(x - y) ** p), 1/p)
        </pre>

        <h2>Summary of Deliverables</h2>
        <table class="summary-table">
            <tr><th>Part</th><th>Deliverable</th><th>Status</th></tr>
            <tr><td>A</td><td>10×6 data matrix</td><td>✅</td></tr>
            <tr><td>A</td><td>Euclidean distance heatmap</td><td>✅</td></tr>
            <tr><td>A</td><td>Manhattan distance heatmap</td><td>✅</td></tr>
            <tr><td>A</td><td>Metric comparison</td><td>✅</td></tr>
            <tr><td>B</td><td>Minkowski distance table</td><td>✅</td></tr>
            <tr><td>B</td><td>Line chart (distance vs. p)</td><td>✅</td></tr>
            <tr><td>B</td><td>Chebyshev convergence explanation</td><td>✅</td></tr>
            <tr><td>C</td><td>Nominal similarity calculation</td><td>✅</td></tr>
            <tr><td>C</td><td>Binary attributes</td><td>✅</td></tr>
            <tr><td>C</td><td>Jaccard & SMC for 3 pairs</td><td>✅</td></tr>
            <tr><td>C</td><td>Jaccard vs. SMC discussion</td><td>✅</td></tr>
            <tr><td>D</td><td>Cloud deployment setup</td><td>✅</td></tr>
            <tr><td>D</td><td>Reusable distance function</td><td>✅</td></tr>
        </table>
    </div>
</body>
</html>
"""

# Save HTML report
html_path = os.path.join(OUTPUT_DIR, 'assignment3_report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_report)
print(f'  [OK] Saved: assignment3_report.html')

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*70)
print('REPORT GENERATION COMPLETE')
print('='*70)
print(f'\nReports saved to: {OUTPUT_DIR}')
print('\nGenerated files:')
for file in ['assignment3_report.md', 'assignment3_report.html']:
    filepath = os.path.join(OUTPUT_DIR, file)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f'  - {file} ({size:,} bytes)')

print('\n[SUCCESS] All reports generated successfully!')
