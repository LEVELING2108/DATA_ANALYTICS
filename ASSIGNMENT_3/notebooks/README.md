# Assignment 3 — Measuring Data Similarity & Dissimilarity

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing)

## Overview
This assignment computes proximity measures on two datasets:
- **Wine dataset** (178 records, all continuous chemical measurements)
- **Nutrient dataset** (27 records, mixed nominal and continuous attributes)

## How to Run

### Option 1: Google Colab (Recommended - No Setup Required)
**Direct Link:** [Open in Google Colab](https://colab.research.google.com/drive/1AIDmsXjo35dPCS3AuQZA397vHeVQeJdg?usp=sharing)

1. Click the "Open in Colab" badge above
2. Run all cells in order (Runtime → Run all)
3. All outputs (CSV, PNG, reports) will be generated automatically
4. Download results from the outputs/ folder or Google Drive

### Option 2: Local Python Scripts
```bash
# Navigate to notebooks directory
cd ASSIGNMENT_3/notebooks

# Run full analysis (generates all CSV outputs)
python assignment3_analysis.py

# Generate all plots
python generate_plots.py

# Generate Markdown and HTML reports
python generate_report.py
```

## Directory Structure
```
ASSIGNMENT_3/
├── data/                          # Raw datasets
│   ├── wine.csv
│   └── nutrient.csv
├── notebooks/                     # Analysis code
│   ├── similarity_dissimilarity_assignment.ipynb  # Main notebook
│   ├── assignment3_analysis.py    # Main analysis script
│   ├── generate_plots.py          # Plot generation
│   └── generate_report.py         # Report generation
├── outputs/                       # Generated results
│   ├── partA_euclidean_heatmap.png
│   ├── partA_manhattan_heatmap.png
│   ├── partA_comparison_heatmaps.png
│   ├── partB_minkowski_chart.png
│   ├── partC_nutrient_groups.png
│   ├── partC_binary_attributes.png
│   ├── partA_euclidean_distance.csv
│   ├── partA_manhattan_distance.csv
│   ├── partA_comparison_stats.csv
│   ├── partB_minkowski_distances.csv
│   ├── partC_nutrient_groups.csv
│   ├── partC_nominal_similarity.csv
│   ├── partC_binary_attributes.csv
│   ├── partC_jaccard_smc.csv
│   ├── assignment3_report.md
│   └── assignment3_report.html
└── scripts/                       # (Empty - no additional scripts needed)
```

## Assignment Parts

### Part A — Data Matrix vs. Dissimilarity Matrix (6 Marks)
- 10×6 data matrix from wine dataset
- Euclidean distance matrix with heatmap
- Manhattan distance matrix with heatmap
- Visual comparison and analysis

### Part B — Minkowski Distance Experiment (5 Marks)
- Pairwise Minkowski distances for p = {1, 2, 3, 10}
- Line chart showing distance vs. parameter p
- Explanation of convergence to Chebyshev distance

### Part C — Proximity for Nominal & Binary Attributes (5 Marks)
- Energy/fat group categorization (Low/Medium/High)
- Simple matching similarity for nominal attributes
- Binary attributes: high_protein, high_iron
- Jaccard similarity and SMC comparison

### Part D — Cloud Deployment (4 Marks)
- Google Colab setup
- Reusable Minkowski distance function
- Cloud environment verification

## Key Findings
- Manhattan distances are ~15% larger than Euclidean on average
- Minkowski distance converges to Chebyshev distance as p→∞
- Jaccard similarity is preferred for asymmetric binary attributes
