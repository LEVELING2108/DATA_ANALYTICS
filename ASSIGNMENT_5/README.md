# Assignment 5 — Exploratory Data Analysis & Pattern Mining (Satellite Dataset)

## Overview
This assignment implements EDA, clustering, and classification on the Satellite dataset (6,435 instances, 36 spectral features). It covers K-Means clustering, Decision Tree classification, and outlier detection using Isolation Forest.

## Parts
| Part | Task | Description |
|------|------|-------------|
| **A** | Initial EDA | Data loading, class distribution (pie/bar charts), missing value check |
| **B** | Clustering | K-Means with Elbow method (k=2-10), silhouette score evaluation |
| **C** | Classification | Decision Tree training, confusion matrix, and outlier detection |
| **D** | Visualization | Dashboard insights and spectral separability analysis |

## Colab Link
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qvlruii_o9K_WHOYnZGZ51hNH0sIdF_8?usp=sharing)

## Files
- `notebooks/Assignment5_Colab.ipynb` — Jupyter notebook with full analysis
- `notebooks/assignment5_analysis.py` — Standalone Python script for local execution
- `outputs/` — Generated plots (`partA_class_distribution.png`, etc.) and `assignment5_report.md`
- `data/` — Local dataset (if applicable)

## How to Run
```bash
# Using Jupyter
jupyter notebook ASSIGNMENT_5/notebooks/Assignment5_Colab.ipynb

# Using Python
python ASSIGNMENT_5/notebooks/assignment5_analysis.py
```
