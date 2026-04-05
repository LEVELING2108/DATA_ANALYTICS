"""
Individual Plot Generator for Assignment 2
Saves each chart from Part C1 and Part D as a separate high-quality image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# ============================================================
# SETUP
# ============================================================
DATA_PATH = os.path.join('..', 'data', 'HR_comma_sep.csv')
OUTPUT_DIR = os.path.join('..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = ['satisfaction_level', 'last_evaluation', 'number_project',
              'average_montly_hours', 'time_spend_company', 'Work_accident',
              'left', 'promotion_last_5years', 'Department', 'salary']

print("Generating individual plots...")

# ============================================================
# PART C1 — INDIVIDUAL PLOTS
# ============================================================
print("\n--- Part C1: Individual Univariate Plots ---")

# C1-A: Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['satisfaction_level'], bins=30, edgecolor='black', color='skyblue', alpha=0.85)
plt.axvline(df['satisfaction_level'].mean(), color='red', linestyle='--', linewidth=2,
            label=f"Mean: {df['satisfaction_level'].mean():.3f}")
plt.axvline(df['satisfaction_level'].median(), color='green', linestyle='--', linewidth=2,
            label=f"Median: {df['satisfaction_level'].median():.3f}")
plt.title('Histogram of Employee Satisfaction Level', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'C1a_histogram_satisfaction.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ C1a_histogram_satisfaction.png")

# C1-B: Box Plot
plt.figure(figsize=(6, 7))
bp = plt.boxplot(df['satisfaction_level'], vert=True, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', edgecolor='navy', linewidth=2),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(color='navy', linewidth=1.5),
                 capprops=dict(color='navy', linewidth=1.5))
plt.title('Box Plot of Employee Satisfaction Level', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Satisfaction Level', fontsize=12)
plt.xticks([1], ['Satisfaction Level'], fontsize=11)

# Add statistics annotation
Q1 = df['satisfaction_level'].quantile(0.25)
Q3 = df['satisfaction_level'].quantile(0.75)
IQR = Q3 - Q1
stats_text = f"Median: {df['satisfaction_level'].median():.3f}\nQ1: {Q1:.3f}\nQ3: {Q3:.3f}\nIQR: {IQR:.3f}\nMin: {df['satisfaction_level'].min():.2f}\nMax: {df['satisfaction_level'].max():.2f}"
plt.gca().text(1.35, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray'))

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'C1b_boxplot_satisfaction.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ C1b_boxplot_satisfaction.png")

# C1-C: Q-Q Plot
fig, ax = plt.subplots(figsize=(8, 7))
stats.probplot(df['satisfaction_level'], dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor('steelblue')
ax.get_lines()[0].set_markeredgecolor('white')
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color('red')
ax.get_lines()[1].set_linewidth(2)
ax.set_title('Q-Q Plot of Employee Satisfaction Level', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Theoretical Quantiles', fontsize=12)
ax.set_ylabel('Sample Quantiles', fontsize=12)
ax.grid(True, alpha=0.3)

# Add interpretation note
skewness = df['satisfaction_level'].skew()
note = (f"Skewness: {skewness:.3f}\n"
        f"Deviation from diagonal\n"
        f"indicates non-normality.\n"
        f"Bimodal distribution\n"
        f"confirmed.")
ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8, edgecolor='#856404'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'C1c_qqplot_satisfaction.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ C1c_qqplot_satisfaction.png")

# ============================================================
# PART D — INDIVIDUAL DASHBOARD CHARTS
# ============================================================
print("\n--- Part D: Individual Dashboard Charts ---")

# D1: Bar Chart — Left vs Stayed by Department
plt.figure(figsize=(12, 7))
dept_left = pd.crosstab(df['Department'], df['left'])
dept_left.columns = ['Stayed', 'Left']
dept_left.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'],
               edgecolor='black', width=0.7)
plt.title('Employees Left vs Stayed by Department', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Employee Count', fontsize=12)
plt.legend(title='Employment Status', fontsize=11)
plt.xticks(rotation=40, ha='right', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'D1_barchart_attrition_by_dept.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ D1_barchart_attrition_by_dept.png")

# D2: Histogram — Satisfaction Level by Salary Band
plt.figure(figsize=(11, 7))
colors = {'low': '#e74c3c', 'medium': '#f39c12', 'high': '#2ecc71'}
for salary_level in ['low', 'medium', 'high']:
    subset = df[df['salary'] == salary_level]['satisfaction_level']
    plt.hist(subset, bins=25, alpha=0.6, label=f'{salary_level.capitalize()} Salary',
             color=colors[salary_level], edgecolor='black', linewidth=0.5)

plt.axvline(df['satisfaction_level'].mean(), color='black', linestyle='--', linewidth=2,
            label=f'Overall Mean: {df["satisfaction_level"].mean():.3f}')
plt.title('Satisfaction Level Distribution by Salary Band', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Salary Band', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'D2_histogram_satisfaction_by_salary.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ D2_histogram_satisfaction_by_salary.png")

# D3: Scatter Plot — Last Evaluation vs Monthly Hours
plt.figure(figsize=(11, 7))
df_plot = df.sample(n=5000, random_state=42)
scatter = plt.scatter(df_plot['average_montly_hours'], df_plot['last_evaluation'],
                      c=df_plot['left'], cmap='coolwarm', alpha=0.5, s=35,
                      edgecolors='none')
plt.title('Last Evaluation vs Average Monthly Hours\n(Color-coded by Attrition)',
          fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Average Monthly Hours', fontsize=12)
plt.ylabel('Last Evaluation Score', fontsize=12)
legend = plt.legend(*scatter.legend_elements(num=2),
                    title="Left (0 = Stayed, 1 = Left)",
                    fontsize=11, title_fontsize=12, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'D3_scatter_evaluation_vs_hours.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ D3_scatter_evaluation_vs_hours.png")

# D4: Box Plot — Satisfaction by Department & Attrition
plt.figure(figsize=(14, 8))
df_box = df.copy()
df_box['Status'] = df_box['left'].map({0: 'Stayed', 1: 'Left'})
sns.boxplot(data=df_box, x='Department', y='satisfaction_level', hue='Status',
            palette={'Stayed': '#2ecc71', 'Left': '#e74c3c'},
            flierprops=dict(marker='o', markersize=3, alpha=0.2, markerfacecolor='gray'))
plt.title('Satisfaction Level by Department & Attrition Status', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Satisfaction Level', fontsize=12)
plt.xticks(rotation=40, ha='right', fontsize=10)
plt.legend(title='Employment Status', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'D4_boxplot_satisfaction_by_dept.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ D4_boxplot_satisfaction_by_dept.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ALL INDIVIDUAL PLOTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"""
Part C1 — Univariate Plots:
  1. C1a_histogram_satisfaction.png
  2. C1b_boxplot_satisfaction.png
  3. C1c_qqplot_satisfaction.png

Part D — Dashboard Charts:
  4. D1_barchart_attrition_by_dept.png
  5. D2_histogram_satisfaction_by_salary.png
  6. D3_scatter_evaluation_vs_hours.png
  7. D4_boxplot_satisfaction_by_dept.png

All files saved in: {OUTPUT_DIR}/
""")
