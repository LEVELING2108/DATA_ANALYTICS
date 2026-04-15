"""
Generate all visualizations for the insurance assignment.
Run: python generate_plots.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and clean data from Google Drive
url = "https://drive.google.com/file/d/1oyN6CXzbJq42dL5Jqkn1cP83Hu93CD6q/view?usp=sharing"
df = pd.read_csv(url)
df = df.drop_duplicates()
df['smoker_encoded'] = df['smoker'].map({'yes': 1, 'no': 0})

# ============================================================
# Plot 1: Scatter — BMI vs. Charges (color-coded by smoker)
# ============================================================
plt.figure(figsize=(10, 6))
smokers = df[df['smoker'] == 'yes']
non_smokers = df[df['smoker'] == 'no']
plt.scatter(non_smokers['bmi'], non_smokers['charges'],
            c='blue', label='Non-Smoker', alpha=0.5, s=30)
plt.scatter(smokers['bmi'], smokers['charges'],
            c='red', label='Smoker', alpha=0.5, s=30)
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Insurance Charges ($)', fontsize=12)
plt.title('BMI vs. Insurance Charges (Color-coded by Smoker Status)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, 'scatter_bmi_vs_charges.png')
plt.savefig(path1, dpi=150)
plt.close()
print(f'[1/3] Saved: {path1}')

# ============================================================
# Plot 2: Histogram + KDE of charges
# ============================================================
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], bins=50, kde=True, color='steelblue',
             edgecolor='black', alpha=0.7)
plt.xlabel('Insurance Charges ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Probability Distribution of Insurance Charges', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, 'histogram_charges.png')
plt.savefig(path2, dpi=150)
plt.close()
print(f'[2/3] Saved: {path2}')

# ============================================================
# Plot 3: Correlation Heatmap
# ============================================================
numeric_df = df[['age', 'bmi', 'children', 'charges', 'smoker_encoded']]
corr_matrix = numeric_df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1,
            xticklabels=['age', 'bmi', 'children', 'charges', 'smoker'],
            yticklabels=['age', 'bmi', 'children', 'charges', 'smoker'],
            cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title('Correlation Heatmap — Numeric Attributes', fontsize=14)
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(path3, dpi=150)
plt.close()
print(f'[3/3] Saved: {path3}')

# ============================================================
# Print analysis summary
# ============================================================
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
)
stacked = upper_triangle.stack()
max_pair = stacked.idxmax()
max_val = stacked.max()
min_pair = stacked.idxmin()
min_val = stacked.min()
skewness = df['charges'].skew()

print('\n===== ANALYSIS SUMMARY =====')
print(f'Dataset shape (after cleaning): {df.shape}')
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Duplicates removed: 1')
print(f'\nSkewness of charges: {skewness:.2f} (right-skewed)')
print(f'\nHighest POSITIVE correlation: {max_pair[0]} <-> {max_pair[1]}  r = {max_val:.3f}')
print(f'Highest NEGATIVE correlation: {min_pair[0]} <-> {min_pair[1]}  r = {min_val:.3f}')
print('\nAll plots generated successfully.')
