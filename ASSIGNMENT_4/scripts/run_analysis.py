"""
Assignment 4 — Data Pre-processing Pipeline
Run this script to generate all outputs (plots + CSVs)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

print("=" * 70)
print("Assignment 4 — Data Pre-processing Pipeline")
print("=" * 70)

# ============================================================
# DOWNLOAD / LOAD DATASETS
# ============================================================
print("\n[1/8] Loading datasets...")

# Try to download HR dataset
if not os.path.exists('HR_comma_sep.csv'):
    print("  Downloading HR_comma_sep.csv from Kaggle/public source...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/rpi-tech/complete-guide-to-data-analytics/main/HR_comma_sep.csv"
        urllib.request.urlretrieve(url, "HR_comma_sep.csv")
        print("  HR_comma_sep.csv downloaded.")
    except Exception as e:
        print(f"  Direct download failed: {e}")
        print("  Generating synthetic HR dataset...")
        np.random.seed(42)
        n = 14999
        df_hr = pd.DataFrame({
            'satisfaction_level': np.round(np.clip(np.random.beta(2, 3, n), 0.01, 1.0), 2),
            'last_evaluation': np.round(np.clip(np.random.beta(3, 2, n), 0.36, 1.0), 2),
            'number_project': np.random.randint(2, 8, n),
            'average_montly_hours': np.random.randint(96, 311, n),
            'time_spend_company': np.random.randint(2, 11, n),
            'Work_accident': np.random.choice([0, 1], n, p=[0.86, 0.14]),
            'left': np.random.choice([0, 1], n, p=[0.76, 0.24]),
            'promotion_last_5years': np.random.choice([0, 1], n, p=[0.98, 0.02]),
            'Department': np.random.choice(['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'finance', 'hr', 'management', 'RandD'], n),
            'salary': np.random.choice(['low', 'medium', 'high'], n, p=[0.49, 0.45, 0.06])
        })
        df_hr.to_csv("HR_comma_sep.csv", index=False)
        print("  Synthetic HR_comma_sep.csv generated.")
else:
    print("  HR_comma_sep.csv found.")

# Download Sonar from UCI or alternative source
if not os.path.exists('Sonar.csv'):
    print("  Downloading Sonar.csv...")
    try:
        import urllib.request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.data'
        urllib.request.urlretrieve(url, 'sonar_raw.data')
        col_names = [f'V{i}' for i in range(1, 61)] + ['Class']
        df_sonar = pd.read_csv('sonar_raw.data', header=None, names=col_names)
        df_sonar.to_csv('Sonar.csv', index=False)
        os.remove('sonar_raw.data')
        print("  Sonar.csv downloaded from UCI.")
    except Exception as e1:
        print(f"  UCI failed: {e1}")
        try:
            import urllib.request
            url2 = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/sonar.csv'
            df_sonar = pd.read_csv(url2)
            if 'class' in df_sonar.columns:
                df_sonar = df_sonar.rename(columns={'class': 'Class'})
            df_sonar.to_csv('Sonar.csv', index=False)
            print("  Sonar.csv downloaded from GitHub/seaborn.")
        except Exception as e2:
            print(f"  GitHub failed: {e2}")
            print("  Generating synthetic Sonar-like dataset...")
            np.random.seed(42)
            n = 208
            data = {}
            for i in range(1, 61):
                data[f'V{i}'] = np.random.uniform(0, 1, n)
            data['Class'] = ['R'] * 111 + ['M'] * 97
            np.random.shuffle(data['Class'])
            df_sonar = pd.DataFrame(data)
            df_sonar.to_csv('Sonar.csv', index=False)
            print("  Synthetic Sonar.csv generated.")
else:
    print("  Sonar.csv found.")

# Load HR dataset
df_hr = pd.read_csv("HR_comma_sep.csv")
df_hr.columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                  'average_montly_hours', 'time_spend_company', 'Work_accident',
                  'left', 'promotion_last_5years', 'Department', 'salary']

# Load Sonar dataset
df_sonar = pd.read_csv('Sonar.csv')

print(f"  HR dataset: {df_hr.shape}")
print(f"  Sonar dataset: {df_sonar.shape}")

# ============================================================
# PART A — DATA CLEANING
# ============================================================
print("\n[2/8] Part A — Data Cleaning...")

# Task 27: Missing values, duplicates, outliers
missing = df_hr.isnull().sum()
missing_total = missing.sum()
duplicates = df_hr.duplicated().sum()

numeric_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company']

outlier_report = []
total_outliers = 0
for col in numeric_cols:
    Q1 = df_hr[col].quantile(0.25)
    Q3 = df_hr[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = len(df_hr[(df_hr[col] < lower_bound) | (df_hr[col] > upper_bound)])
    total_outliers += outlier_count
    outlier_report.append({
        'Column': col, 'Q1': round(Q1, 3), 'Q3': round(Q3, 3),
        'IQR': round(IQR, 3), 'Lower Bound': round(lower_bound, 3),
        'Upper Bound': round(upper_bound, 3), 'Outlier Count': outlier_count
    })

print(f"  Missing values: {missing_total}")
print(f"  Duplicates: {duplicates}")
print(f"  Total outliers (IQR): {total_outliers}")

# Task 28: Remove duplicates & winsorize
df_hr_clean = df_hr.drop_duplicates().copy()
removed = len(df_hr) - len(df_hr_clean)
print(f"  Duplicates removed: {removed}")

lower_cap = df_hr_clean['average_montly_hours'].quantile(0.05)
upper_cap = df_hr_clean['average_montly_hours'].quantile(0.95)
df_hr_clean['average_montly_hours_winsorized'] = df_hr_clean['average_montly_hours'].clip(lower=lower_cap, upper=upper_cap)

# Winsorization histogram
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(df_hr_clean['average_montly_hours'], bins=50, edgecolor='black', color='salmon', alpha=0.8)
axes[0].axvline(lower_cap, color='red', linestyle='--', linewidth=2, label=f'5th %ile ({lower_cap:.0f})')
axes[0].axvline(upper_cap, color='red', linestyle='--', linewidth=2, label=f'95th %ile ({upper_cap:.0f})')
axes[0].set_title('Before Winsorization', fontweight='bold')
axes[0].set_xlabel('Average Monthly Hours')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].hist(df_hr_clean['average_montly_hours_winsorized'], bins=50, edgecolor='black', color='teal', alpha=0.8)
axes[1].axvline(lower_cap, color='red', linestyle='--', linewidth=2, label=f'5th %ile ({lower_cap:.0f})')
axes[1].axvline(upper_cap, color='red', linestyle='--', linewidth=2, label=f'95th %ile ({upper_cap:.0f})')
axes[1].set_title('After Winsorization (5th-95th %ile)', fontweight='bold')
axes[1].set_xlabel('Average Monthly Hours')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.suptitle('Histogram Comparison: Before vs After Winsorization', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('partA_winsorization_histogram')

print(f"  Winsorization bounds: Lower={lower_cap:.1f}, Upper={upper_cap:.1f}")

# Task 29: Data type consistency
binary_cols = ['Work_accident', 'left', 'promotion_last_5years']
for col in binary_cols:
    if df_hr_clean[col].dtype != 'int64' and df_hr_clean[col].dtype != 'int32':
        df_hr_clean[col] = df_hr_clean[col].astype(int)

df_hr_clean['Department'] = df_hr_clean['Department'].astype('category')
df_hr_clean['salary'] = df_hr_clean['salary'].astype('category')
print("  Data types validated and fixed.")

# ============================================================
# PART B — DATA INTEGRATION & TRANSFORMATION
# ============================================================
print("\n[3/8] Part B — Data Integration & Transformation...")

# Task 30: Split & merge
cols_first5 = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
cols_last5 = ['Work_accident', 'left', 'promotion_last_5years', 'Department', 'salary']

df_half1 = df_hr_clean[cols_first5].copy()
df_half2 = df_hr_clean[cols_last5].copy()
df_half1['_merge_idx'] = df_half1.index
df_half2['_merge_idx'] = df_half2.index

df_merged = pd.merge(df_half1, df_half2, on='_merge_idx', how='inner')
df_merged = df_merged.drop('_merge_idx', axis=1)

assert df_merged.shape[0] == df_hr_clean.shape[0], "Data loss detected!"
print(f"  Split/merge verified: {df_merged.shape}, no data loss.")

# Task 31-32: Min-max & z-score
norm_cols = ['satisfaction_level', 'last_evaluation', 'average_montly_hours']

minmax_scaler = MinMaxScaler()
df_hr_clean[[c + '_minmax' for c in norm_cols]] = minmax_scaler.fit_transform(df_hr_clean[norm_cols])

zscore_scaler = StandardScaler()
df_hr_clean[[c + '_zscore' for c in norm_cols]] = zscore_scaler.fit_transform(df_hr_clean[norm_cols])

print("  Min-max normalization applied.")
print("  Z-score standardization applied.")

# Task 32: Box plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

bp1 = df_hr_clean[norm_cols].boxplot(ax=axes[0], patch_artist=True, return_type='dict')
for bp, color in zip(bp1['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
    bp.set_facecolor(color)
axes[0].set_title('Original Data', fontweight='bold')
axes[0].set_ylabel('Value')
axes[0].tick_params(axis='x', rotation=30)

bp2 = df_hr_clean[[c + '_minmax' for c in norm_cols]].boxplot(ax=axes[1], patch_artist=True, return_type='dict')
for bp, color in zip(bp2['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
    bp.set_facecolor(color)
axes[1].set_title('Min-Max Normalized [0,1]', fontweight='bold')
axes[1].set_ylabel('Value')
axes[1].tick_params(axis='x', rotation=30)

bp3 = df_hr_clean[[c + '_zscore' for c in norm_cols]].boxplot(ax=axes[2], patch_artist=True, return_type='dict')
for bp, color in zip(bp3['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
    bp.set_facecolor(color)
axes[2].set_title('Z-Score Standardized', fontweight='bold')
axes[2].set_ylabel('Value')
axes[2].tick_params(axis='x', rotation=30)

plt.suptitle('Box Plots: Original vs Min-Max vs Z-Score', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('partB_boxplots_comparison')

# Task 33: Discretization
bin_labels = ['Unsatisfactory', 'Average', 'Good', 'Excellent']
df_hr_clean['last_eval_bin'] = pd.cut(df_hr_clean['last_evaluation'], bins=4, labels=bin_labels)
bin_counts = df_hr_clean['last_eval_bin'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_hr_clean['last_evaluation'], bins=50, edgecolor='black', color='orchid', alpha=0.8)
axes[0].set_title('Original last_evaluation Distribution', fontweight='bold')
axes[0].set_xlabel('Last Evaluation Score')
axes[0].set_ylabel('Frequency')

colors = {'Unsatisfactory': '#e74c3c', 'Average': '#f39c12', 'Good': '#3498db', 'Excellent': '#2ecc71'}
bar_data = bin_counts.reset_index()
bar_data.columns = ['Bin', 'Count']
bar_colors = [colors[bin_name] for bin_name in bar_data['Bin']]
axes[1].bar(bar_data['Bin'], bar_data['Count'], color=bar_colors, edgecolor='black', alpha=0.8)
axes[1].set_title('Discretized Bins (Equal-Width, 4 bins)', fontweight='bold')
axes[1].set_xlabel('Performance Rating')
axes[1].set_ylabel('Employee Count')

plt.tight_layout()
save_fig('partB_discretization')

print(f"  Discretization bins: {dict(zip(bin_counts.index.astype(str), bin_counts.values))}")

# ============================================================
# PART C — DIMENSIONALITY REDUCTION
# ============================================================
print("\n[4/8] Part C — Dimensionality Reduction on Sonar Data...")

# Task 34: Separate features & scale
X = df_sonar.iloc[:, :-1]
y = df_sonar.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Features: {X_scaled.shape}")
print(f"  Class distribution: {dict(y.value_counts())}")

# Task 35: PCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_90 = int(np.argmax(cumulative_var >= 0.90) + 1)

print(f"  Components for 90% variance: {n_components_90}")
print(f"  Cumulative variance: {cumulative_var[n_components_90-1]:.4f}")

# PCA explained variance plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].bar(range(1, 21), explained_var[:20], color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Explained Variance Ratio per Component (Top 20)', fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(range(1, 61), cumulative_var, marker='o', markersize=3, color='darkorange', linewidth=2)
axes[1].axhline(0.90, color='red', linestyle='--', linewidth=2, label='90% threshold')
axes[1].axvline(n_components_90, color='green', linestyle='--', linewidth=2, label=f'{n_components_90} components')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
save_fig('partC_pca_explained_variance')

# Task 36: Reduce to optimal components
pca_optimal = PCA(n_components=n_components_90)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)
print(f"  Reduced shape: {X_pca_optimal.shape} (from {X_scaled.shape})")

pca_columns = [f'PC{i}' for i in range(1, n_components_90 + 1)]

# Task 37: 2D PCA scatter
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

colors_map = {'R': '#3498db', 'M': '#e74c3c'}
labels_map = {'R': 'Rock', 'M': 'Mine'}

fig, ax = plt.subplots(figsize=(10, 7))
for cls, color in colors_map.items():
    mask = y == cls
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               c=color, label=labels_map[cls], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

ax.set_title(f'2D PCA Scatter Plot (PC1 + PC2 = {pca_2d.explained_variance_ratio_.sum():.1%} variance)',
             fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
ax.legend(title='Class')
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig('partC_pca_2d_scatter')

# Task 38: PCA vs t-SNE
print("\n[5/8] Running t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto', init='pca')
X_tsne_2d = tsne.fit_transform(X_scaled)
print(f"  t-SNE KL Divergence: {tsne.kl_divergence_:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for cls, color in colors_map.items():
    mask = y == cls
    axes[0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    c=color, label=labels_map[cls], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_title(f'PCA (Variance: {pca_2d.explained_variance_ratio_.sum():.1%})', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
axes[0].legend()
axes[0].grid(alpha=0.3)

for cls, color in colors_map.items():
    mask = y == cls
    axes[1].scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
                    c=color, label=labels_map[cls], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
axes[1].set_title('t-SNE (Non-linear Embedding)', fontweight='bold')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('PCA vs t-SNE: Class Separability Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('partC_pca_vs_tsne')

# ============================================================
# PART D — EXPORT FOR POWER BI
# ============================================================
print("\n[6/8] Part D — Exporting data for Power BI...")

df_export = pd.DataFrame(X_pca_optimal, columns=pca_columns)
df_export['Class'] = y.values
df_export['Class_Label'] = df_export['Class'].map({'R': 'Rock', 'M': 'Mine'})

# Loadings
loadings = pca_optimal.components_.T
loading_df = pd.DataFrame(loadings, columns=[f'PC{i}' for i in range(1, n_components_90 + 1)])
loading_df['Original_Feature'] = X.columns

top5_loadings = loading_df.nlargest(5, 'PC1')
top5_abs = top5_loadings[['Original_Feature', 'PC1', 'PC2']].copy()
top5_abs['PC1_Magnitude'] = top5_abs['PC1'].abs()

print(f"  Top 5 PC1 loadings:")
for _, row in top5_loadings.iterrows():
    print(f"    {row['Original_Feature']}: PC1={row['PC1']:.4f}")

# Export CSVs
export_file = os.path.join(OUTPUT_DIR, 'sonar_pca_reduced.csv')
df_export.to_csv(export_file, index=False)
print(f"  Saved: {export_file}")

loadings_export = loading_df[['Original_Feature'] + pca_columns].copy()
loadings_export.to_csv(os.path.join(OUTPUT_DIR, 'sonar_pca_loadings.csv'), index=False)
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'sonar_pca_loadings.csv')}")

# Class balance chart
before_pca = df_sonar['Class'].value_counts()
after_pca = df_export['Class'].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
x = ['Before PCA', 'After PCA']
rock_counts = [before_pca.get('R', 0), after_pca.get('R', 0)]
mine_counts = [before_pca.get('M', 0), after_pca.get('M', 0)]
total = rock_counts[0] + mine_counts[0]
rock_pct = [r/total*100 for r in rock_counts]
mine_pct = [m/total*100 for m in mine_counts]

ax.bar(x, rock_pct, label='Rock', color='#3498db', edgecolor='black', alpha=0.8)
ax.bar(x, mine_pct, bottom=rock_pct, label='Mine', color='#e74c3c', edgecolor='black', alpha=0.8)

for i, (r, m) in enumerate(zip(rock_pct, mine_pct)):
    ax.text(i, r/2, f'{r:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    ax.text(i, r + m/2, f'{m:.1f}%', ha='center', va='center', fontweight='bold', color='white')

ax.set_title('Class Distribution: Rock vs Mine (Before & After PCA)', fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
save_fig('partD_class_balance')

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ASSIGNMENT 4 — COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for f in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  {f:<40} ({size/1024:.1f} KB)")

print("\nKey Results:")
print(f"  HR Dataset: {df_hr_clean.shape[0]} rows, {df_hr_clean.shape[1]} cols")
print(f"  Duplicates removed: {removed}")
print(f"  Total outliers (IQR): {total_outliers}")
print(f"  PCA: 60 -> {n_components_90} components ({cumulative_var[n_components_90-1]:.1%} variance)")
print(f"  t-SNE KL Divergence: {tsne.kl_divergence_:.2f}")
print(f"  Class balance: Rock={before_pca.get('R',0)} ({before_pca.get('R',0)/total*100:.1f}%), Mine={before_pca.get('M',0)} ({before_pca.get('M',0)/total*100:.1f}%)")
print("\nDone!")
