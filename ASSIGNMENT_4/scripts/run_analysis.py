"""
Assignment 4 — Data Pre-processing Pipeline
Run this script to generate all outputs (individual plots + CSVs)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")
    plt.close()

print("=" * 70)
print("Assignment 4 — Data Pre-processing Pipeline")
print("=" * 70)

# ============================================================
# LOAD / GENERATE DATASETS
# ============================================================
print("\n[1/8] Loading datasets...")

if not os.path.exists('HR_comma_sep.csv'):
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
    print("  Done.")
else:
    print("  HR_comma_sep.csv found.")

if not os.path.exists('Sonar.csv'):
    print("  Generating synthetic Sonar-like dataset...")
    np.random.seed(42)
    n = 208
    data = {f'V{i}': np.random.uniform(0, 1, n) for i in range(1, 61)}
    data['Class'] = ['R'] * 111 + ['M'] * 97
    np.random.shuffle(data['Class'])
    df_sonar = pd.DataFrame(data)
    df_sonar.to_csv('Sonar.csv', index=False)
    print("  Done.")
else:
    print("  Sonar.csv found.")

df_hr = pd.read_csv("HR_comma_sep.csv")
df_hr.columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                  'average_montly_hours', 'time_spend_company', 'Work_accident',
                  'left', 'promotion_last_5years', 'Department', 'salary']
df_sonar = pd.read_csv('Sonar.csv')

print(f"  HR dataset: {df_hr.shape}")
print(f"  Sonar dataset: {df_sonar.shape}")

# ============================================================
# PART A — DATA CLEANING
# ============================================================
print("\n[2/8] Part A — Data Cleaning...")

# Task 27
missing_total = df_hr.isnull().sum().sum()
duplicates = df_hr.duplicated().sum()

numeric_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company']

outlier_report = []
total_outliers = 0
for col in numeric_cols:
    Q1 = df_hr[col].quantile(0.25)
    Q3 = df_hr[col].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    oc = len(df_hr[(df_hr[col] < lb) | (df_hr[col] > ub)])
    total_outliers += oc
    outlier_report.append({'Column': col, 'Q1': round(Q1,3), 'Q3': round(Q3,3),
                           'IQR': round(IQR,3), 'Lower': round(lb,3), 'Upper': round(ub,3), 'Outliers': oc})

print(f"  Missing: {missing_total} | Duplicates: {duplicates} | Outliers: {total_outliers}")

# Task 28: Winsorization
df_hr_clean = df_hr.drop_duplicates().copy()
removed = len(df_hr) - len(df_hr_clean)
lower_cap = df_hr_clean['average_montly_hours'].quantile(0.05)
upper_cap = df_hr_clean['average_montly_hours'].quantile(0.95)
df_hr_clean['average_montly_hours_winsorized'] = df_hr_clean['average_montly_hours'].clip(lower=lower_cap, upper=upper_cap)
print(f"  Winsorization: [{lower_cap:.0f}, {upper_cap:.0f}]")

# --- INDIVIDUAL PLOT: Before Winsorization ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_hr_clean['average_montly_hours'], bins=50, edgecolor='black', color='salmon', alpha=0.85)
ax.axvline(lower_cap, color='red', linestyle='--', linewidth=2, label=f'5th %ile ({lower_cap:.0f})')
ax.axvline(upper_cap, color='red', linestyle='--', linewidth=2, label=f'95th %ile ({upper_cap:.0f})')
ax.set_title('Average Monthly Hours — Before Winsorization', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Monthly Hours')
ax.set_ylabel('Frequency')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
save_fig('partA_winsorization_before')

# --- INDIVIDUAL PLOT: After Winsorization ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_hr_clean['average_montly_hours_winsorized'], bins=50, edgecolor='black', color='teal', alpha=0.85)
ax.axvline(lower_cap, color='red', linestyle='--', linewidth=2, label=f'5th %ile ({lower_cap:.0f})')
ax.axvline(upper_cap, color='red', linestyle='--', linewidth=2, label=f'95th %ile ({upper_cap:.0f})')
ax.set_title('Average Monthly Hours — After Winsorization (5th-95th %ile)', fontsize=14, fontweight='bold')
ax.set_xlabel('Average Monthly Hours')
ax.set_ylabel('Frequency')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
save_fig('partA_winsorization_after')

# Task 29: Data types
binary_cols = ['Work_accident', 'left', 'promotion_last_5years']
for col in binary_cols:
    if df_hr_clean[col].dtype != 'int64' and df_hr_clean[col].dtype != 'int32':
        df_hr_clean[col] = df_hr_clean[col].astype(int)
df_hr_clean['Department'] = df_hr_clean['Department'].astype('category')
df_hr_clean['salary'] = df_hr_clean['salary'].astype('category')
print("  Data types validated.")

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
df_merged = pd.merge(df_half1, df_half2, on='_merge_idx', how='inner').drop('_merge_idx', axis=1)
assert df_merged.shape[0] == df_hr_clean.shape[0]
print(f"  Split/merge verified: {df_merged.shape}")

# Task 31-32: Normalize & standardize
norm_cols = ['satisfaction_level', 'last_evaluation', 'average_montly_hours']
minmax_scaler = MinMaxScaler()
df_hr_clean[[c + '_minmax' for c in norm_cols]] = minmax_scaler.fit_transform(df_hr_clean[norm_cols])
zscore_scaler = StandardScaler()
df_hr_clean[[c + '_zscore' for c in norm_cols]] = zscore_scaler.fit_transform(df_hr_clean[norm_cols])

# --- INDIVIDUAL PLOT: Boxplot Original ---
fig, ax = plt.subplots(figsize=(8, 6))
bp1 = df_hr_clean[norm_cols].boxplot(ax=ax, patch_artist=True, return_type='dict')
for bp, color in zip(bp1['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
    bp.set_facecolor(color)
ax.set_title('Box Plot — Original Data', fontsize=14, fontweight='bold')
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)
save_fig('partB_boxplot_original')

# --- INDIVIDUAL PLOT: Boxplot Min-Max ---
fig, ax = plt.subplots(figsize=(8, 6))
bp2 = df_hr_clean[[c + '_minmax' for c in norm_cols]].boxplot(ax=ax, patch_artist=True, return_type='dict')
for bp, color in zip(bp2['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
    bp.set_facecolor(color)
ax.set_title('Box Plot — Min-Max Normalized [0, 1]', fontsize=14, fontweight='bold')
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)
save_fig('partB_boxplot_minmax')

# --- INDIVIDUAL PLOT: Boxplot Z-Score ---
fig, ax = plt.subplots(figsize=(8, 6))
bp3 = df_hr_clean[[c + '_zscore' for c in norm_cols]].boxplot(ax=ax, patch_artist=True, return_type='dict')
for bp, color in zip(bp3['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
    bp.set_facecolor(color)
ax.set_title('Box Plot — Z-Score Standardized', fontsize=14, fontweight='bold')
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)
save_fig('partB_boxplot_zscore')

# Task 33: Discretization
bin_labels = ['Unsatisfactory', 'Average', 'Good', 'Excellent']
df_hr_clean['last_eval_bin'] = pd.cut(df_hr_clean['last_evaluation'], bins=4, labels=bin_labels)
bin_counts = df_hr_clean['last_eval_bin'].value_counts().sort_index()

# --- INDIVIDUAL PLOT: Discretization Histogram ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_hr_clean['last_evaluation'], bins=50, edgecolor='black', color='orchid', alpha=0.85)
ax.set_title('Last Evaluation — Original Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Last Evaluation Score')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha=0.3)
save_fig('partB_discretization_histogram')

# --- INDIVIDUAL PLOT: Discretization Bar Chart ---
colors_map_bins = {'Unsatisfactory': '#e74c3c', 'Average': '#f39c12', 'Good': '#3498db', 'Excellent': '#2ecc71'}
bar_data = bin_counts.reset_index()
bar_data.columns = ['Bin', 'Count']
bar_colors = [colors_map_bins[b] for b in bar_data['Bin']]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(bar_data['Bin'], bar_data['Count'], color=bar_colors, edgecolor='black', alpha=0.85)
for bar_obj, count in zip(bars, bar_data['Count']):
    ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 80,
            str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_title('Equal-Width Discretization — last_evaluation (4 Bins)', fontsize=14, fontweight='bold')
ax.set_xlabel('Performance Rating')
ax.set_ylabel('Employee Count')
ax.grid(axis='y', alpha=0.3)
save_fig('partB_discretization_bins')

print(f"  Bins: {dict(zip(bin_counts.index.astype(str), bin_counts.values))}")

# ============================================================
# PART C — DIMENSIONALITY REDUCTION
# ============================================================
print("\n[4/8] Part C — Dimensionality Reduction...")

# Task 34
X = df_sonar.iloc[:, :-1]
y = df_sonar.iloc[:, -1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Features: {X_scaled.shape} | Classes: {dict(y.value_counts())}")

# Task 35: PCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_90 = int(np.argmax(cumulative_var >= 0.90) + 1)
print(f"  90% variance at {n_components_90} components ({cumulative_var[n_components_90-1]:.4f})")

# --- INDIVIDUAL PLOT: PCA Variance Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, 21), explained_var[:20], color='steelblue', edgecolor='black', alpha=0.85)
ax.set_title('PCA — Explained Variance Ratio per Component (Top 20)', fontsize=14, fontweight='bold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(explained_var[:20]):
    ax.text(i+1, v + 0.001, f'{v:.3f}', ha='center', fontsize=8, rotation=0)
save_fig('partC_pca_variance_bar')

# --- INDIVIDUAL PLOT: PCA Cumulative Variance ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 61), cumulative_var, marker='o', markersize=3, color='darkorange', linewidth=2)
ax.axhline(0.90, color='red', linestyle='--', linewidth=2, label='90% threshold')
ax.axvline(n_components_90, color='green', linestyle='--', linewidth=2, label=f'{n_components_90} components')
ax.set_title('PCA — Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
save_fig('partC_pca_variance_cumulative')

# Task 36: Reduce
pca_optimal = PCA(n_components=n_components_90)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)
pca_columns = [f'PC{i}' for i in range(1, n_components_90 + 1)]
print(f"  Reduced: {X_pca_optimal.shape}")

# Task 37: 2D PCA — already individual
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

colors_class = {'R': '#3498db', 'M': '#e74c3c'}
labels_class = {'R': 'Rock', 'M': 'Mine'}

fig, ax = plt.subplots(figsize=(10, 7))
for cls, color in colors_class.items():
    mask = y == cls
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               c=color, label=labels_class[cls], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
ax.set_title(f'2D PCA Scatter Plot (PC1+PC2 = {pca_2d.explained_variance_ratio_.sum():.1%} variance)',
             fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
ax.legend(title='Class')
ax.grid(alpha=0.3)
save_fig('partC_pca_2d_scatter')

# Task 38: t-SNE
print("\n[5/8] Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto', init='pca')
X_tsne_2d = tsne.fit_transform(X_scaled)
print(f"  t-SNE KL Divergence: {tsne.kl_divergence_:.2f}")

# --- INDIVIDUAL PLOT: PCA Scatter (for comparison) ---
fig, ax = plt.subplots(figsize=(10, 7))
for cls, color in colors_class.items():
    mask = y == cls
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               c=color, label=labels_class[cls], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_title(f'PCA 2D Embedding (Variance: {pca_2d.explained_variance_ratio_.sum():.1%})',
             fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
ax.legend(title='Class')
ax.grid(alpha=0.3)
save_fig('partC_pca_comparison')

# --- INDIVIDUAL PLOT: t-SNE Scatter ---
fig, ax = plt.subplots(figsize=(10, 7))
for cls, color in colors_class.items():
    mask = y == cls
    ax.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
               c=color, label=labels_class[cls], alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_title(f't-SNE 2D Embedding (KL Divergence: {tsne.kl_divergence_:.2f})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(title='Class')
ax.grid(alpha=0.3)
save_fig('partC_tsne_scatter')

# ============================================================
# PART D — EXPORT
# ============================================================
print("\n[6/8] Part D — Exporting...")

df_export = pd.DataFrame(X_pca_optimal, columns=pca_columns)
df_export['Class'] = y.values
df_export['Class_Label'] = df_export['Class'].map({'R': 'Rock', 'M': 'Mine'})

loadings = pca_optimal.components_.T
loading_df = pd.DataFrame(loadings, columns=[f'PC{i}' for i in range(1, n_components_90 + 1)])
loading_df['Original_Feature'] = X.columns
top5_loadings = loading_df.nlargest(5, 'PC1')

print("  Top 5 PC1 loadings:")
for _, row in top5_loadings.iterrows():
    print(f"    {row['Original_Feature']}: {row['PC1']:.4f}")

# Export CSVs
pd.DataFrame(X_pca_optimal, columns=pca_columns).assign(Class=y.values, Class_Label=lambda df: df['Class'].map({'R':'Rock','M':'Mine'})).to_csv(os.path.join(OUTPUT_DIR, 'sonar_pca_reduced.csv'), index=False)
loading_df[['Original_Feature'] + pca_columns].to_csv(os.path.join(OUTPUT_DIR, 'sonar_pca_loadings.csv'), index=False)
print(f"  Saved: sonar_pca_reduced.csv, sonar_pca_loadings.csv")

# --- INDIVIDUAL PLOT: Class Balance ---
before_pca = df_sonar['Class'].value_counts()
after_pca = df_export['Class'].value_counts()
total = before_pca.sum()

fig, ax = plt.subplots(figsize=(8, 5))
x = ['Before PCA', 'After PCA']
rock_pct = [before_pca.get('R',0)/total*100, after_pca.get('R',0)/total*100]
mine_pct = [before_pca.get('M',0)/total*100, after_pca.get('M',0)/total*100]

ax.bar(x, rock_pct, label='Rock', color='#3498db', edgecolor='black', alpha=0.85)
ax.bar(x, mine_pct, bottom=rock_pct, label='Mine', color='#e74c3c', edgecolor='black', alpha=0.85)
for i, (r, m) in enumerate(zip(rock_pct, mine_pct)):
    ax.text(i, r/2, f'{r:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    ax.text(i, r + m/2, f'{m:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
ax.set_title('Class Distribution: Rock vs Mine (Before & After PCA)', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.legend(fontsize=11)
ax.set_ylim(0, 105)
save_fig('partD_class_balance')

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ASSIGNMENT 4 — COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(filepath):
        size = os.path.getsize(filepath)
        print(f"  {f:<50} ({size/1024:.1f} KB)")

print(f"\nKey Results:")
print(f"  HR Dataset: {df_hr_clean.shape}")
print(f"  Duplicates removed: {removed}")
print(f"  Total outliers (IQR): {total_outliers}")
print(f"  PCA: 60 -> {n_components_90} components ({cumulative_var[n_components_90-1]:.1%})")
print(f"  t-SNE KL Divergence: {tsne.kl_divergence_:.2f}")
print(f"  Class balance: Rock={before_pca.get('R',0)} ({before_pca.get('R',0)/total*100:.1f}%), Mine={before_pca.get('M',0)} ({before_pca.get('M',0)/total*100:.1f}%)")
print("\nDone!")
