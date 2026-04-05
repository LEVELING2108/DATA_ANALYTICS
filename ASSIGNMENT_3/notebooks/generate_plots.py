"""
Assignment 3 - Plot Generation Script
Generates all visualizations for the assignment.

Run from the notebooks/ directory:
    python generate_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import os

# Create outputs directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.dpi'] = 150

print('='*70)
print('ASSIGNMENT 3 - GENERATING PLOTS')
print('='*70)

# =============================================================================
# Load Data
# =============================================================================
wine_df = pd.read_csv('../data/wine.csv')
nutrient_df = pd.read_csv('../data/nutrient.csv')

selected_columns = ['Alcohol', 'Malic', 'Ash', 'Magnesium', 'Phenols', 'Flavanoids']
wine_subset = wine_df[selected_columns].head(10)

print('\n[1/6] Generating Euclidean Distance Heatmap...')

# =============================================================================
# Plot 1: Euclidean Distance Heatmap
# =============================================================================
euclidean_dist = cdist(wine_subset.values, wine_subset.values, metric='euclidean')
euclidean_df = pd.DataFrame(
    euclidean_dist,
    index=[f'Wine {i}' for i in range(1, 11)],
    columns=[f'Wine {i}' for i in range(1, 11)]
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(euclidean_df, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Euclidean Distance'}, square=True,
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_title('Euclidean Dissimilarity Matrix - Wine Samples', fontsize=14, fontweight='bold')
ax.set_xlabel('Wine Samples', fontsize=12)
ax.set_ylabel('Wine Samples', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'partA_euclidean_heatmap.png'), bbox_inches='tight')
plt.close()
print('  [OK] Saved: partA_euclidean_heatmap.png')

# =============================================================================
# Plot 2: Manhattan Distance Heatmap
# =============================================================================
print('\n[2/6] Generating Manhattan Distance Heatmap...')

manhattan_dist = cdist(wine_subset.values, wine_subset.values, metric='cityblock')
manhattan_df = pd.DataFrame(
    manhattan_dist,
    index=[f'Wine {i}' for i in range(1, 11)],
    columns=[f'Wine {i}' for i in range(1, 11)]
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(manhattan_df, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Manhattan Distance'}, square=True,
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_title('Manhattan (L1) Dissimilarity Matrix - Wine Samples', fontsize=14, fontweight='bold')
ax.set_xlabel('Wine Samples', fontsize=12)
ax.set_ylabel('Wine Samples', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'partA_manhattan_heatmap.png'), bbox_inches='tight')
plt.close()
print('  [OK] Saved: partA_manhattan_heatmap.png')

# =============================================================================
# Plot 3: Side-by-Side Comparison
# =============================================================================
print('\n[3/6] Generating Comparison Plot...')

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(euclidean_df, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Euclidean Distance'}, square=True,
            linewidths=0.5, linecolor='white', ax=axes[0])
axes[0].set_title('Euclidean Distance (L2)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Wine Samples')
axes[0].set_ylabel('Wine Samples')

sns.heatmap(manhattan_df, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Manhattan Distance'}, square=True,
            linewidths=0.5, linecolor='white', ax=axes[1])
axes[1].set_title('Manhattan Distance (L1)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Wine Samples')
axes[1].set_ylabel('Wine Samples')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'partA_comparison_heatmaps.png'), bbox_inches='tight')
plt.close()
print('  [OK] Saved: partA_comparison_heatmaps.png')

# =============================================================================
# Plot 4: Minkowski Distance Line Chart
# =============================================================================
print('\n[4/6] Generating Minkowski Distance Chart...')

def minkowski_distance(x, y, p):
    return np.power(np.sum(np.abs(x - y) ** p), 1/p)

wine_1 = wine_subset.iloc[0].values
wine_2 = wine_subset.iloc[4].values
wine_3 = wine_subset.iloc[8].values

p_values = [1, 2, 3, 10]
pairs = [
    ('Wine 1 vs Wine 5', wine_1, wine_2),
    ('Wine 1 vs Wine 9', wine_1, wine_3),
    ('Wine 5 vs Wine 9', wine_2, wine_3)
]

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']

for idx, (pair_name, x, y) in enumerate(pairs):
    distances = [minkowski_distance(x, y, p) for p in p_values]
    ax.plot(p_values, distances, marker=markers[idx], linewidth=2,
            markersize=8, label=pair_name, color=colors[idx])

ax.set_xlabel('Minkowski Parameter (p)', fontsize=12)
ax.set_ylabel('Minkowski Distance', fontsize=12)
ax.set_title('Minkowski Distance vs. Parameter p', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(p_values)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'partB_minkowski_chart.png'), bbox_inches='tight')
plt.close()
print('  [OK] Saved: partB_minkowski_chart.png')

# =============================================================================
# Plot 5: Nutrient Data - Energy and Fat Groups
# =============================================================================
print('\n[5/6] Generating Nutrient Groups Chart...')

# Categorize
energy_quantiles = pd.qcut(nutrient_df['energy'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
nutrient_df['energy_group'] = energy_quantiles
fat_quantiles = pd.qcut(nutrient_df['fat'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
nutrient_df['fat_group'] = fat_quantiles

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Energy groups
energy_counts = nutrient_df['energy_group'].value_counts().sort_index()
colors_energy = ['#ff9999', '#ffcc66', '#66cc66']
axes[0].bar(energy_counts.index.astype(str), energy_counts.values, color=colors_energy, edgecolor='black', alpha=0.8)
axes[0].set_title('Energy Group Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Energy Group', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Fat groups
fat_counts = nutrient_df['fat_group'].value_counts().sort_index()
axes[1].bar(fat_counts.index.astype(str), fat_counts.values, color=colors_energy, edgecolor='black', alpha=0.8)
axes[1].set_title('Fat Group Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fat Group', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'partC_nutrient_groups.png'), bbox_inches='tight')
plt.close()
print('  [OK] Saved: partC_nutrient_groups.png')

# =============================================================================
# Plot 6: Binary Attributes Comparison
# =============================================================================
print('\n[6/6] Generating Binary Attributes Chart...')

median_protein = nutrient_df['protein'].median()
median_iron = nutrient_df['iron'].median()
nutrient_df['high_protein'] = (nutrient_df['protein'] > median_protein).astype(int)
nutrient_df['high_iron'] = (nutrient_df['iron'] > median_iron).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# High protein distribution
protein_counts = nutrient_df['high_protein'].value_counts().sort_index()
axes[0].bar(['Not High', 'High'], protein_counts.values, color=['#ff9999', '#66b3ff'], edgecolor='black', alpha=0.8)
axes[0].set_title('High Protein Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Protein Level', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(True, alpha=0.3)

# High iron distribution
iron_counts = nutrient_df['high_iron'].value_counts().sort_index()
axes[1].bar(['Not High', 'High'], iron_counts.values, color=['#ff9999', '#66b3ff'], edgecolor='black', alpha=0.8)
axes[1].set_title('High Iron Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Iron Level', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'partC_binary_attributes.png'), bbox_inches='tight')
plt.close()
print('  [OK] Saved: partC_binary_attributes.png')

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*70)
print('PLOT GENERATION COMPLETE')
print('='*70)
print(f'\nAll plots saved to: {OUTPUT_DIR}')
print('\nGenerated files:')
for file in sorted(os.listdir(OUTPUT_DIR)):
    if file.endswith('.png'):
        filepath = os.path.join(OUTPUT_DIR, file)
        size = os.path.getsize(filepath)
        print(f'  - {file} ({size:,} bytes)')

print('\n[SUCCESS] All plots generated successfully!')
