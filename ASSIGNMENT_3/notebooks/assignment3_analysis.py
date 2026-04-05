"""
Assignment 3 - Measuring Data Similarity & Dissimilarity
Main Analysis Script

This script computes all distance matrices, similarity measures, and saves results.
Run from the notebooks/ directory:
    python assignment3_analysis.py
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

# Create outputs directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*70)
print('ASSIGNMENT 3 - MEASURING DATA SIMILARITY & DISSIMILARITY')
print('='*70)

# =============================================================================
# PART A - Data Matrix and Dissimilarity Matrices
# =============================================================================
print('\n' + '='*70)
print('PART A - Data Matrix vs. Dissimilarity Matrix on Wine Data')
print('='*70)

# Load wine dataset
wine_df = pd.read_csv('../data/wine.csv')
selected_columns = ['Alcohol', 'Malic', 'Ash', 'Magnesium', 'Phenols', 'Flavanoids']
wine_subset = wine_df[selected_columns].head(10)

print(f'\nLoaded wine dataset: {wine_df.shape}')
print(f'Selected subset: {wine_subset.shape}')

# Task 19: Display data matrix
data_matrix = wine_subset.copy()
data_matrix.index = range(1, 11)
print('\n=== 10 x 6 Data Matrix ===')
print(data_matrix.round(2))

# Task 20: Euclidean distance matrix
euclidean_dist = cdist(wine_subset.values, wine_subset.values, metric='euclidean')
euclidean_df = pd.DataFrame(
    euclidean_dist,
    index=[f'Wine {i}' for i in range(1, 11)],
    columns=[f'Wine {i}' for i in range(1, 11)]
)

print('\n=== Euclidean Dissimilarity Matrix ===')
print(euclidean_df.round(3))

# Save Euclidean matrix
euclidean_df.round(3).to_csv(os.path.join(OUTPUT_DIR, 'partA_euclidean_distance.csv'))

# Task 21: Manhattan distance matrix
manhattan_dist = cdist(wine_subset.values, wine_subset.values, metric='cityblock')
manhattan_df = pd.DataFrame(
    manhattan_dist,
    index=[f'Wine {i}' for i in range(1, 11)],
    columns=[f'Wine {i}' for i in range(1, 11)]
)

print('\n=== Manhattan (L1) Dissimilarity Matrix ===')
print(manhattan_df.round(3))

# Save Manhattan matrix
manhattan_df.round(3).to_csv(os.path.join(OUTPUT_DIR, 'partA_manhattan_distance.csv'))

# Comparison statistics
upper_tri_euclidean = euclidean_dist[np.triu_indices_from(euclidean_dist, k=1)]
upper_tri_manhattan = manhattan_dist[np.triu_indices_from(manhattan_dist, k=1)]

print('\n=== Comparison Statistics ===')
print(f'Euclidean  - Mean: {upper_tri_euclidean.mean():.3f}, Max: {upper_tri_euclidean.max():.3f}')
print(f'Manhattan  - Mean: {upper_tri_manhattan.mean():.3f}, Max: {upper_tri_manhattan.max():.3f}')
print(f'Ratio (Manhattan/Euclidean): {upper_tri_manhattan.mean() / upper_tri_euclidean.mean():.3f}')

# Save comparison stats
comparison_stats = pd.DataFrame({
    'Metric': ['Euclidean', 'Manhattan'],
    'Mean': [upper_tri_euclidean.mean(), upper_tri_manhattan.mean()],
    'Max': [upper_tri_euclidean.max(), upper_tri_manhattan.max()],
    'Min': [upper_tri_euclidean.min(), upper_tri_manhattan.min()],
    'Std': [upper_tri_euclidean.std(), upper_tri_manhattan.std()]
})
comparison_stats.to_csv(os.path.join(OUTPUT_DIR, 'partA_comparison_stats.csv'), index=False)

print('\n[OK] Part A complete - distance matrices saved to outputs/')


# =============================================================================
# PART B - Minkowski Distance Experiment
# =============================================================================
print('\n' + '='*70)
print('PART B - Minkowski Distance Experiment')
print('='*70)

# Select three wine samples
wine_1 = wine_subset.iloc[0].values
wine_2 = wine_subset.iloc[4].values
wine_3 = wine_subset.iloc[8].values

def minkowski_distance(x, y, p):
    """Compute Minkowski distance between two vectors for parameter p."""
    return np.power(np.sum(np.abs(x - y) ** p), 1/p)

p_values = [1, 2, 3, 10]
pairs = [
    ('Wine 1 vs Wine 5', wine_1, wine_2),
    ('Wine 1 vs Wine 9', wine_1, wine_3),
    ('Wine 5 vs Wine 9', wine_2, wine_3)
]

# Compute distances
results = {}
for pair_name, x, y in pairs:
    results[pair_name] = []
    for p in p_values:
        dist = minkowski_distance(x, y, p)
        results[pair_name].append(dist)

results_df = pd.DataFrame(results, index=[f'p={p}' for p in p_values])

print('\n=== Minkowski Distance Table ===')
print(results_df.round(4))

# Save results
results_df.round(4).to_csv(os.path.join(OUTPUT_DIR, 'partB_minkowski_distances.csv'))

# Print trends
print('\n=== Minkowski Distance Trends ===')
for pair_name, x, y in pairs:
    dist_p1 = minkowski_distance(x, y, 1)
    dist_p10 = minkowski_distance(x, y, 10)
    print(f'{pair_name}: p=1 -> {dist_p1:.4f}, p=10 -> {dist_p10:.4f}')

print('\n[OK] Part B complete - Minkowski distances saved to outputs/')


# =============================================================================
# PART C - Proximity for Nominal & Binary Attributes
# =============================================================================
print('\n' + '='*70)
print('PART C - Proximity for Nominal & Binary Attributes on Nutrient Data')
print('='*70)

# Load nutrient dataset
nutrient_df = pd.read_csv('../data/nutrient.csv')
print(f'\nLoaded nutrient dataset: {nutrient_df.shape}')

# Task 25: Categorize energy and fat into Low/Medium/High
energy_quantiles = pd.qcut(nutrient_df['energy'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
nutrient_df['energy_group'] = energy_quantiles

fat_quantiles = pd.qcut(nutrient_df['fat'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
nutrient_df['fat_group'] = fat_quantiles

print('\n=== Nutrient Data with Groups ===')
print(nutrient_df[['Food_Item', 'energy', 'energy_group', 'fat', 'fat_group']].to_string())

# Save grouped data
nutrient_df[['Food_Item', 'energy', 'energy_group', 'fat', 'fat_group']].to_csv(
    os.path.join(OUTPUT_DIR, 'partC_nutrient_groups.csv'), index=False
)

# Nominal similarity function
def nominal_similarity(row1, row2, df):
    """Compute simple matching similarity based on nominal attribute matches."""
    item1 = df.iloc[row1]['Food_Item']
    item2 = df.iloc[row2]['Food_Item']
    
    energy1 = df.iloc[row1]['energy_group']
    energy2 = df.iloc[row2]['energy_group']
    
    fat1 = df.iloc[row1]['fat_group']
    fat2 = df.iloc[row2]['fat_group']
    
    energy_match = 1 if energy1 == energy2 else 0
    fat_match = 1 if fat1 == fat2 else 0
    
    total_attributes = 2
    matches = energy_match + fat_match
    similarity = matches / total_attributes
    
    return {
        'item1': item1,
        'item2': item2,
        'energy1': str(energy1),
        'energy2': str(energy2),
        'fat1': str(fat1),
        'fat2': str(fat2),
        'energy_match': energy_match,
        'fat_match': fat_match,
        'total_matches': matches,
        'similarity': similarity
    }

# Calculate nominal similarities
print('\n=== Nominal Similarity Calculations ===')
nominal_pairs = [(0, 1), (0, 2), (1, 5)]
nominal_results = []

for row1, row2 in nominal_pairs:
    result = nominal_similarity(row1, row2, nutrient_df)
    nominal_results.append(result)
    print(f"\n{result['item1']} vs {result['item2']}:")
    print(f"  Energy: {result['energy1']} vs {result['energy2']} -> Match: {result['energy_match']}")
    print(f"  Fat: {result['fat1']} vs {result['fat2']} -> Match: {result['fat_match']}")
    print(f"  Similarity: {result['similarity']:.2f}")

# Save nominal results
nominal_df = pd.DataFrame(nominal_results)
nominal_df.to_csv(os.path.join(OUTPUT_DIR, 'partC_nominal_similarity.csv'), index=False)

# Task 26: Binary attributes
median_protein = nutrient_df['protein'].median()
median_iron = nutrient_df['iron'].median()

nutrient_df['high_protein'] = (nutrient_df['protein'] > median_protein).astype(int)
nutrient_df['high_iron'] = (nutrient_df['iron'] > median_iron).astype(int)

print(f'\n=== Median Values ===')
print(f'Median Protein: {median_protein:.2f}')
print(f'Median Iron: {median_iron:.2f}')

print('\n=== Binary Attributes ===')
print(nutrient_df[['Food_Item', 'protein', 'high_protein', 'iron', 'high_iron']].to_string())

# Save binary attributes
nutrient_df[['Food_Item', 'protein', 'high_protein', 'iron', 'high_iron']].to_csv(
    os.path.join(OUTPUT_DIR, 'partC_binary_attributes.csv'), index=False
)

# Jaccard and SMC functions
def jaccard_similarity(a, b):
    """Compute Jaccard similarity between two binary vectors."""
    M11 = np.sum((a == 1) & (b == 1))
    M01 = np.sum((a == 0) & (b == 1))
    M10 = np.sum((a == 1) & (b == 0))
    
    denominator = M01 + M10 + M11
    if denominator == 0:
        return 0
    return M11 / denominator

def smc(a, b):
    """Compute Simple Matching Coefficient between two binary vectors."""
    M11 = np.sum((a == 1) & (b == 1))
    M00 = np.sum((a == 0) & (b == 0))
    M01 = np.sum((a == 0) & (b == 1))
    M10 = np.sum((a == 1) & (b == 0))
    
    total = M00 + M01 + M10 + M11
    if total == 0:
        return 0
    return (M11 + M00) / total

# Calculate Jaccard and SMC
pair_indices = [(0, 1), (0, 5), (5, 24)]

print('\n=== Jaccard Similarity and SMC ===')
binary_results = []
for idx1, idx2 in pair_indices:
    item1 = nutrient_df.iloc[idx1]['Food_Item']
    item2 = nutrient_df.iloc[idx2]['Food_Item']
    
    vec1 = nutrient_df.iloc[idx1][['high_protein', 'high_iron']].values
    vec2 = nutrient_df.iloc[idx2][['high_protein', 'high_iron']].values
    
    jaccard = jaccard_similarity(vec1, vec2)
    smc_val = smc(vec1, vec2)
    
    M11 = int(np.sum((vec1 == 1) & (vec2 == 1)))
    M00 = int(np.sum((vec1 == 0) & (vec2 == 0)))
    M01 = int(np.sum((vec1 == 0) & (vec2 == 1)))
    M10 = int(np.sum((vec1 == 1) & (vec2 == 0)))
    
    binary_results.append({
        'Pair': f'{item1} vs {item2}',
        'M11': M11,
        'M00': M00,
        'M01': M01,
        'M10': M10,
        'Jaccard': round(jaccard, 4),
        'SMC': round(smc_val, 4)
    })
    
    print(f'\n{item1} vs {item2}:')
    print(f'  Vectors: {vec1} vs {vec2}')
    print(f'  Jaccard: {jaccard:.4f}, SMC: {smc_val:.4f}')

# Save binary results
binary_df = pd.DataFrame(binary_results)
binary_df.to_csv(os.path.join(OUTPUT_DIR, 'partC_jaccard_smc.csv'), index=False)

print('\n[OK] Part C complete - all similarity measures saved to outputs/')


# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*70)
print('ASSIGNMENT 3 - ANALYSIS COMPLETE')
print('='*70)
print(f'\nAll outputs saved to: {OUTPUT_DIR}')
print('\nGenerated files:')
for file in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, file)
    size = os.path.getsize(filepath)
    print(f'  - {file} ({size:,} bytes)')

print('\n[SUCCESS] Assignment 3 analysis complete!')
