import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
import os

# Set working directory for outputs
output_dir = "ASSIGNMENT_5/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Part A – Data Loading & Initial EDA
# 39. Load Satellite.csv from Google Drive
import io
import requests

url = "https://drive.google.com/uc?export=download&id=1ykwDH-9nGWR-8pIilhiD7PcIhvIDy3OF"
response = requests.get(url)
content = response.content.decode('utf-8')
lines = content.splitlines()

# Clean quotes and split by semicolon
data = []
for line in lines:
    row = line.strip().replace('"', '').split(';')
    data.append(row)

# The first row is columns
df = pd.DataFrame(data[1:], columns=data[0])

# Ensure all columns are present and clean
df.columns = [c.strip() for c in df.columns]

# Check columns
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Ensure numeric features are numeric
for col in df.columns:
    if col != 'classes':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where 'classes' might be empty
df = df.dropna(subset=['classes'])
df = df[df['classes'] != '']

# Report shape, data types, and class distribution
print(f"Final Dataset Shape: {df.shape}")
print(df.dtypes)
class_dist = df['classes'].value_counts()
print("\nClass Distribution:")
print(class_dist)
print(df.dtypes)
class_dist = df['classes'].value_counts()
print("\nClass Distribution:")
print(class_dist)

# 40. Plot a pie chart and a bar chart
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
class_dist.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Class Distribution')

plt.subplot(1, 2, 2)
class_dist.plot(kind='bar')
plt.title('Bar Chart of Class Distribution')
plt.xlabel('Land-use Type')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{output_dir}/partA_class_distribution.png")
plt.close()

# 41. Check for missing values
missing_values = df.isnull().sum().sum()
print(f"\nMissing values: {missing_values}")

# Part B – Clustering
# 42. Scale the 36 spectral features
features = df.drop('classes', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 43. Elbow Method (plot inertia for k = 2 to 10)
inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig(f"{output_dir}/partB_elbow_method.png")
plt.close()

# 44. Fit K-Means with optimal k (assuming k=6 or 7 based on true classes, but looking at elbow)
# The true classes are: 1, 2, 3, 4, 5, 7 (6 classes). Let's use k=6 as a starting point if the elbow isn't sharp.
# For the sake of the assignment, let's pick k=6.
optimal_k = 6 
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title(f'K-Means Clustering (k={optimal_k}) - First 2 Features')
plt.xlabel('Feature x1 (scaled)')
plt.ylabel('Feature x2 (scaled)')
plt.colorbar(label='Cluster')
plt.savefig(f"{output_dir}/partB_kmeans_scatter.png")
plt.close()

# 45. Compute Silhouette Score for chosen k and compare with k=3 and k=7
score_opt = silhouette_score(features_scaled, clusters)
score_3 = silhouette_score(features_scaled, KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(features_scaled))
score_7 = silhouette_score(features_scaled, KMeans(n_clusters=7, random_state=42, n_init=10).fit_predict(features_scaled))

with open(f"{output_dir}/partB_silhouette_scores.txt", "w") as f:
    f.write(f"Silhouette Score (k={optimal_k}): {score_opt}\n")
    f.write(f"Silhouette Score (k=3): {score_3}\n")
    f.write(f"Silhouette Score (k=7): {score_7}\n")

# 46. Cross-tabulate
cross_tab = pd.crosstab(df['classes'], clusters, rownames=['True'], colnames=['Cluster'])
cross_tab.to_csv(f"{output_dir}/partB_cross_tabulation.csv")

# Part C – Classification & Outlier Detection
# 47. Split and Train Decision Tree
X_train, X_test, y_train, y_test = train_test_split(features, df['classes'], test_size=0.20, stratify=df['classes'], random_state=42)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{output_dir}/partC_classification_report.csv")

# 48. Confusion Matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dt.classes_, yticklabels=dt.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f"{output_dir}/partC_confusion_matrix.png")
plt.close()

# 49. Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(features)
num_anomalies = (outliers == -1).sum()

plt.figure(figsize=(8, 6))
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c='blue', label='Normal', alpha=0.5)
plt.scatter(features.iloc[outliers == -1, 0], features.iloc[outliers == -1, 1], c='red', label='Anomaly')
plt.title(f'Isolation Forest Outlier Detection ({num_anomalies} anomalies)')
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.legend()
plt.savefig(f"{output_dir}/partC_outliers_scatter.png")
plt.close()

# Part D - Logic for Tableau (Simulated as requested in dash requirements)
# A heat map of average x1 and x2 feature values by land-use class.
heatmap_data = df.groupby('classes')[['x.1', 'x.2']].mean()
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
plt.title('Average x1 and x2 by Land-use Class')
plt.savefig(f"{output_dir}/partD_tableau_heatmap_sim.png")
plt.close()

# Stacked bar chart comparing true class distribution vs K-Means cluster distribution
# This is tricky because clusters aren't true labels, but we can plot the counts.
dist_df = pd.DataFrame({
    'Type': ['True Classes'] * len(class_dist) + ['K-Means Clusters'] * optimal_k,
    'Label': list(class_dist.index) + [f'Cluster {i}' for i in range(optimal_k)],
    'Count': list(class_dist.values) + list(pd.Series(clusters).value_counts().sort_index().values)
})

plt.figure(figsize=(10, 6))
sns.barplot(data=dist_df, x='Label', y='Count', hue='Type')
plt.title('True Class vs K-Means Cluster Distribution')
plt.xticks(rotation=45)
plt.savefig(f"{output_dir}/partD_tableau_bar_sim.png")
plt.close()

print("Analysis complete. Outputs saved to", output_dir)
