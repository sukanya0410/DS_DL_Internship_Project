import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import numpy as np


try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found. Please ensure the file is in the correct directory.")
    print("You can download it manually from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
    exit() 

print("Original Data Head:")
print(df.head())
print("\nData Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())



df = df.rename(columns={'Annual Income (k$)': 'AnnualIncome', 'Spending Score (1-100)': 'SpendingScore'})

numerical_features = ['Age', 'AnnualIncome', 'SpendingScore']
categorical_features = ['Gender']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


wcss = []
X_transformed = preprocessor.fit_transform(df)

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_transformed)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K (Mall Customers)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

silhouette_avg_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_transformed)
    silhouette_avg = silhouette_score(X_transformed, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_avg_scores, marker='o', linestyle='--')
plt.title('Silhouette Score for Optimal K (Mall Customers)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


optimal_k = 5


kmeans_final = Pipeline(steps=[('preprocessor', preprocessor),
                               ('kmeans', KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10))])
kmeans_final.fit(df)
df['Cluster'] = kmeans_final.named_steps['kmeans'].labels_

print(f"\nCustomer Segmentation with {optimal_k} Clusters:")
print(df.head())
print("\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

cluster_profiles_num = df.groupby('Cluster')[numerical_features].mean()
print("\nNumerical Feature Averages per Cluster:")
print(cluster_profiles_num)

for col in categorical_features:
    print(f"\n{col} Distribution per Cluster:")
    print(df.groupby('Cluster')[col].value_counts(normalize=True).unstack(fill_value=0))

plt.figure(figsize=(12, 8))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(15, 5))
for i, feature in enumerate(numerical_features):
    plt.subplot(1, len(numerical_features), i + 1)
    sns.boxplot(x='Cluster', y=feature, data=df, palette='viridis')
    plt.title(f'{feature} by Cluster')
plt.tight_layout()
plt.show()

print("\n--- Example Actionable Insights for Mall Customers ---")
print("Based on the cluster profiles (e.g., for K=5):")
print("Cluster 0: Average Age, Average Income, Average Spending Score -> 'Balanced Customers'")
print("   - Strategy: General promotions, loyalty programs.")
print("Cluster 1: High Annual Income, Low Spending Score -> 'Careful Spenders / High Income Low Engagement'")
print("   - Strategy: Targeted marketing for premium products, exclusive events to encourage spending.")
print("Cluster 2: Low Annual Income, High Spending Score -> 'Budget-Conscious Spenders / Young Enthusiasts'")
print("   - Strategy: Value-for-money offers, discounts, engaging online content.")
print("Cluster 3: Low Annual Income, Low Spending Score -> 'Frugal Customers'")
print("   - Strategy: Entry-level products, basic necessities, focus on retention with essential offers.")
print("Cluster 4: High Annual Income, High Spending Score -> 'Target Customers / VIPs'")
print("   - Strategy: Personalized recommendations, VIP services, early access to new collections.")