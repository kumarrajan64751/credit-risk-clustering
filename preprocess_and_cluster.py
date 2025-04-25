import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv('german_credit_data.csv')


df.fillna({
    'Saving accounts': 'unknown',
    'Checking account': 'unknown'
}, inplace=True)

df['Age'].fillna(df['Age'].median(), inplace=True)


categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


features = df.drop(columns=['Cluster', 'PCA1', 'PCA2'])  # Drop non-feature columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]


joblib.dump(pca, 'pca_model.pkl')


plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2')
plt.title('Customer Segments using KMeans Clustering')
plt.show()

# Save processed data with cluster labels
df.to_csv('clustered_credit_data.csv', index=False)

# Save the KMeans model and the scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Clustering complete. Output saved as 'clustered_credit_data.csv'")

score = silhouette_score(scaled_features, kmeans.labels_)
print(f"Silhouette Score: {score:.4f}")

print("✅ Clustering complete. Output saved as 'clustered_credit_data.csv'")
