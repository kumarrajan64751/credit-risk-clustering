import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib

# Load the dataset
df = pd.read_csv('german_credit_data.csv', index_col=0)


df.fillna({
    'Saving accounts': 'unknown',
    'Checking account': 'unknown'
}, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)


categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']
df_numeric = df[numeric_cols]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)


joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')


df.to_csv('clustered_german_credit_data.csv', index=False)

print("✅ Model and preprocessing objects saved!")
