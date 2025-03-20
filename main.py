import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

#===================================================================
# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("dhoogla/cicids2017")

# print("Path to dataset files:", path)
#===================================================================

# ensures path is in right format
dataset_path = os.path.normpath(r'cicids2017\versions\3')

# List all .parquet files in the folder
parquet_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.parquet')]

# Check if any .parquet files were found
if not parquet_files:
    raise FileNotFoundError("No .parquet files found in the specified folder.")

# Read and concatenate all .parquet files into a single DataFrame
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

# # Display the first few rows of the combined DataFrame
# print(df.head())

# # Check the shape of the combined DataFrame
# print(f"Shape of the combined DataFrame: {df.shape}")

#===================================================================
# Data Representation

df.info()
print("\nDataset Shape:", df.shape)

print("\nColumns:", df.columns.tolist())

df.head()
#===================================================================

#===================================================================
# Data processing

print("\nMissing Values:")
print(df.isnull().sum())

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print("\nCategorical Columns:", categorical_columns.tolist())

for col in categorical_columns:
    print(f"\nColumn: {col}")
    print(df[col].unique())

df = df.dropna()
#===================================================================

#===================================================================
# Normalize Features and Apply PCA

scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

pca = PCA(n_components=2)
pca_features = pca.fit_transform(df[numeric_columns])
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df[categorical_columns[0]])
plt.title("PCA Visualization of CIC-IDS2017 Dataset")
plt.show()
#===================================================================

#===================================================================
# Encoding Categorial Data using One-hot Encoding

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
#===================================================================
# Splitting Data Training:Testing (80:20)

X = df.drop('Label', axis=1)  # "Label" target column for CIC-IDS2017 Dataset
y = df['Label']               # "BENGIN" for normal traffic; "DDos, PortScan, Brute Force, ..." for some types of attacks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#===================================================================

#===================================================================
# Training Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
#===================================================================
